"""
STAGE compare — one report over every configuration.

  1. Downstream limitation quality per configuration x system (precision / recall / F1 from
     scilimagents_main's LLM-judge evaluation), with 95% bootstrap CIs over papers
  2. Paired differences vs. `baseline` (paired bootstrap over papers, 95% CI, two-sided p),
     Holm-corrected within each system x metric family
  3. Seed variance of the baseline (configs named baseline@seed<n>)
  4. Retrieval quality (IR) per configuration and its link to downstream quality:
     Spearman rho (per configuration, and pooled with configuration fixed effects) and
     F1 by nDCG tercile
  5. Year breakdown (retrieval recall and downstream recall by target publication year)
  6. Cost: generation minutes per paper and retrieved-context size vs F1 (Pareto plot)
Outputs in <work>/report/: RAG_ABLATION_REPORT.txt, *.csv, plots/*.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["f1", "precision", "recall"]


def boot_ci(v, n=2000, seed=0):
    v = np.asarray([x for x in v if not pd.isna(x)], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan,) * 3
    bs = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def paired(a: pd.Series, b: pd.Series, n=2000, seed=0):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(j)
    d = (j.a - j.b).to_numpy()
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), float(p), len(d)


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = np.where(~np.isnan(ps))[0]
    order = ok[np.argsort(ps[ok])]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        out[j] = run
    return out


def spearman(x, y):
    j = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(j) < 5:
        return np.nan, len(j)
    return float(j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank())), len(j)


def load_eval(work):
    rows = []
    for d in sorted((work / "configs").glob("*")):
        ev = d / "out" / "eval"
        if not ev.exists():
            continue
        for f in ev.glob("eval__*.csv"):
            m = re.match(r"eval__(.+?)__(.+)\.csv$", f.name)
            if not m or m.group(1) != m.group(2):
                continue
            col = m.group(1)
            df = pd.read_csv(f)
            keep = ["row_id"] + [f"{col}__{x}" for x in METRICS if f"{col}__{x}" in df.columns]
            x = df[keep].rename(columns={f"{col}__{k}": k for k in METRICS})
            x["config"], x["system"] = d.name, col
            if f"{col}__n_unique_llm" in df.columns:
                x["n_llm"] = df[f"{col}__n_unique_llm"]
            rows.append(x)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_compare(a):
    work = Path(a.work_dir)
    out = work / "report"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    E = load_eval(work)
    L = []
    L.append("=" * 100)
    L.append("RAG ABLATIONS — limitation generation (downstream) + retrieval quality (IR)")
    L.append("=" * 100)
    cfgs = {d.name: json.loads((d / "config.json").read_text()) for d in sorted((work / "configs").glob("*"))
            if (d / "config.json").exists()}
    L.append("Configurations (one factor changed vs baseline):")
    for n, c in cfgs.items():
        L.append(f"  {n:18s} alpha={c['alpha']} query={c['query']} sections={','.join(c['sections'])} k={c['k']} "
                 f"(n={c['n_papers']}) fmt={c['fmt']} embed={c['embed']} temporal={c['temporal']} memory={c['memory']}"
                 f"{' ORIGINAL-COLUMNS' if c.get('original') else ''} seed={c.get('seed', 0)}")
    L.append("")
    if E.empty:
        L.append("No evaluation files yet (run stages generate + eval).")
    else:
        E = E[E.row_id.notna()]
        E["row_id"] = E.row_id.astype(int)
        valid = E.dropna(subset=["f1"])
        if "n_llm" in valid:
            valid = valid[valid.n_llm.fillna(1) > 0]
        # 1. summary
        S = []
        for (c, s), d in valid.groupby(["config", "system"]):
            r = {"config": c, "system": s, "n_papers": len(d)}
            for m in METRICS:
                r[m], r[m + "_lo"], r[m + "_hi"] = boot_ci(d[m], seed=a.seed)
            S.append(r)
        S = pd.DataFrame(S)
        S.to_csv(out / "downstream_summary.csv", index=False)
        piv = S.pivot(index="config", columns="system", values="f1")
        L.append("1. DOWNSTREAM F1 (LLM-judge vs author+reviewer limitations; rows = configuration)")
        L.append((piv * 100).round(2).to_string())
        L.append("   (95% CIs in downstream_summary.csv)")
        L.append("")
        # 2. paired vs baseline + Holm
        P = []
        if "baseline" in set(valid.config):
            for s, ds in valid.groupby("system"):
                b = ds[ds.config == "baseline"].set_index("row_id")
                for c, dc in ds.groupby("config"):
                    if c == "baseline":
                        continue
                    dc = dc.set_index("row_id")
                    for m in METRICS:
                        md, lo, hi, p, n = paired(dc[m], b[m], seed=a.seed)
                        P.append({"system": s, "config": c, "metric": m, "diff": md, "lo": lo, "hi": hi, "p": p,
                                  "n_papers": n})
            P = pd.DataFrame(P)
            if len(P):
                P["p_holm"] = np.nan
                for (s, m), d in P.groupby(["system", "metric"]):
                    P.loc[d.index, "p_holm"] = holm(d.p.to_numpy())
                P["sig_0.05"] = P.p_holm < 0.05
                P.to_csv(out / "paired_vs_baseline.csv", index=False)
                L.append("2. PAIRED DIFFERENCE vs baseline (F1 x100; paired bootstrap 95% CI; Holm-corrected p within system)")
                F = P[P.metric == "f1"].copy()
                for c in ("diff", "lo", "hi"):
                    F[c] = F[c] * 100
                L.append(F[["system", "config", "diff", "lo", "hi", "p", "p_holm", "sig_0.05", "n_papers"]]
                         .round(3).to_string(index=False))
                L.append("")
        else:
            P = pd.DataFrame()
        # 3. seed variance
        seeds = [c for c in set(valid.config) if c == "baseline" or c.startswith("baseline@seed")]
        if len(seeds) > 1:
            V = []
            for s, ds in valid[valid.config.isin(seeds)].groupby("system"):
                w = ds.pivot_table(index="row_id", columns="config", values="f1")
                V.append({"system": s, "n_seeds": w.shape[1], "mean_f1": float(w.mean().mean()),
                          "sd_of_system_mean_f1": float(w.mean().std()),
                          "mean_per_paper_sd": float(w.std(axis=1).mean())})
            V = pd.DataFrame(V)
            V.to_csv(out / "seed_variance.csv", index=False)
            L.append("3. SEED VARIANCE (baseline repeated with different generation seeds)")
            L.append(V.round(4).to_string(index=False))
            L.append("")
    # 4. IR + link
    irp = work / "ir" / "ir_per_paper.csv"
    if irp.exists():
        IR = pd.read_csv(irp)
        IRS = pd.read_csv(work / "ir" / "ir_summary.csv")
        cols = ["config", "papers_with_relevant_in_corpus"] + [c for c in IRS.columns
                                                               if re.match(r"^(P|R|nDCG)@(3|9|15)$", c)] + ["MRR"]
        L.append("4. RETRIEVAL QUALITY (relevance = the paper's own cited papers found in the RAG corpus; paper level)")
        L.append(IRS[[c for c in cols if c in IRS.columns]].round(3).to_string(index=False))
        jc = [c for c in IRS.columns if c.startswith(("jP@", "jnDCG@")) and not c.endswith(("_lo", "_hi"))]
        if jc and IRS[jc].notna().any().any():
            L.append("   LLM-judge graded relevance (judged sample; jP = share rated >= 1):")
            L.append(IRS[IRS[jc].notna().any(axis=1)][["config"] + jc].round(3).to_string(index=False))
        ag = work / "ir" / "ir_judge_agreement.json"
        if ag.exists():
            L.append("   judge vs human: " + ag.read_text().replace("\n", " "))
        L.append("")
        if not E.empty:
            key = a.link_metric if a.link_metric in IR.columns else next(c for c in IR.columns if c.startswith("nDCG@"))
            IR["row_id"] = IR.row_id.astype(int)
            J = valid.merge(IR[["config", "row_id", key, "n_relevant_in_corpus"]], on=["config", "row_id"])
            J = J[J.n_relevant_in_corpus > 0]
            LK = []
            for s, ds in J.groupby("system"):
                for c, dc in ds.groupby("config"):
                    rho, n = spearman(dc[key], dc.f1)
                    LK.append({"system": s, "config": c, "rho": rho, "n": n})
                dm = ds.assign(x=ds[key] - ds.groupby("config")[key].transform("mean"),
                               y=ds.f1 - ds.groupby("config").f1.transform("mean"))
                rho, n = spearman(dm.x, dm.y)
                LK.append({"system": s, "config": "POOLED (config fixed effects)", "rho": rho, "n": n})
            LK = pd.DataFrame(LK)
            LK.to_csv(out / "ir_vs_f1_correlation.csv", index=False)
            L.append(f"5. DOES BETTER RETRIEVAL GIVE BETTER LIMITATIONS?  Spearman rho({key}, F1) per paper")
            L.append(LK[LK.config.str.startswith("POOLED") | (LK.config == "baseline")].round(3).to_string(index=False))
            T = []
            for s, ds in J[J.config == "baseline"].groupby("system"):
                if ds[key].nunique() < 3:
                    continue
                t = pd.qcut(ds[key].rank(method="first"), 3, labels=["low", "mid", "high"])
                for lvl, dd in ds.groupby(t, observed=False):
                    m, lo, hi = boot_ci(dd.f1, seed=a.seed)
                    T.append({"system": s, f"{key}_tercile": lvl, "f1": m, "lo": lo, "hi": hi, "n": len(dd)})
            if T:
                T = pd.DataFrame(T)
                T.to_csv(out / "f1_by_retrieval_tercile.csv", index=False)
                L.append("   baseline F1 by retrieval-quality tercile:")
                L.append(T.round(3).to_string(index=False))
            L.append("")
            # year breakdown
            Y = IR[IR.config == "baseline"][["row_id", "target_year", "n_relevant_in_corpus"] +
                                           [c for c in IR.columns if c.startswith("R@")]]
            if Y.target_year.notna().any():
                JY = valid[valid.config == "baseline"].merge(Y, on="row_id")
                kk = key.split("@")[-1]
                rk = f"R@{kk}" if f"R@{kk}" in IR.columns else next((c for c in IR.columns if c.startswith("R@")), None)
                YB = (JY.groupby(["system", "target_year"])
                      .agg(n=("row_id", "size"), downstream_recall=("recall", "mean"), downstream_f1=("f1", "mean"),
                           ir_recall=(rk, "mean")).reset_index())
                YB.to_csv(out / "by_target_year.csv", index=False)
                L.append(f"6. BY TARGET PUBLICATION YEAR (baseline; ir_recall = {rk} of cited papers)")
                L.append(YB.round(3).to_string(index=False))
                L.append("")
    # 7. cost
    C = []
    for n in cfgs:
        d = work / "configs" / n
        t = json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {}
        inp = d / "input.csv"
        ch = pd.read_csv(inp, usecols=["rag_abl__context_chars"]).rag_abl__context_chars.mean() if inp.exists() else np.nan
        nr = len(pd.read_csv(inp, usecols=["row_id"])) if inp.exists() else np.nan
        C.append({"config": n, "gen_minutes_total": sum(t.values()) / 60 if t else np.nan,
                  "gen_seconds_per_paper": sum(t.values()) / nr if t and nr else np.nan,
                  "retrieved_context_chars_per_paper": ch})
    C = pd.DataFrame(C)
    C.to_csv(out / "cost.csv", index=False)
    L.append("7. COST (generation wall time includes every system in SYSTEMS; context = retrieved text per paper)")
    L.append(C.round(2).to_string(index=False))
    L.append("")
    L.append("NOTES")
    L.append("  * Relevance labels for IR come from each paper's own references that exist in the RAG corpus; papers")
    L.append("    citing nothing in the corpus are excluded from R@k / nDCG@k (coverage is printed in section 4).")
    L.append("  * Each configuration changes ONE factor; compare to `baseline`. `original` uses the retrieval columns")
    L.append("    already in the input CSV (checks that the rebuilt baseline reproduces them).")
    L.append("  * p values: paired bootstrap over papers, Holm-corrected across configurations within system x metric.")
    (out / "RAG_ABLATION_REPORT.txt").write_text("\n".join(str(x) for x in L) + "\n")
    print("\n".join(str(x) for x in L[:80]))
    try:
        plots(out, E if not E.empty else None, P if not E.empty else pd.DataFrame(), C, work, a)
    except Exception as e:  # noqa: BLE001
        print(f"[compare] plots skipped: {e}")
    print(f"[compare] -> {out / 'RAG_ABLATION_REPORT.txt'}")


def plots(out, E, P, C, work, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": .6, "font.size": 9})
    if len(P):
        F = P[P.metric == "f1"]
        systems = list(F.system.unique())[:4]
        fig, axs = plt.subplots(1, len(systems), figsize=(4.2 * len(systems), 0.32 * F.config.nunique() + 1.8),
                                sharey=True, squeeze=False)
        for ax, s in zip(axs[0], systems):
            d = F[F.system == s].sort_values("config")
            y = np.arange(len(d))
            ax.errorbar(d["diff"] * 100, y, xerr=[(d["diff"] - d.lo) * 100, (d.hi - d["diff"]) * 100], fmt="o",
                        color=pal[0], ecolor=pal[0], ms=4, capsize=2)
            ax.axvline(0, color=MUTED, lw=1, ls="--")
            ax.set_yticks(y, d.config)
            ax.set_title(s[:30], loc="left", color=INK)
            ax.set_xlabel("ΔF1 vs baseline (x100, 95% CI)")
        fig.tight_layout(); fig.savefig(out / "plots" / "delta_f1_vs_baseline.png", dpi=160); plt.close(fig)
    irp = work / "ir" / "ir_summary.csv"
    if irp.exists():
        I = pd.read_csv(irp)
        k = a.link_metric if a.link_metric in I.columns else next(c for c in I.columns if c.startswith("nDCG@"))
        I = I.sort_values(k)
        fig, ax = plt.subplots(figsize=(6, 0.3 * len(I) + 1.6))
        y = np.arange(len(I))
        ax.errorbar(I[k], y, xerr=[I[k] - I[k + "_lo"], I[k + "_hi"] - I[k]], fmt="o", color=pal[1], ms=4, capsize=2)
        ax.set_yticks(y, I.config); ax.set_xlabel(f"{k} (cited papers as relevance, 95% CI)")
        ax.set_title("Retrieval quality by configuration", loc="left", color=INK)
        fig.tight_layout(); fig.savefig(out / "plots" / "ir_by_config.png", dpi=160); plt.close(fig)
    if E is not None and len(C):
        S = E.groupby(["config", "system"]).f1.mean().reset_index().merge(C, on="config")
        S = S.dropna(subset=["gen_seconds_per_paper"])
        if len(S):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for i, (s, d) in enumerate(S.groupby("system")):
                ax.scatter(d.gen_seconds_per_paper, d.f1 * 100, s=30, color=pal[i % 8], label=s[:28], zorder=3)
            ax.set_xlabel("generation seconds per paper (all systems of the config)"); ax.set_ylabel("F1 (x100)")
            ax.legend(frameon=False, fontsize=7); ax.set_title("Quality vs cost", loc="left", color=INK)
            fig.tight_layout(); fig.savefig(out / "plots" / "f1_vs_cost.png", dpi=160); plt.close(fig)
