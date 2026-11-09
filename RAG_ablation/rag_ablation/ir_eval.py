"""
STAGE ir — retrieval quality measured directly (no LLM needed).

Relevance labels = the target paper's OWN reference list, mapped onto the RAG corpus:
  * titles from pdf_text["references"][*]["title"] (science-parse) and, if present, the
    titles in the `cited_in` column (REL_SOURCE = references | cited_in | both)
  * a reference is "in corpus" if its normalised title matches a corpus paper title exactly,
    or with token-Jaccard >= 0.85 among corpus titles sharing the first three words
  * the target paper itself is never relevant
Metrics per target paper and configuration (k = 1 3 5 9 15 by default, paper-level, binary):
  P@k, R@k (denominator = cited papers present in the corpus), nDCG@k, Hit@k, MRR
Papers whose references are not in the corpus are excluded from R/nDCG (coverage is reported).
Optional graded relevance from the LLM judge (stage irjudge): jP@k, jnDCG@k on the judged sample.
Outputs: <work>/ir/ir_per_paper.csv, ir_summary.csv (mean + 95% bootstrap CI), ir_by_year.csv
"""
from __future__ import annotations

import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import coerce_text, norm_title, parse_obj


def ref_titles(row, source="both", pdf_col="pdf_text", cited_col="cited_in"):
    out = []
    if source in ("references", "both"):
        d = parse_obj(row.get(pdf_col)) if pdf_col in row else None
        if isinstance(d, dict):
            for r in d.get("references", []) or []:
                if isinstance(r, dict) and r.get("title"):
                    out.append(coerce_text(r["title"]))
    if source in ("cited_in", "both") and cited_col in row:
        d = parse_obj(row.get(cited_col))
        vals = d.values() if isinstance(d, dict) else (d if isinstance(d, list) else [])
        for v in vals:
            if isinstance(v, dict) and v.get("title"):
                out.append(coerce_text(v["title"]))
    return out


class TitleMatcher:
    def __init__(self, title_index):
        self.exact = {t: set(v) for t, v in title_index.items() if t}
        self.bucket = defaultdict(list)
        for t in self.exact:
            self.bucket[" ".join(t.split()[:3])].append(t)

    def match(self, title):
        n = norm_title(title)
        if len(n) < 8:
            return set()
        if n in self.exact:
            return self.exact[n]
        a = set(n.split())
        out = set()
        for t in self.bucket.get(" ".join(n.split()[:3]), []):
            b = set(t.split())
            if len(a & b) / max(1, len(a | b)) >= 0.85:
                out |= self.exact[t]
        return out


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def metrics(ranked, relevant, ks, graded=None):
    out = {}
    rel = [1 if d in relevant else 0 for d in ranked]
    for k in ks:
        top = rel[:k]
        out[f"P@{k}"] = sum(top) / k
        out[f"R@{k}"] = sum(top) / len(relevant) if relevant else np.nan
        ideal = dcg([1] * min(k, len(relevant)))
        out[f"nDCG@{k}"] = dcg(top) / ideal if ideal > 0 else np.nan
        out[f"Hit@{k}"] = float(any(top)) if relevant else np.nan
        if graded is not None:
            g = [graded.get(d, np.nan) for d in ranked[:k]]
            gv = [x for x in g if not np.isnan(x)]
            out[f"jP@{k}"] = float(np.mean([x >= 1 for x in gv])) if gv else np.nan
            ideal_g = dcg(sorted([x for x in graded.values() if not np.isnan(x)], reverse=True)[:k])
            out[f"jnDCG@{k}"] = dcg([0 if np.isnan(x) else x for x in g]) / ideal_g if ideal_g > 0 else np.nan
    first = next((i for i, r in enumerate(rel) if r), None)
    out["MRR"] = 1.0 / (first + 1) if first is not None else (0.0 if relevant else np.nan)
    return out


def boot(v, n=1000, seed=0):
    v = np.asarray([x for x in v if not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan,) * 3
    bs = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def run_ir(a):
    work = Path(a.work_dir)
    with open(work / "corpus_docs.pkl", "rb") as f:
        cd = pickle.load(f)
    TM = TitleMatcher(cd["title_index"])
    ks = [int(x) for x in a.ks.split()]
    judged = {}
    jp = work / "ir" / "ir_judge_labels.json"
    if jp.exists():
        for k, v in json.loads(jp.read_text()).items():
            cfg, rid, did = k.split("|", 2)
            judged.setdefault((cfg, rid), {})[did] = v
    rows = []
    rel_cache = {}
    for d in sorted((work / "configs").glob("*")):
        if not (d / "input.csv").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("original") or not cfg.get("memory", True):
            continue
        df = pd.read_csv(d / "input.csv")
        for _, r in df.iterrows():
            rid = str(r["row_id"])
            if rid not in rel_cache:
                titles = ref_titles(r, a.rel_source, a.pdf_col)
                tp = parse_obj(r.get(a.pdf_col)) if a.pdf_col in r else None
                self_ids = TM.match(coerce_text(tp.get("title", ""))) if isinstance(tp, dict) else set()
                relevant = set().union(*[TM.match(t) for t in titles]) - self_ids if titles else set()
                rel_cache[rid] = (len(titles), relevant)
            n_refs, relevant = rel_cache[rid]
            ranked = [m["doc_id"] for m in json.loads(r["rag_abl__retrieved"])]
            g = judged.get((cfg["name"], rid))
            rows.append({"config": cfg["name"], "row_id": rid, "target_year": r.get("rag_abl__target_year"),
                         "n_references": n_refs, "n_relevant_in_corpus": len(relevant), "n_retrieved": len(ranked),
                         **metrics(ranked, relevant, ks, g)})
    P = pd.DataFrame(rows)
    out = work / "ir"
    out.mkdir(exist_ok=True)
    P.to_csv(out / "ir_per_paper.csv", index=False)
    if P.empty:
        print("[ir] nothing to evaluate")
        return
    S = []
    mcols = [c for c in P.columns if "@" in c or c == "MRR"]
    for cfg, d in P.groupby("config"):
        s = {"config": cfg, "n_papers": len(d), "papers_with_refs": int((d.n_references > 0).sum()),
             "papers_with_relevant_in_corpus": int((d.n_relevant_in_corpus > 0).sum()),
             "mean_relevant_in_corpus": float(d.n_relevant_in_corpus.mean())}
        dd = d[d.n_relevant_in_corpus > 0]
        for c in mcols:
            src = d if c.startswith(("P@", "jP@", "jnDCG")) else dd
            s[c], s[c + "_lo"], s[c + "_hi"] = boot(src[c], seed=a.seed)
        S.append(s)
    S = pd.DataFrame(S)
    S.to_csv(out / "ir_summary.csv", index=False)
    if P.target_year.notna().any():
        Y = (P[P.n_relevant_in_corpus > 0].groupby(["config", "target_year"])
             [[c for c in mcols if c.startswith(("R@", "nDCG@"))]].mean().reset_index())
        Y["n_papers"] = P[P.n_relevant_in_corpus > 0].groupby(["config", "target_year"]).size().values
        Y.to_csv(out / "ir_by_year.csv", index=False)
    show = ["config", "papers_with_relevant_in_corpus"] + [c for c in S.columns if re.match(r"^(R|nDCG|P)@(3|9)$", c)] + ["MRR"]
    print(S[[c for c in show if c in S.columns]].round(3).to_string(index=False))
    cov = (P.drop_duplicates("row_id").n_relevant_in_corpus > 0).mean()
    print(f"[ir] {cov:.0%} of target papers cite at least one paper that is in the RAG corpus")
