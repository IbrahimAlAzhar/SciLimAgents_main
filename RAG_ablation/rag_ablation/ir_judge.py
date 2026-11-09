"""
STAGE irjudge — graded relevance of retrieved papers by an LLM judge + human annotation export.

For a random sample of target papers (JUDGE_N, default 100) and the chosen configurations
(JUDGE_CONFIGS, default baseline), every retrieved paper (top n) is rated against the target
paper (title + abstract):
    2 = directly relevant prior work (should be cited / compared against)
    1 = related topic or method, useful background
    0 = not relevant
Writes <work>/ir/ir_judge_labels.json (used by stage `ir` for jP@k / jnDCG@k),
       <work>/ir/human_relevance_sheet.csv (blind sheet: fill `human_label` with 0/1/2), and,
       when HUMAN_LABELS points to the filled sheet, judge-human agreement (Cohen / weighted kappa).
"""
from __future__ import annotations

import json
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .build import approx_trunc, target_parts

SYSTEM = "You are an expert scientific reviewer judging literature relevance. Output JSON only."
PROMPT = """TARGET PAPER
Title: {t_title}
Abstract: {t_abs}

CANDIDATE RETRIEVED PAPER
Title: {c_title}
Abstract: {c_abs}

How relevant is the candidate as prior work for judging the target paper's novelty and limitations?
2 = directly relevant prior work (the target should cite it or compare against it)
1 = related topic or method, useful background
0 = not relevant
Return ONLY JSON: {{"label": 0, "reason": "<one sentence>"}}"""


def kappa(a, b, weights=None):
    a, b = np.asarray(a, int), np.asarray(b, int)
    cats = sorted(set(a) | set(b))
    if len(a) == 0 or len(cats) < 2:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    if weights == "linear":
        W = np.abs(np.subtract.outer(range(len(cats)), range(len(cats)))) / (len(cats) - 1)
    else:
        W = 1 - np.eye(len(cats))
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else 1.0


def run_irjudge(a):
    from scilim.llm.client import ChatClient, is_openai_model, parse_json_obj
    work = Path(a.work_dir)
    out = work / "ir"
    out.mkdir(exist_ok=True)
    lp = out / "ir_judge_labels.json"
    labels = json.loads(lp.read_text()) if lp.exists() else {}
    with open(work / "corpus_docs.pkl", "rb") as f:
        docs = pickle.load(f)["docs"]
    llm = ChatClient(model=a.judge_model, base_url=None if is_openai_model(a.judge_model) or a.judge_model == "mock"
                     else a.judge_base_url, max_model_len=128000 if is_openai_model(a.judge_model) else 32768,
                     timeout=600, max_retries=5)
    jobs, sheet = [], []
    for cfg in a.judge_configs.split():
        p = work / "configs" / cfg / "input.csv"
        if not p.exists():
            print(f"[irjudge] {cfg}: no input.csv — skipped")
            continue
        df = pd.read_csv(p)
        rids = sorted(df.row_id.astype(str))
        random.Random(a.seed).shuffle(rids)
        pick = set(rids[:a.judge_n])
        for _, r in df[df.row_id.astype(str).isin(pick)].iterrows():
            title, secs, _ = target_parts(r, a.pdf_col, a.text_col)
            for m in json.loads(r["rag_abl__retrieved"]):
                key = f"{cfg}|{r['row_id']}|{m['doc_id']}"
                d = docs.get(m["doc_id"], {})
                c_abs = d.get("sections", {}).get("abstract") or next(iter(d.get("sections", {}).values()), "")
                prompt = PROMPT.format(t_title=title or "(untitled)", t_abs=approx_trunc(secs.get("abstract", ""), 400),
                                       c_title=d.get("title", "") or "(untitled)", c_abs=approx_trunc(c_abs, 400))
                if key not in labels:
                    jobs.append((key, prompt))
                sheet.append({"item": key, "target_title": title, "target_abstract": approx_trunc(secs.get("abstract", ""), 250),
                              "candidate_title": d.get("title", ""), "candidate_abstract": approx_trunc(c_abs, 250),
                              "rank": m["rank"], "human_label": ""})
    print(f"[irjudge] {len(jobs)} pairs to judge with {a.judge_model}")

    def fn(job):
        key, prompt = job
        if llm.mock:
            return key, random.Random(key).choice([0, 1, 1, 2])
        obj = parse_json_obj(llm.chat(SYSTEM, prompt, max_tokens=120, temperature=0.0, json_mode=llm.is_openai))
        try:
            return key, int(obj.get("label"))
        except (TypeError, ValueError):
            return key, None

    with ThreadPoolExecutor(a.workers) as ex:
        for n, (k, v) in enumerate(ex.map(fn, jobs), 1):
            if v in (0, 1, 2):
                labels[k] = v
            if n % 100 == 0:
                lp.write_text(json.dumps(labels))
    lp.write_text(json.dumps(labels))
    S = pd.DataFrame(sheet).sample(frac=1, random_state=a.seed)
    S.to_csv(out / "human_relevance_sheet.csv", index=False)
    print(f"[irjudge] {len(labels)} judged pairs; blind sheet for annotators -> {out / 'human_relevance_sheet.csv'}")
    if a.human_labels and Path(a.human_labels).exists():
        H = pd.read_csv(a.human_labels)
        H = H[pd.to_numeric(H.human_label, errors="coerce").notna()]
        H["judge"] = H.item.map(labels)
        H = H[H.judge.notna()]
        h, j = H.human_label.astype(int), H.judge.astype(int)
        res = {"n": len(H), "cohen_kappa": kappa(h, j), "weighted_kappa_linear": kappa(h, j, "linear"),
               "exact_agreement": float((h == j).mean()),
               "human_P_rel>=1": float((h >= 1).mean()), "judge_P_rel>=1": float((j >= 1).mean())}
        (out / "ir_judge_agreement.json").write_text(json.dumps(res, indent=2))
        print(f"[irjudge] human vs judge: {res}")
