"""
STAGE summarize — the `summary6` memory format.

Every retrieved paper used by a summary6 configuration is summarised ONCE (cached by paper id)
by the served generator into six fields, then placed in relevant_papers_list / retrieved_abs_int
under the same per-paper token budget as the other formats.
"""
from __future__ import annotations

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .build import approx_trunc
from .corpus import parse_obj

SYSTEM = "You are a concise academic summariser."
PROMPT = """Summarise the paper below for comparing it with another paper's limitations and novelty.
Write EXACTLY these six fields, one line each, plain text, no invention (write "Not stated" if absent):
Problem: ...
Method: ...
Data/Setup: ...
Key results: ...
Claimed contributions: ...
Stated limitations: ...

PAPER:
{text}"""


def run_summarize(a):
    from scilim.llm.client import ChatClient
    work = Path(a.work_dir)
    sp = work / "summary6.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    with open(work / "corpus_docs.pkl", "rb") as f:
        docs = pickle.load(f)["docs"]
    cfg_dirs = [d for d in sorted((work / "configs").glob("*")) if (d / "summary6_todo.json").exists()]
    todo = sorted({p for d in cfg_dirs for p in json.loads((d / "summary6_todo.json").read_text())} - set(summ))
    if not cfg_dirs:
        print("[summarize] no summary6 configuration — nothing to do")
        return
    llm = ChatClient(model=a.llm_model, base_url=a.llm_base_url, tokenizer=a.tokenizer, max_model_len=a.max_model_len,
                     family=a.family, timeout=600, max_retries=5)
    print(f"[summarize] {len(todo)} papers to summarise with {a.llm_model}")

    def work_fn(did):
        d = docs.get(did, {})
        txt = "\n".join(f"{k}: {v}" for k, v in [("Title", d.get("title", ""))] + list(d.get("sections", {}).items()) if v)
        return did, llm.chat(SYSTEM, PROMPT.format(text=approx_trunc(txt, 3000)), max_tokens=400, temperature=0.0)

    with ThreadPoolExecutor(a.workers) as ex:
        for n, (did, s) in enumerate(ex.map(work_fn, todo), 1):
            if not str(s).startswith("ERROR"):
                summ[did] = s
            if n % 50 == 0:
                sp.write_text(json.dumps(summ))
    sp.write_text(json.dumps(summ))
    for d in cfg_dirs:                                    # fill the placeholders in place
        cfg = json.loads((d / "config.json").read_text())
        df = pd.read_csv(d / "input.csv")
        rel, ab = [], []
        for _, r in df.iterrows():
            meta = json.loads(r["rag_abl__retrieved"])
            items = parse_obj(r["relevant_papers_list"]) or []
            recs = parse_obj(r["retrieved_abs_int"]) or {}
            for i, m in enumerate(meta):
                body = summ.get(m["doc_id"], "Summary not available.")
                head = f"Title: {m['title']}\n" if m["title"] else ""
                if i < len(items):
                    items[i] = head + approx_trunc(body, max(50, cfg["item_tokens"] - len(str(m["title"]).split())))
                if m["doc_id"] in recs:
                    recs[m["doc_id"]] = {"title": m["title"], "abstractText": items[i].split("\n", 1)[-1]}
            rel.append(repr(items)); ab.append(repr(recs))
        df["relevant_papers_list"], df["retrieved_abs_int"] = rel, ab
        df["rag_abl__context_chars"] = [sum(len(x) for x in (parse_obj(v) or [])) for v in rel]
        df.to_csv(d / "input.csv", index=False)
        (d / "summary6_todo.json").unlink()
        print(f"[summarize] filled {d.name}/input.csv")
