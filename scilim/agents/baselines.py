"""Zero-shot, RAG, RAG-only single-call baselines, and the LimAgents+NovAgents merge."""
from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd

from ..llm.client import ChatClient
from ..prompts import baselines as B
from ..prompts.merge import build_merge_messages
from ..utils.retrieval import build_shared_memory
from ..utils.text import as_text, is_missing


def _abstract(text: str) -> str:
    m = re.search(r"abstract[:\s]*(.{200,2500}?)(?:\n\s*\n|introduction)", text, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text[:2500]).strip()


def run_single(llm: ChatClient, row: pd.Series, kind: str, text_col="input_text_cleaned",
               retrieval_cols=("retrieved_abs_int", "ret_abs_int_cit"), top_k=8,
               retrieval_max_tokens=4000, paper_tokens=20000, max_new=1800,
               temperature=0.2, top_p=0.9, rep=1.1, model: Optional[str] = None,
               seed: Optional[int] = None) -> Dict[str, str]:
    paper = as_text(row.get(text_col))
    if len(paper) < 100:
        return {"final": "SKIPPED_SHORT_TEXT"}
    paper_tr = llm.truncate(paper, paper_tokens)
    out = {}
    if kind == "zero_shot":
        prompt = B.zero_shot_prompt(paper_tr)
    else:
        block, used = build_shared_memory(row, mode="auto", cols=list(retrieval_cols), top_k=top_k,
                                          tokenizer=llm.tok.hf, max_tokens=retrieval_max_tokens,
                                          max_chars=None if llm.tok.hf else retrieval_max_tokens * 4)
        out["retrieval_source"] = used
        out["retrieval_chars"] = len(block)
        prompt = B.rag_prompt(paper_tr, block) if kind == "rag" else B.rag_only_prompt(_abstract(paper), block)
    out["final"] = llm.chat(B.ZERO_SHOT_SYSTEM, prompt, max_tokens=max_new, temperature=temperature,
                            top_p=top_p, repetition_penalty=rep, model=model, seed=seed)
    return out


def run_merge(llm: ChatClient, lim_text: str, nov_text: str, style: str = "anchor",
              model: Optional[str] = None, temperature: float = 0.2, max_new: int = 4096) -> str:
    """merge.py / merge_and_eval.py / lim_nov.py: GPT-4o-mini by default, or any served model."""
    bad = {"SKIPPED_SHORT_TEXT", "SKIPPED_SHORT_MAIN_TEXT", "NO_REPORT_GENERATED", "NO_OUTPUT_FROM_MASTER"}
    if is_missing(lim_text) or is_missing(nov_text) or str(lim_text).strip() in bad or str(nov_text).strip() in bad:
        return "SKIPPED_MISSING_INPUT"
    system, user = build_merge_messages(str(lim_text), str(nov_text), style)
    return llm.chat(system, user, max_tokens=max_new, temperature=temperature, top_p=1.0, model=model)
