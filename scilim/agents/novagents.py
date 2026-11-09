"""
NovAgents pipeline (novagents.py): summarize up to 3 retrieved papers (Paper B)
-> 6 novelty specialists compare Paper A vs B -> Master writes the final
novelty & significance limitations report.

Roles: "summarizer", "worker" (specialists), "master" -> served model names.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..llm.client import ChatClient
from ..prompts import novagents as N
from ..utils.text import as_text, parse_relevant_list_nov


@dataclass
class NovAgentsConfig:
    agents: Optional[List[str]] = None
    with_master: bool = True
    roles: Dict[str, Optional[str]] = field(default_factory=dict)
    text_col: str = "input_text_cleaned"
    relevant_col: str = "relevant_papers_list"
    k_relevant: int = 3
    paper_tokens: int = 18000
    citation_tokens: int = 5000
    per_paper_input_tokens: int = 2500
    summary_max_new: int = 700
    specialist_max_new: int = 1500
    master_max_new: int = 2500
    handoff_tokens: int = 10000
    temperature: float = 0.2
    top_p: float = 0.9
    rep: float = 1.1
    min_chars: int = 200
    worker_threads: int = 6
    seed: Optional[int] = None


class NovAgents:
    def __init__(self, llm: ChatClient, cfg: NovAgentsConfig):
        self.llm, self.cfg = llm, cfg
        self.tok = llm.tok

    def _chat(self, role, system, user, max_new, temperature=None, top_p=None, seed=None):
        c = self.cfg
        return self.llm.chat(system, user, max_tokens=max_new,
                             temperature=c.temperature if temperature is None else temperature,
                             top_p=c.top_p if top_p is None else top_p, repetition_penalty=c.rep,
                             model=self.cfg.roles.get(role) or None,
                             seed=c.seed if seed is None else seed)

    # -- Paper-B summaries (shared with rollouts) --------------------------
    def build_inputs(self, row, temperature=None, seed=None):
        c = self.cfg
        main = as_text(row.get(c.text_col))
        rel = parse_relevant_list_nov(row.get(c.relevant_col, ""))
        summaries = []
        for idx in range(c.k_relevant):
            if idx < len(rel):
                raw = self.tok.truncate(as_text(rel[idx]), c.per_paper_input_tokens, suffix="")
                summaries.append(self._chat("summarizer", N.SUMMARISER_SYSTEM, N.summariser_user(raw),
                                            c.summary_max_new, temperature, None, seed))
            else:
                summaries.append(N.MISSING_B_SUMMARY)
        b_combined = N.combine_b_summaries(summaries)
        combined = N.build_dual_paper_input(self.tok.truncate(main, c.paper_tokens, suffix=""),
                                            self.tok.truncate(b_combined, c.citation_tokens, suffix=""))
        return main, summaries, combined

    def run_specialist(self, name, combined, temperature=None, top_p=None, seed=None):
        return self._chat("worker", N.specialist_system(name), N.specialist_user(combined),
                          self.cfg.specialist_max_new, temperature, top_p, seed)

    def run_master(self, specialist_outputs, temperature=None, top_p=None, seed=None):
        handoff = self.tok.truncate(N.handoff_block(specialist_outputs), self.cfg.handoff_tokens, suffix="")
        out = self._chat("master", N.MASTER_AGENT_PROMPT, N.master_user(handoff),
                         self.cfg.master_max_new, temperature, top_p, seed)
        return out.replace("TERMINATE", "").strip(), handoff

    def run(self, row: pd.Series) -> Dict[str, str]:
        c = self.cfg
        main = as_text(row.get(c.text_col))
        names = c.agents or [n for n, _ in N.SPECIALIST_AGENTS]
        if len(main) < c.min_chars:
            return {"final": "SKIPPED_SHORT_MAIN_TEXT"}
        _, summaries, combined = self.build_inputs(row)
        out = {f"relevant_paper_sum{i+1}": (summaries[i] if i < len(summaries) else "")
               for i in range(c.k_relevant)}
        with ThreadPoolExecutor(max_workers=max(1, c.worker_threads)) as ex:
            res = list(ex.map(lambda n: (n, self.run_specialist(n, combined)), names))
        history = {}
        for n, o in res:
            out[n] = o
            history[n] = o
        if len(names) == 1 and not c.with_master:
            out["final"] = res[0][1]
            return out
        final, _ = self.run_master(res)
        history["Master_Agent"] = final
        if len(final) < 50 or final.startswith("ERROR"):
            final = "NO_REPORT_GENERATED"
        out["final"] = final
        out["full_chat_history"] = str(history)
        return out
