"""
LimAgents pipelines.

style="v1"        Original LimAgents: summarize relevant papers -> 7 specialists
                  (incl. Citation) -> Master.          (limagents.py)
style="seq"       7 workers -> Leader consolidation -> Master, "sequential-
                  aligned" prompts; optional 3-hop workers. (dpo/grpo inference.py)
style="canonical" 6 canonical workers -> Leader MODE B -> Master (+ToT); the
                  prompts the rollouts / SFT / DPO / GRPO adapters were trained
                  on; optional shared memory, 3-hop workers and critique rounds.
                  (inference_shared_memory.py, rollout_qwen_vllm.py)

Agent subsets: `agents=[...]` runs only those specialists. With exactly one
agent and `with_master=False`, the agent's own list is the final output
(single-agent ablation). Leave-one-out masters are in `run_v1_leave_one_out`.

`roles` maps role -> served model name (base model or LoRA adapter), e.g.
{"worker": "limagents-jsd-worker-dpo", "leader": "...-leader-sft", ...}.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..llm.client import ChatClient
from ..prompts import canonical as C
from ..prompts import limagents_seq as Q
from ..prompts import limagents_v1 as V1
from ..utils.retrieval import build_shared_memory
from ..utils.text import (Tok, as_text, extract_intro_and_abstract, clean_text_detailed,
                          parse_relevant_papers_list)


@dataclass
class LimAgentsConfig:
    style: str = "v1"
    agents: Optional[List[str]] = None           # None = all
    with_master: bool = True
    roles: Dict[str, Optional[str]] = field(default_factory=dict)
    text_col: str = "input_text_cleaned"
    novelty_col: str = "input_text_for_novelty"
    cited_col: str = "cited_in"
    relevant_col: str = "relevant_papers_list"
    # v1 knobs (limagents.py)
    v1_temperature: float = 0.3
    v1_top_p: float = 0.9
    v1_rep: float = 1.1
    v1_max_new: int = 700
    v1_summary_max_new: int = 500
    v1_summary_item_tokens: int = 8000
    v1_summary_items: int = 3
    v1_paper_tokens: int = 16000
    v1_citation_tokens: int = 4000
    summarize_relevant: bool = True
    # seq knobs (dpo inference.py)
    seq_paper_tokens: int = 20000
    seq_citation_tokens: int = 5000
    seq_worker_max_new: int = 2048
    seq_leader_max_new: int = 2048
    seq_master_max_new: int = 3000
    seq_combined_tokens: int = 12000
    hops: bool = False
    # canonical knobs (inference_shared_memory.py)
    temperature: float = 0.2
    top_p: float = 0.9
    rep: float = 1.1
    paper_chars: int = 20000
    worker_max_new: int = 1400
    leader_max_new: int = 1800
    master_max_new: int = 1800
    master_tot: bool = True
    critique_rounds: int = 0
    hop_mode: str = "strict_grounding"           # canonical 3-hop exploration mode
    shared_memory: bool = False
    shared_memory_roles: tuple = ("worker", "leader", "master")
    shared_memory_mode: str = "auto"
    retrieval_cols: tuple = ("retrieved_abs_int", "ret_abs_int_cit")
    retrieval_top_k: int = 8
    retrieval_max_tokens: int = 2500
    leader_ascii_prompt: bool = False
    worker_threads: int = 7
    seed: Optional[int] = None


class LimAgents:
    def __init__(self, llm: ChatClient, cfg: LimAgentsConfig):
        self.llm, self.cfg = llm, cfg
        self.tok: Tok = llm.tok

    def role(self, r: str) -> Optional[str]:
        return self.cfg.roles.get(r) or None

    def _chat(self, role, system, user, max_new, temperature, top_p, rep, seed=None):
        return self.llm.chat(system, user, max_tokens=max_new, temperature=temperature,
                             top_p=top_p, repetition_penalty=rep, model=self.role(role),
                             seed=seed if seed is not None else self.cfg.seed)

    def run(self, row: pd.Series) -> Dict[str, str]:
        s = self.cfg.style
        if s == "v1":
            return self.run_v1(row)
        if s == "seq":
            return self.run_seq(row)
        if s == "canonical":
            return self.run_canonical(row)
        raise ValueError(f"unknown LimAgents style {s}")

    # ================================================================= v1 ==
    def _v1_inputs(self, row):
        c = self.cfg
        out = {}
        main_text = as_text(row.get(c.novelty_col)) or as_text(row.get(c.text_col))
        rel_sum = as_text(row.get("relevant_papers_sum"))
        if c.summarize_relevant and not rel_sum and (c.agents is None or c.agents != ["Citation_Agent"]):
            rel_list = parse_relevant_papers_list(row.get(c.relevant_col, ""))
            summaries = []
            for j, item in enumerate(rel_list[:c.v1_summary_items], start=1):
                item_tr = self.tok.truncate(clean_text_detailed(item), c.v1_summary_item_tokens)
                sp = V1.summarization_prompt_for_relevant_paper(item_tr, j)
                summaries.append(self._chat("summarizer", V1.SUMMARIZER_SYSTEM, sp,
                                            c.v1_summary_max_new, c.v1_temperature,
                                            c.v1_top_p, c.v1_rep).strip())
            rel_sum = "\n\n".join(summaries).strip()
        out["relevant_papers_sum"] = rel_sum
        combined = V1.build_v1_combined_input(self.tok.truncate(main_text, c.v1_paper_tokens),
                                              self.tok.truncate(rel_sum, c.v1_citation_tokens))
        paper_for_citation = as_text(row.get(c.text_col))
        citation_text = self.tok.truncate(
            extract_intro_and_abstract(row.get(c.cited_col, ""), style="v1"), c.v1_citation_tokens)
        return out, combined, paper_for_citation, citation_text

    def run_v1(self, row) -> Dict[str, str]:
        c = self.cfg
        out, combined, paper_cit, cit = self._v1_inputs(row)
        names = c.agents or [n for n, _ in V1.SPECIALIST_COLUMNS]

        def one(name):
            fn = V1.AGENT_PROMPT_FUNCS[name]
            p = fn(paper_cit, cit) if name == "Citation_Agent" else fn(combined)
            return name, self._chat("worker", V1.AGENT_SYSTEM_PROMPTS[name], p, c.v1_max_new,
                                    c.v1_temperature, c.v1_top_p, c.v1_rep)

        with ThreadPoolExecutor(max_workers=max(1, c.worker_threads)) as ex:
            results = dict(ex.map(one, names))
        for n in names:
            out[n] = results[n]

        if len(names) == 1 and not c.with_master:
            out["final"] = results[names[0]]
            return out
        if c.agents is None:
            mp = V1.get_master_synthesis_prompt(combined, results)
        else:
            included = [(t, results.get(col, "")) for col, t in V1.SPECIALIST_COLUMNS if col in results]
            mp = V1.build_master_prompt_subset(combined, included)
        out["final"] = self._chat("master", V1.MASTER_SYSTEM, mp, c.v1_max_new,
                                  c.v1_temperature, c.v1_top_p, c.v1_rep)
        return out

    def run_v1_leave_one_out(self, row, specialist_outputs: Dict[str, str],
                             paper_tokens: int = 3500, relevant_tokens: int = 1000,
                             max_new: int = 1500) -> Dict[str, str]:
        """leave_one_out_limagents.py: master re-run 7x, each excluding one agent.
        Paper context = input_text_cleaned (3500 tok) + relevant sum (1000 tok)."""
        c = self.cfg
        ctx = V1.build_v1_combined_input(
            self.tok.truncate(as_text(row.get(c.text_col)), paper_tokens),
            self.tok.truncate(as_text(row.get("relevant_papers_sum") or specialist_outputs.get(
                "relevant_papers_sum", "")), relevant_tokens))
        out = {}
        for ablated, _t in V1.SPECIALIST_COLUMNS:
            included = [(t, specialist_outputs.get(col, "")) for col, t in V1.SPECIALIST_COLUMNS
                        if col != ablated]
            out[f"without_{ablated}"] = self._chat(
                "master", V1.MASTER_SYSTEM, V1.build_master_prompt_subset(ctx, included),
                max_new, c.v1_temperature, 0.9, 1.1)
        return out

    # ================================================================ seq ==
    def _hops_seq(self, label, base_system):
        outs = {}
        for hop, modes in Q.HOP_MODE_MAP:
            sp = Q.build_hop_system_prompt(base_system, hop, modes)
            if hop == "search":
                um = Q.HOP_USER_INSTRUCTIONS["search"]
            elif hop == "query":
                um = Q.HOP_USER_INSTRUCTIONS["query"].format(
                    search_out=self.tok.truncate(outs.get("search", ""), 3000))
            else:
                um = Q.HOP_USER_INSTRUCTIONS["draft"].format(
                    query_out=self.tok.truncate(outs.get("query", ""), 3500))
            outs[hop] = self._chat("worker", sp, um, Q.HOP_MAX_TOKENS.get(hop, 1536),
                                   0.2, 0.9, 1.1)
        draft = outs.get("draft", "").strip()
        if len(draft) < 30 or draft.startswith("ERROR"):
            draft = outs.get("query", draft).strip()
        trace = (f"----- SEARCH -----\n{outs.get('search', '')}\n\n"
                 f"----- QUERY -----\n{outs.get('query', '')}\n\n"
                 f"----- DRAFT -----\n{outs.get('draft', '')}")
        return draft, trace

    def run_seq(self, row) -> Dict[str, str]:
        c = self.cfg
        paper = self.tok.truncate(as_text(row.get(c.text_col)), c.seq_paper_tokens)
        cit = self.tok.truncate(extract_intro_and_abstract(row.get(c.cited_col, ""), style="v1"),
                                c.seq_citation_tokens)
        cols = [x[0] for x in Q.SPECIALIST_CONFIG]
        if len(paper) < 100:
            return {k: "SKIPPED_SHORT_TEXT" for k in cols + ["leader_consolidated", "final"]}
        config = [x for x in Q.SPECIALIST_CONFIG
                  if c.agents is None or x[0] in c.agents or x[0].replace("lim_", "") in c.agents]

        def one(cfg_row):
            col, fn, needs_cit = cfg_row
            base = fn(paper, cit) if needs_cit else fn(paper)
            label = col.replace("lim_", "").replace("_", " ").title()
            if c.hops:
                draft, trace = self._hops_seq(label, base)
                return col, label, draft, trace
            um = Q.CITATION_USER_MSG if needs_cit else Q.WORKER_USER_MSG
            return col, label, self._chat("worker", base, um, c.seq_worker_max_new, 0.2, 0.9, 1.1), None

        with ThreadPoolExecutor(max_workers=max(1, c.worker_threads)) as ex:
            res = list(ex.map(one, config))
        out, blocks = {}, []
        for col, label, o, trace in res:
            out[col] = o
            if trace is not None:
                out[col + "_hops"] = trace
            blocks.append(f"=== {'Citation Agent' if col == 'lim_citation' else label} ===\n{o}")
        if len(config) == 1 and not c.with_master:
            out["final"] = res[0][2]
            return out
        combined = self.tok.truncate("\n\n".join(blocks), c.seq_combined_tokens)
        leader = self._chat("leader", Q.get_leader_agent_prompt(paper, hops=c.hops),
                            Q.leader_input(combined, hops=c.hops), c.seq_leader_max_new, 0.2, 0.9, 1.1)
        out["leader_consolidated"] = leader
        final = self._chat("master", Q.get_master_agent_prompt(paper, hops=c.hops),
                           Q.master_input(self.tok.truncate(leader, c.seq_combined_tokens)),
                           c.seq_master_max_new, 0.2, 0.9, 1.1)
        if len(final.strip()) < 50 or final.startswith("ERROR"):
            final = "NO_OUTPUT_FROM_MASTER"
        out["final"] = final
        return out

    # ========================================================== canonical ==
    def _memory(self, row):
        c = self.cfg
        if not c.shared_memory:
            return "", "off"
        body, used = build_shared_memory(row, mode=c.shared_memory_mode, cols=list(c.retrieval_cols),
                                         top_k=c.retrieval_top_k,
                                         tokenizer=self.tok.hf, max_tokens=c.retrieval_max_tokens,
                                         max_chars=None if self.tok.hf else c.retrieval_max_tokens * 4)
        if body:
            body = C.SHARED_MEMORY_HEADER + body.split("===\n", 1)[-1]
        return body, used

    def _fit_worker_system(self, prompt_fn, paper, memory_block, max_input_tokens=12000):
        """Shrink the PAPER (never the instructions) until the prompt fits."""
        paper_text = paper
        for _ in range(6):
            sp = prompt_fn(paper_text) + memory_block
            n = self.tok.count(sp)
            if n <= max_input_tokens - 200:
                return sp
            over = n - (max_input_tokens - 200)
            cut = max(500, int(over * 4 * 1.1))
            paper_text = (paper_text[: max(1000, len(paper_text) // 2)] if cut >= len(paper_text)
                          else paper_text[: len(paper_text) - cut])
        return prompt_fn(paper_text) + memory_block

    def _canonical_worker(self, wcfg, paper, w_mem, evidence_text, row_seed):
        name, fn, spec, _key, col = wcfg
        c = self.cfg
        trace = None
        if c.hops:
            action = C.MODE_ACTION_CONFIG[c.hop_mode]
            worker_paper = evidence_text if action.get("cite_evidence_ids") else paper
            base = self._fit_worker_system(fn, worker_paper, w_mem, self.llm.max_model_len - 2000)
            hop_out = {}
            for hop in ("search", "query", "draft"):
                sp = C.build_agent_system(base, c.hop_mode, hop, action)
                if hop == "search":
                    um = C.hop_search_user(name, spec)
                elif hop == "query":
                    um = C.HOP_USER_INSTRUCTIONS["query"].format(
                        search_out=self.tok.truncate(hop_out.get("search", ""), 3000, suffix=" ... [TRUNCATED]"))
                else:
                    um = C.HOP_USER_INSTRUCTIONS["draft"].format(
                        query_out=self.tok.truncate(hop_out.get("query", ""), 3500, suffix=" ... [TRUNCATED]"))
                hop_out[hop] = self._chat("worker", sp, um, 900 if hop != "draft" else 1400,
                                          c.temperature, c.top_p, c.rep, seed=row_seed)
            current = hop_out.get("draft", "").strip() or hop_out.get("query", "").strip()
            trace = "\n\n".join(f"----- {h.upper()} -----\n{hop_out.get(h, '')}" for h in ("search", "query", "draft"))
            base_for_rev = base
        else:
            base_for_rev = self._fit_worker_system(fn, paper, w_mem, min(12000, self.llm.max_model_len - 2000))
            current = self._chat("worker", base_for_rev, C.WORKER_USER.format(specialty=spec),
                                 c.worker_max_new, c.temperature, c.top_p, c.rep, seed=row_seed)
        # optional leader critique rounds (MODE A) + worker revision
        leader_sys = C.get_leader_system_prompt_ascii() if c.leader_ascii_prompt else C.get_leader_system_prompt()
        for r in range(c.critique_rounds):
            fb = self._chat("leader", leader_sys, C.leader_feedback_user(name, spec, r + 1, current),
                            700, 0.4, 0.9, c.rep, seed=row_seed)
            rev_sys = (C.build_agent_system(base_for_rev, c.hop_mode, "draft", C.MODE_ACTION_CONFIG[c.hop_mode])
                       if c.hops else base_for_rev)
            current = self._chat("worker", rev_sys, C.worker_revision_user(current, fb),
                                 1400, c.temperature, c.top_p, c.rep, seed=row_seed)
        return name, col, current, trace

    def run_canonical(self, row) -> Dict[str, str]:
        c = self.cfg
        paper = as_text(row.get(c.text_col))[: c.paper_chars]
        cols = [w[4] for w in C.WORKER_CONFIGS]
        if len(paper) < 100:
            return {k: "SKIPPED_SHORT_TEXT" for k in cols + ["leader_consolidated", "final"]}
        mem, used = self._memory(row)
        role_mem = lambda r: (mem + C.SHARED_MEMORY_NOTES[r]) if (mem and r in c.shared_memory_roles) else ""
        from ..utils.text import build_evidence_tagged
        evidence = build_evidence_tagged(paper) if c.hops else ""
        workers = [w for w in C.WORKER_CONFIGS if c.agents is None or w[0] in c.agents or w[4] in c.agents]
        rid = int(row.get("row_id", 0) or 0)
        with ThreadPoolExecutor(max_workers=max(1, c.worker_threads)) as ex:
            res = list(ex.map(lambda w: self._canonical_worker(w, paper, role_mem("worker"), evidence,
                                                               None if c.seed is None else c.seed + rid),
                              workers))
        out = {"shared_memory_chars": len(mem), "shared_memory_source": used} if c.shared_memory else {}
        worker_outputs = {}
        for name, col, o, trace in res:
            out[col] = o
            if trace:
                out[col + "_hops"] = trace
            worker_outputs[name] = o
        if len(workers) == 1 and not c.with_master:
            out["final"] = res[0][2]
            return out
        leader_sys = C.get_leader_system_prompt_ascii() if c.leader_ascii_prompt else C.get_leader_system_prompt()
        leader = self._chat("leader", leader_sys + role_mem("leader"), C.leader_bundle_user(worker_outputs),
                            c.leader_max_new, c.temperature, c.top_p, c.rep)
        out["leader_consolidated"] = leader
        master_sys = C.get_master_system_prompt() + (C.MASTER_TOT_SUFFIX if c.master_tot else "")
        final = self._chat("master", master_sys + role_mem("master"), leader,
                           c.master_max_new, c.temperature, c.top_p, c.rep)
        if len(final.strip()) < 30 or final.startswith("ERROR"):
            final = "NO_OUTPUT_FROM_MASTER"
        out["final"] = final
        return out
