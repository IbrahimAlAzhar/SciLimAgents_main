"""
Rollout generation (no rewards) — generalized rollout_qwen_vllm.py.

--system limagents : per paper N rollouts (default 4), each bound to an
    exploration mode + sampling config; 6 workers x 3 hops (SEARCH -> QUERY ->
    DRAFT) + the mode's Leader critique rounds, Leader MODE-B handoff, Master
    (+Tree-of-Thought). Identical to the original script.
--system novagents : (NEW) per paper N sampled rollouts of the NovAgents
    pipeline (shared Paper-B summaries, 6 specialists, Master). Enables
    NovAgents SFT / DPO / GRPO.

TEACHER: who generates the rollouts.
    --teacher self            the student model being trained (on-policy)
    --teacher gpt-4o-mini     OpenAI model (distillation from a bigger model)
    --teacher <served name> --teacher_base_url http://...   any open model in vLLM

Output schema is unchanged: rollout_data_full.json (+ _worker/_leader/_master
splits) and per-paper checkpoints for resume. Rows 0:200 are the held-out
evaluation set; the default slice is 300:450.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .. import config as CFG
from ..agents.novagents import NovAgents, NovAgentsConfig
from ..llm.client import ChatClient
from ..prompts import canonical as C
from ..prompts import novagents as N
from ..utils.text import as_text, build_evidence_tagged, extract_intro_and_abstract

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x


class Roller:
    def __init__(self, args):
        self.a = args
        self.llm = ChatClient(model=args.teacher_model, base_url=args.teacher_base_url or None,
                              tokenizer=args.tokenizer, max_model_len=args.max_model_len,
                              family=args.family, default_repetition_penalty=None)
        self.worker_model = args.worker_model or None
        self.leader_model = args.leader_model or None
        self.master_model = args.master_model or None
        self.rollout_cfgs = C.ROLLOUT_CONFIGS[: args.n_rollouts]
        self.rep = None if self.llm.is_openai else 1.05   # original extra_body repetition_penalty

    def call(self, system, user, cfg, max_tokens, model=None):
        return self.llm.chat(system, user, max_tokens=max_tokens, temperature=cfg["temperature"],
                             top_p=cfg.get("top_p", 1.0), seed=cfg.get("seed", 42),
                             repetition_penalty=self.rep, model=model)

    # ============================================================ LimAgents ==
    def worker_hops(self, name, base_sys, spec, mode, action, rc, retrieval_context):
        recs, hop_out = [], {}
        for hop in ("search", "query", "draft"):
            sp = C.build_agent_system(base_sys, mode, hop, action, retrieval_context if hop == "search" else "")
            if hop == "search":
                um = C.hop_search_user(name, spec)
            elif hop == "query":
                um = C.HOP_USER_INSTRUCTIONS["query"].format(
                    search_out=self.llm.truncate(hop_out.get("search", ""), 3000, suffix=" ... [TRUNCATED]"))
            else:
                um = C.HOP_USER_INSTRUCTIONS["draft"].format(
                    query_out=self.llm.truncate(hop_out.get("query", ""), 3500, suffix=" ... [TRUNCATED]"))
            out = self.call(sp, um, rc, 900 if hop != "draft" else 1400, self.worker_model)
            hop_out[hop] = out
            recs.append({"agent_role": "worker", "agent_name": name, "turn_type": f"hop_{hop}",
                         "mode": mode, "hop": hop,
                         "input_messages": [{"role": "system", "content": sp}, {"role": "user", "content": um}],
                         "output": out})
        return (hop_out.get("draft", "").strip() or hop_out.get("query", "").strip()), recs

    def worker_full(self, wspec, worker_paper, mode, action, rc, retrieval_context, leader_sys,
                    leader_cfg, pid, gt, rc_meta, leader_meta):
        name, fn, spec = wspec[0], wspec[1], wspec[2]
        base = fn(worker_paper)
        draft, recs = self.worker_hops(name, base, spec, mode, action, rc, retrieval_context)
        samples = []
        for r in recs:
            r.update({"paper_idx": pid, "rollout_id": rc["rollout_id"], "rollout_config": rc_meta,
                      "ground_truth": gt})
            samples.append(r)
        current = draft
        for r in range(action.get("critique_rounds", 0)):
            fb_user = C.leader_feedback_user(name, spec, r + 1, current)
            fb = self.call(leader_sys, fb_user, leader_cfg, 700, self.leader_model)
            samples.append({"paper_idx": pid, "rollout_id": rc["rollout_id"], "rollout_config": leader_meta,
                            "agent_role": "leader", "agent_name": "Leader_Agent",
                            "turn_type": "feedback_to_worker", "mode": mode, "critique_round": r + 1,
                            "target_worker": name,
                            "input_messages": [{"role": "system", "content": leader_sys},
                                               {"role": "user", "content": fb_user}],
                            "output": fb, "ground_truth": gt})
            rev_sys = C.build_agent_system(base, mode, "draft", action)
            rev_user = C.worker_revision_user(current, fb)
            current = self.call(rev_sys, rev_user, rc, 1400, self.worker_model)
            samples.append({"paper_idx": pid, "rollout_id": rc["rollout_id"], "rollout_config": rc_meta,
                            "agent_role": "worker", "agent_name": name, "turn_type": "revised",
                            "mode": mode, "critique_round": r + 1,
                            "input_messages": [{"role": "system", "content": rev_sys},
                                               {"role": "user", "content": rev_user}],
                            "output": current, "ground_truth": gt})
        return name, current, samples

    def lim_rollout(self, pid, paper_text, evidence_text, citation_text, gt, rc):
        a = self.a
        mode = rc["mode"]
        action = C.MODE_ACTION_CONFIG[mode]
        worker_paper = evidence_text if action.get("cite_evidence_ids") else paper_text
        retrieval_context = ""
        if action.get("use_retrieval"):
            retrieval_context = self.llm.truncate(
                citation_text, min(a.citation_token_limit, action.get("retrieval_top_k", 8) * 500),
                suffix=" ... [TRUNCATED]")
        leader_sys, master_sys = C.get_leader_system_prompt(), C.get_master_system_prompt()
        rc_meta = {k: rc[k] for k in ("temperature", "top_p", "seed", "mode")}
        ridx = rc["rollout_id"]
        lcfg = C.LEADER_CONFIGS[ridx % len(C.LEADER_CONFIGS)]
        mcfg = C.MASTER_CONFIGS[ridx % len(C.MASTER_CONFIGS)]
        lmeta = {**{k: lcfg[k] for k in ("temperature", "top_p", "seed")}, "mode": mode}
        mmeta = {**{k: mcfg[k] for k in ("temperature", "top_p", "seed")}, "mode": mode}

        with ThreadPoolExecutor(max_workers=max(1, a.worker_threads)) as ex:
            futs = [ex.submit(self.worker_full, w, worker_paper, mode, action, rc, retrieval_context,
                              leader_sys, lcfg, pid, gt, rc_meta, lmeta) for w in C.WORKER_CONFIGS]
            results = [f.result() for f in futs]
        order = {w[0]: i for i, w in enumerate(C.WORKER_CONFIGS)}
        results.sort(key=lambda t: order[t[0]])
        samples, finals = [], {}
        for name, final, s in results:
            finals[name] = final
            samples.extend(s)

        bundle = C.leader_bundle_user(finals)
        handoff = self.call(leader_sys, bundle, lcfg, 1800, self.leader_model)
        samples.append({"paper_idx": pid, "rollout_id": ridx, "rollout_config": lmeta,
                        "agent_role": "leader", "agent_name": "Leader_Agent", "turn_type": "handoff_to_master",
                        "mode": mode, "input_messages": [{"role": "system", "content": leader_sys},
                                                         {"role": "user", "content": bundle}],
                        "output": handoff, "ground_truth": gt})
        msys = master_sys + (C.MASTER_TOT_SUFFIX if a.apply_master_tot else "")
        master = self.call(msys, handoff, mcfg, 1800, self.master_model)
        samples.append({"paper_idx": pid, "rollout_id": ridx, "rollout_config": mmeta,
                        "agent_role": "master", "agent_name": "Master_Agent", "turn_type": "synthesis",
                        "mode": mode, "input_messages": [{"role": "system", "content": msys},
                                                         {"role": "user", "content": handoff}],
                        "output": master, "ground_truth": gt})
        return {"mode": mode, "samples": samples, "final_master_output": master, "worker_final": finals}

    # ============================================================ NovAgents ==
    def nov_paper(self, pid, row, gt):
        nov = NovAgents(self.llm, NovAgentsConfig(text_col=self.a.text_col,
                                                  roles={"worker": self.worker_model,
                                                         "master": self.master_model}))
        main, summaries, combined = nov.build_inputs(row, temperature=0.2, seed=42)
        rollouts = []
        for rc in self.rollout_cfgs:
            samples, outs = [], []
            for name, _ in N.SPECIALIST_AGENTS:
                o = nov.run_specialist(name, combined, rc["temperature"], rc["top_p"], rc["seed"])
                outs.append((name, o))
                samples.append({"paper_idx": pid, "rollout_id": rc["rollout_id"], "rollout_config": rc,
                                "agent_role": "worker", "agent_name": name, "turn_type": "draft",
                                "mode": f"sample_{rc['rollout_id']}",
                                "input_messages": [{"role": "system", "content": N.specialist_system(name)},
                                                   {"role": "user", "content": N.specialist_user(combined)}],
                                "output": o, "ground_truth": gt})
            mcfg = C.MASTER_CONFIGS[rc["rollout_id"] % len(C.MASTER_CONFIGS)]
            final, handoff = nov.run_master(outs, mcfg["temperature"], mcfg["top_p"], mcfg["seed"])
            samples.append({"paper_idx": pid, "rollout_id": rc["rollout_id"], "rollout_config": mcfg,
                            "agent_role": "master", "agent_name": "Master_Agent", "turn_type": "synthesis",
                            "mode": f"sample_{rc['rollout_id']}",
                            "input_messages": [{"role": "system", "content": N.MASTER_AGENT_PROMPT},
                                               {"role": "user", "content": N.master_user(handoff)}],
                            "output": final, "ground_truth": gt})
            rollouts.append({"rollout_id": rc["rollout_id"], "rollout_config": rc, "mode": f"sample_{rc['rollout_id']}",
                             "samples": samples, "final_master_output": final,
                             "worker_final": dict(outs)})
        return {"combined_input": combined, "b_summaries": summaries}, rollouts

    # ================================================================= main ==
    def run(self):
        a = self.a
        out_dir = a.out_dir
        ckpt_dir = os.path.join(out_dir, "checkpoints_per_paper")
        os.makedirs(ckpt_dir, exist_ok=True)
        if not self.llm.mock and not self.llm.is_openai:
            self.llm.wait_ready([m for m in (self.worker_model, self.leader_model, self.master_model) if m])
        df_full = pd.read_csv(a.input_csv)
        end = a.end_row if a.end_row is not None else len(df_full)
        df = df_full.iloc[a.start_row:end].copy().reset_index(drop=True)
        print(f"[rollout:{a.system}] teacher={a.teacher_model} rows [{a.start_row}:{end}] -> "
              f"{len(df)} papers x {len(self.rollout_cfgs)} rollouts", flush=True)
        existing = set(os.listdir(ckpt_dir))
        records = []
        for li in tqdm(range(len(df)), desc="Papers"):
            gi = a.start_row + li
            ck = f"paper_{gi:05d}.json"
            cp = os.path.join(ckpt_dir, ck)
            if ck in existing:
                try:
                    cached = json.load(open(cp))
                    if any(len(r.get("samples", [])) > 0 for r in cached.get("rollouts", [])):
                        records.append(cached)
                        continue
                    print(f"[RETRY] paper {gi}: checkpoint empty, re-running")
                except Exception:
                    pass
            row = df.iloc[li]
            row = row.copy()
            row["row_id"] = gi
            paper = as_text(row.get(a.text_col))
            gt = as_text(row.get(a.gt_col))
            if len(paper) < 100:
                print(f"[SKIP] paper {gi}: text too short")
                continue
            paper_tr = self.llm.truncate(paper, a.paper_token_limit, suffix=" ... [TRUNCATED]")
            cit_raw = row.get(a.citation_col, "") if a.citation_col in df.columns else ""
            citation = self.llm.truncate(extract_intro_and_abstract(cit_raw, style="rollout", top_k=8),
                                         a.citation_token_limit, suffix=" ... [TRUNCATED]")
            rec = {"paper_idx": int(gi), "system": a.system, "teacher": a.teacher_model,
                   "paper_text": paper_tr, "citation_text": citation, "ground_truth": gt, "rollouts": []}
            if a.system == "novagents":
                try:
                    extra, rollouts = self.nov_paper(gi, row, gt)
                    rec.update(extra)
                    rec["rollouts"] = rollouts
                except Exception as e:  # noqa: BLE001
                    print(f"  [ERROR] paper {gi}: {e}")
                    rec["rollouts"].append({"rollout_id": 0, "error": str(e), "samples": []})
            else:
                evidence = self.llm.truncate(build_evidence_tagged(paper_tr), a.paper_token_limit,
                                             suffix=" ... [TRUNCATED]")
                for rc in self.rollout_cfgs:
                    t0 = time.time()
                    try:
                        res = self.lim_rollout(gi, paper_tr, evidence, citation, gt, rc)
                        rec["rollouts"].append({"rollout_id": rc["rollout_id"], "rollout_config": rc,
                                                "mode": res["mode"], "samples": res["samples"],
                                                "final_master_output": res["final_master_output"],
                                                "worker_final": res["worker_final"]})
                        print(f"  paper {gi} rollout {rc['rollout_id']} ({rc['mode']}) "
                              f"{time.time() - t0:.0f}s, {len(res['samples'])} samples", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [ERROR] paper {gi} rollout {rc['rollout_id']}: {e}")
                        rec["rollouts"].append({"rollout_id": rc["rollout_id"], "rollout_config": rc,
                                                "mode": rc["mode"], "error": str(e), "samples": [],
                                                "final_master_output": "", "worker_final": {}})
            json.dump(rec, open(cp, "w"), indent=2)
            records.append(rec)
            if (li + 1) % 5 == 0:
                consolidate(records, out_dir)
        consolidate(records, out_dir)
        print(f"Done. Rollouts in {out_dir}")


def consolidate(records, out_dir):
    json.dump(records, open(os.path.join(out_dir, "rollout_data_full.json"), "w"), indent=2)
    split = {"worker": [], "leader": [], "master": []}
    for p in records:
        for r in p.get("rollouts", []):
            for s in r.get("samples", []):
                if s.get("agent_role") in split:
                    split[s["agent_role"]].append(s)
    for k, v in split.items():
        json.dump(v, open(os.path.join(out_dir, f"rollout_data_{k}.json"), "w"), indent=2)
    print(f"  workers:{len(split['worker'])} leader:{len(split['leader'])} master:{len(split['master'])}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default="limagents", choices=["limagents", "novagents"])
    p.add_argument("--teacher_model", required=True, help="served name | gpt-4o-mini | mock")
    p.add_argument("--teacher_base_url", default=None)
    p.add_argument("--worker_model", default=None, help="served LoRA name for workers (iterated on-policy)")
    p.add_argument("--leader_model", default=None)
    p.add_argument("--master_model", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--family", default="auto")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--input_csv", default=CFG.DEFAULT_INPUT_CSV)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--start_row", type=int, default=300)
    p.add_argument("--end_row", type=int, default=450)
    p.add_argument("--n_rollouts", type=int, default=4)
    p.add_argument("--worker_threads", type=int, default=6)
    p.add_argument("--paper_token_limit", type=int, default=8000)
    p.add_argument("--citation_token_limit", type=int, default=4000)
    p.add_argument("--apply_master_tot", type=int, default=1)
    p.add_argument("--text_col", default=CFG.COLS["text"])
    p.add_argument("--gt_col", default=CFG.COLS["gt"])
    p.add_argument("--citation_col", default=CFG.COLS["citation_rollout"])
    args = p.parse_args(argv)
    Roller(args).run()


if __name__ == "__main__":
    main()
