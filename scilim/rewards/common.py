"""
Shared reward machinery: judge client (with disk cache), candidate extraction
from rollouts, SFT/DPO selection, per-paper resume shards, outputs.
(ported from stepwise_reward_select.py; behaviour unchanged for LimAgents,
extended to NovAgents rollouts)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm.client import ChatClient, is_openai_model, parse_json_obj
from ..prompts import canonical as C
from ..prompts import novagents as N
from ..utils.text import split_bullets, strip_think  # noqa: F401  (re-exported)

SHARD_SUBDIR = "per_paper"
CANON_PROMPT_CHARS = 20000


# =============================================================================
# Judge (gpt-4o-mini by default, or any local served model)
# =============================================================================
class Judge:
    """JSON-returning LLM judge with a disk cache (re-runs are free)."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: Optional[str] = None,
                 cache_dir: str = "reward_cache", tokenizer: Optional[str] = None,
                 max_model_len: int = 32768, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client = None
        if enabled:
            try:
                self.client = ChatClient(model=model, base_url=base_url or None, tokenizer=tokenizer,
                                         max_model_len=max_model_len, timeout=180, max_retries=3)
            except Exception as e:  # noqa: BLE001
                print(f"[judge] disabled ({e})")
                self.enabled = False

    @property
    def available(self) -> bool:
        return self.enabled and self.client is not None

    def json_call(self, tag: str, system: str, prompt: str, max_tokens: int = 1500) -> dict:
        key = hashlib.sha256(json.dumps([tag, self.model, system, prompt]).encode()).hexdigest()[:24]
        path = self.cache / f"{re.sub(r'[^A-Za-z0-9_]', '_', tag)[:40]}_{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                path.unlink(missing_ok=True)
        if not self.available:
            return {}
        for attempt in range(3):
            out = self.client.chat(system, prompt, max_tokens=max_tokens, temperature=0.0, top_p=1.0,
                                   json_mode=self.client.is_openai)
            obj = parse_json_obj(out)
            if obj:
                path.write_text(json.dumps(obj, indent=2))
                return obj
        print(f"  [JUDGE ERROR] {tag}: unparseable output")
        return {}

    def text_call(self, system: str, prompt: str, temperature: float, max_tokens: int) -> str:
        if not self.available:
            return ""
        return self.client.chat(system, prompt, max_tokens=max_tokens, temperature=temperature, top_p=1.0)


def make_judge(args) -> Judge:
    model = getattr(args, "judge_model", None) or "gpt-4o-mini"
    base = getattr(args, "judge_base_url", None)
    if model in ("none", "off"):
        return Judge(enabled=False, cache_dir=args.cache_dir)
    return Judge(model=model, base_url=None if is_openai_model(model) else base,
                 cache_dir=args.cache_dir, tokenizer=getattr(args, "judge_tokenizer", None),
                 enabled=not getattr(args, "no_llm_judge", False))


# =============================================================================
# Rollout -> candidates
# =============================================================================
def source_text_of(paper: dict) -> str:
    s = paper.get("paper_text", "") or ""
    if paper.get("citation_text"):
        s = s + "\n\n" + paper["citation_text"]
    return s


def extract_worker_final(rollout: dict) -> Dict[str, dict]:
    """Per worker: search / retrieve(query) / synthesize(final draft or last revision)."""
    hops, revised, drafts = defaultdict(dict), defaultdict(list), {}
    for s in rollout.get("samples", []):
        if s.get("agent_role") != "worker":
            continue
        name, tt = s.get("agent_name"), s.get("turn_type")
        if tt in ("hop_search", "hop_query", "hop_draft"):
            hops[name][tt] = s
        elif tt == "revised":
            revised[name].append(s)
        elif tt in ("draft", "initial"):
            drafts[name] = s
    out = {}
    for name in set(hops) | set(drafts):
        hs = hops.get(name, {})
        draft_s = hs.get("hop_draft") or drafts.get(name)
        final_s = max(revised[name], key=lambda x: x.get("critique_round", 0)) if revised.get(name) else draft_s
        out[name] = {
            "search": (hs.get("hop_search") or {}).get("output", ""),
            "retrieve": (hs.get("hop_query") or {}).get("output", ""),
            "synthesize": (final_s or {}).get("output", ""),
            "draft": (draft_s or {}).get("output", ""),
            "final_sample": final_s,
            "draft_sample": draft_s,
            "has_hops": bool(hs),
        }
    return out


def find_sample(rollout, role, turn):
    return next((s for s in rollout.get("samples", [])
                 if s.get("agent_role") == role and s.get("turn_type") == turn), None)


def canonical_worker_prompt(system: str, worker_name: str, paper: dict, tr: Optional[dict] = None,
                            user_style: str = "canonical") -> List[dict]:
    """Mode-free TRAINING prompt for a worker (exploration-mode text never leaks)."""
    if system == "novagents":
        s = (tr or {}).get("final_sample") or {}
        msgs = s.get("input_messages")
        if msgs:
            return msgs
        combined = paper.get("combined_input") or paper.get("paper_text", "")
        return [{"role": "system", "content": N.specialist_system(worker_name)},
                {"role": "user", "content": N.specialist_user(combined)}]
    return C.canonical_worker_messages(worker_name, source_text_of(paper), CANON_PROMPT_CHARS, user_style)


def specialty_of(system: str, worker_name: str) -> str:
    if system == "novagents":
        return N.NOV_SPECIALTY.get(worker_name, "novelty_significance")
    return C.WORKER_SPECIALTY.get(worker_name, "other")


def specialty_label(system: str, worker_name: str) -> str:
    if system == "novagents":
        return N.NOV_SPECIALTY_LABEL.get(worker_name, "novelty and significance")
    return C.WORKER_SPECIALTY_LABEL.get(worker_name, "the paper's limitations")


# =============================================================================
# Selection: SFT (best) and DPO (best vs worst) — stepwise_reward_select.py
# =============================================================================
def sft_best(cands, role, name, pid, sft, min_score=0.0):
    cands = [c for c in cands if (c.get("output") or "").strip()]
    if not cands:
        return
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = cands[0]
    if best["score"] < min_score:
        return
    prompt = best.get("input_messages") or []
    sft.append({
        "paper_idx": pid, "agent_role": role, "agent_name": name,
        "mode": best.get("mode"), "rollout_id": best.get("rollout_id"),
        "final_score": best["score"], "stepwise_score": best.get("stepwise_score"),
        "gt_similarity": best.get("gt_similarity"), "score_detail": best.get("score_detail"),
        "prompt": prompt, "output": best["output"],
        "messages": prompt + [{"role": "assistant", "content": best["output"]}],
    })


def select(cands, prompt, role, name, pid, sft, dpo, min_score=0.0, min_gap=0.05,
           chosen_floor: float = 0.0, sft_top_k: int = 1):
    cands = [c for c in cands if (c.get("output") or "").strip()]
    if not cands:
        return
    cands.sort(key=lambda c: c["score"], reverse=True)
    best = cands[0]
    if best["score"] < min_score:
        return
    for cand in cands[:max(1, sft_top_k)]:
        if cand["score"] < min_score:
            continue
        sft.append({
            "paper_idx": pid, "agent_role": role, "agent_name": name,
            "mode": cand.get("mode"), "rollout_id": cand.get("rollout_id"),
            "final_score": cand["score"], "stepwise_score": cand.get("stepwise_score"),
            "gt_similarity": cand.get("gt_similarity"), "score_detail": cand.get("score_detail"),
            "prompt": prompt, "output": cand["output"],
            "messages": prompt + [{"role": "assistant", "content": cand["output"]}],
        })
    if len(cands) < 2 or best["score"] < chosen_floor:
        return
    worst = cands[-1]
    gap = best["score"] - worst["score"]
    if gap < min_gap:
        return
    dpo.append({
        "paper_idx": pid, "agent_role": role, "agent_name": name, "prompt": prompt,
        "chosen": best["output"], "rejected": worst["output"],
        "chosen_mode": best.get("mode"), "rejected_mode": worst.get("mode"),
        "chosen_score": best["score"], "rejected_score": worst["score"],
        "chosen_stepwise": best.get("stepwise_score"), "rejected_stepwise": worst.get("stepwise_score"),
        "chosen_gt_similarity": best.get("gt_similarity"), "rejected_gt_similarity": worst.get("gt_similarity"),
        "chosen_rollout_id": best.get("rollout_id"), "rejected_rollout_id": worst.get("rollout_id"),
        "chosen_score_detail": best.get("score_detail"), "rejected_score_detail": worst.get("score_detail"),
        "score_gap": round(gap, 4),
    })


# =============================================================================
# Resume shards
# =============================================================================
def shard_path(shard_dir: Path, pid) -> Path:
    try:
        return shard_dir / f"paper_{int(pid):05d}.json"
    except (TypeError, ValueError):
        return shard_dir / f"paper_{pid}.json"


def write_shard(shard_dir: Path, pid, cands, sft_rows, dpo_rows):
    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp = shard_path(shard_dir, pid).with_suffix(".tmp")
    tmp.write_text(json.dumps({"paper_idx": pid, "candidates": cands, "sft": sft_rows, "dpo": dpo_rows},
                              indent=2, default=str))
    tmp.replace(shard_path(shard_dir, pid))


def load_shards(shard_dir: Path) -> dict:
    out = {}
    if not shard_dir.exists():
        return out
    for f in sorted(shard_dir.glob("paper_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  [resume] corrupt shard skipped: {f.name} ({e})")
            continue
        out[d.get("paper_idx")] = {"candidates": d.get("candidates", []), "sft": d.get("sft", []),
                                   "dpo": d.get("dpo", [])}
    return out


_AGGREGATE_SOURCES = {
    "candidates": ["stepwise_candidate_scores.json"],
    "sft": ["sft_dataset.json", "sft_dataset_worker.json", "sft_dataset_leader.json", "sft_dataset_master.json"],
    "dpo": ["dpo_dataset.json", "dpo_dataset_worker.json"],
}
_DEDUP_KEYS = {
    "candidates": ("paper_idx", "agent_role", "agent_name", "rollout_id", "mode"),
    "sft": ("paper_idx", "agent_role", "agent_name", "rollout_id"),
    "dpo": ("paper_idx", "agent_role", "agent_name"),
}


def _dedup(rows, section):
    seen, out = set(), []
    for r in rows:
        k = tuple(r.get(f) for f in _DEDUP_KEYS[section])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def backfill_shards_from_aggregates(out_dir: Path, shard_dir: Path, known: set) -> int:
    """Turn previously written aggregate JSONs (pre-shard runs) into shards."""
    by_paper = defaultdict(lambda: {"candidates": [], "sft": [], "dpo": []})
    for section, names in _AGGREGATE_SOURCES.items():
        combined = [n for n in names if n in ("stepwise_candidate_scores.json", "sft_dataset.json",
                                              "dpo_dataset.json") and (out_dir / n).exists()]
        picked = combined[:1] or [n for n in names if (out_dir / n).exists()]
        for n in picked:
            try:
                rows = json.loads((out_dir / n).read_text())
            except Exception:
                continue
            for r in rows if isinstance(rows, list) else []:
                if isinstance(r, dict) and "paper_idx" in r:
                    by_paper[r["paper_idx"]][section].append(r)
    written = 0
    for pid, sec in by_paper.items():
        if pid in known or not (sec["candidates"] or sec["sft"] or sec["dpo"]):
            continue
        write_shard(shard_dir, pid, _dedup(sec["candidates"], "candidates"),
                    _dedup(sec["sft"], "sft"), _dedup(sec["dpo"], "dpo"))
        written += 1
    return written


def merge_shards(shards: dict):
    cands, sft, dpo = [], [], []

    def _key(pid):
        try:
            return (0, int(pid))
        except (TypeError, ValueError):
            return (1, str(pid))
    for pid in sorted(shards, key=_key):
        cands.extend(shards[pid]["candidates"])
        sft.extend(shards[pid]["sft"])
        dpo.extend(shards[pid]["dpo"])
    return cands, sft, dpo


def write_outputs(out_dir: Path, cands, sft, dpo, csv_cols: List[str], scores_name="stepwise_candidate_scores"):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{scores_name}.json").write_text(json.dumps(cands, indent=2, default=str))
    with open(out_dir / f"{scores_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for c in cands:
            steps = (c.get("score_detail") or {}).get("steps", {}) or {}
            row = []
            for col in csv_cols:
                if col == "final_score":
                    row.append(c.get("score"))
                elif col in c:
                    row.append(c.get(col))
                elif col in steps:
                    row.append(steps.get(col))
                elif col == "n_units":
                    row.append((c.get("score_detail") or {}).get("n_units"))
                else:
                    row.append(None)
            w.writerow(row)
    (out_dir / "sft_dataset.json").write_text(json.dumps(sft, indent=2, default=str))
    for role in ("worker", "leader", "master"):
        (out_dir / f"sft_dataset_{role}.json").write_text(
            json.dumps([s for s in sft if s["agent_role"] == role], indent=2, default=str))
    (out_dir / "dpo_dataset.json").write_text(json.dumps(dpo, indent=2, default=str))
    (out_dir / "dpo_dataset_worker.json").write_text(
        json.dumps([d for d in dpo if d["agent_role"] == "worker"], indent=2, default=str))


def diagnostics(title, cands, sft, dpo):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    by_role = defaultdict(list)
    for c in cands:
        if c.get("score") is not None:
            by_role[c["agent_role"]].append(c["score"])
    print(f"\n{'Role':<10}{'Cand':>7}{'Mean':>8}{'Std':>8}{'Min':>8}{'Max':>8}")
    for role, v in sorted(by_role.items()):
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{role:<10}{len(v):>7}{statistics.mean(v):>8.3f}{sd:>8.3f}{min(v):>8.3f}{max(v):>8.3f}")
    sb, db, gaps = defaultdict(int), defaultdict(int), []
    for s in sft:
        sb[s["agent_role"]] += 1
    for d in dpo:
        db[d["agent_role"]] += 1
        gaps.append(d["score_gap"])
    print(f"\nSFT samples: {len(sft)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(sb.items())))
    print(f"DPO pairs:   {len(dpo)}  " + ", ".join(f"{k}:{v}" for k, v in sorted(db.items())))
    if gaps:
        print(f"DPO gaps: mean={statistics.mean(gaps):.3f} min={min(gaps):.3f} max={max(gaps):.3f}")
    print("=" * 70)


def run_sharded(papers, score_paper_fn, out_dir: Path, resume: bool = True):
    """Score papers one at a time, checkpointing a shard per paper."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):
            return x
    shard_dir = out_dir / SHARD_SUBDIR
    shards = {}
    if resume:
        shards = load_shards(shard_dir)
        n_back = backfill_shards_from_aggregates(out_dir, shard_dir, set(shards))
        if n_back:
            print(f"[resume] backfilled {n_back} papers from existing aggregates")
            shards = load_shards(shard_dir)
        todo = [p for p in papers if p.get("paper_idx") not in shards]
        print(f"[resume] {len(papers) - len(todo)} papers already scored, {len(todo)} to score")
    else:
        todo = papers
    for p in tqdm(todo, desc="Scoring papers"):
        c, s, d = score_paper_fn(p)
        write_shard(shard_dir, p.get("paper_idx"), c, s, d)
        shards[p.get("paper_idx")] = {"candidates": c, "sft": s, "dpo": d}
    return merge_shards(shards)
