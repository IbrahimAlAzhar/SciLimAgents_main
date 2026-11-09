"""
COMBINED reward (two_stage_reward.py): normalize each signal across all
candidates, then a weighted mean over the components present:

    score = (w_jsd * JSD_n + w_judge * judge_n + w_ground * grounding_n) / sum(w present)
    defaults 0.25 / 0.50 / 0.25, min-max normalization

stage 1  JSD reward          (<jsd_dir>/stepwise_candidate_scores.json)
stage 2  LLM-judge GT score  (<stepwise_dir>/stepwise_candidate_scores.json: gt_similarity)
grounding stepwise score     (same file: stepwise_score)
The outcome reward can replace/extend the judge via --outcome_scores.
Missing stage-2 scores are computed on demand (cached); --cascade_top_k K
judges only the top-K by JSD per group.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from . import common as K
from . import stepwise as S

KEY = ("paper_idx", "agent_role", "agent_name", "rollout_id")


def ckey(r):
    return tuple(r.get(f) for f in KEY)


def build_normalizer(values, mode):
    vals = [v for v in values if v is not None]
    if not vals or mode == "none":
        return (lambda x: x), "identity"
    if mode == "zscore":
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if sd == 0:
            return (lambda x: 0.5), "zscore(sd=0)"
        return (lambda x: 1.0 / (1.0 + math.exp(-(x - mu) / sd))), f"zscore(mu={mu:.4f}, sd={sd:.4f})"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return (lambda x: 0.5), "minmax(flat)"
    return (lambda x: (x - lo) / (hi - lo)), f"minmax(lo={lo:.4f}, hi={hi:.4f})"


def pearson(xs, ys):
    pts = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pts) < 3:
        return float("nan")
    x2, y2 = zip(*pts)
    mx, my = statistics.mean(x2), statistics.mean(y2)
    num = sum((a - mx) * (b - my) for a, b in pts)
    den = math.sqrt(sum((a - mx) ** 2 for a in x2) * sum((b - my) ** 2 for b in y2))
    return num / den if den else float("nan")


def run_combined(papers, jsd_scores, stepwise_scores, out_dir: Path, judge, w_jsd=0.25, w_judge=0.50,
                 w_ground=0.25, normalize="minmax", cascade_top_k=None, min_gap=0.05, sft_min=0.0,
                 no_judge_calls=False, outcome_scores=None, w_outcome=0.0):
    load = lambda p: json.loads(Path(p).read_text()) if p and Path(p).exists() else []
    jsd_by = {ckey(r): r for r in load(jsd_scores)}
    step_by = {ckey(r): r for r in load(stepwise_scores)}
    out_by = {ckey(r): r for r in load(outcome_scores) if r.get("agent_role") != "leader_feedback"}
    base = jsd_by or step_by or out_by
    print(f"[combined] jsd={len(jsd_by)} stepwise={len(step_by)} outcome={len(out_by)} candidates")
    if not base:
        raise SystemExit("No candidate scores found — run the jsd/stepwise/outcome rewards first.")
    keys = set(jsd_by) | set(step_by) | set(out_by)
    pinfo = {p.get("paper_idx"): p for p in papers}
    groups = defaultdict(list)
    for k in keys:
        jr, sr, orr = jsd_by.get(k, {}), step_by.get(k, {}), out_by.get(k, {})
        any_r = jr or sr or orr
        groups[k[:3]].append({
            "paper_idx": k[0], "agent_role": k[1], "agent_name": k[2], "rollout_id": k[3],
            "mode": any_r.get("mode"), "output": any_r.get("output", ""), "system": any_r.get("system"),
            "input_messages": jr.get("input_messages") or sr.get("input_messages") or orr.get("input_messages") or [],
            "jsd": jr.get("jsd"), "stage1_jsd_reward": jr.get("jsd_reward"),
            "stage2_judge": sr.get("gt_similarity"), "grounding": sr.get("stepwise_score"),
            "outcome": orr.get("score")})
    to_judge = []
    for g in groups.values():
        g.sort(key=lambda c: (c["stage1_jsd_reward"] is None, -(c["stage1_jsd_reward"] or 0)))
        elig = g if cascade_top_k is None else g[:cascade_top_k]
        for c in g:
            c["cascade_selected"] = c in elig
        to_judge += [c for c in elig if c["stage2_judge"] is None and w_judge > 0]
    if to_judge and not no_judge_calls:
        print(f"[combined] computing {len(to_judge)} missing judge scores")
        for c in to_judge:
            p = pinfo.get(c["paper_idx"], {})
            gt = p.get("ground_truth", "") or ""
            items = S.split_gt(gt)
            if c["agent_role"] == "worker":
                spec = K.specialty_of(c.get("system") or p.get("system", "limagents"), c["agent_name"])
                routes = S.route_gt_to_specialties(items) if items else {}
                c["stage2_judge"] = S.worker_gt_alignment(judge, c["output"], routes.get(spec) or items, spec)
            else:
                c["stage2_judge"] = S.gt_similarity(c["output"], gt)
    allc = [c for g in groups.values() for c in g]
    fj, dj = build_normalizer([c["stage1_jsd_reward"] for c in allc], normalize)
    fu, du = build_normalizer([c["stage2_judge"] for c in allc], normalize)
    fg, dg = build_normalizer([c["grounding"] for c in allc], normalize)
    fo, do = build_normalizer([c["outcome"] for c in allc], normalize)
    print(f"normalizers: jsd {dj} | judge {du} | grounding {dg} | outcome {do}")
    for c in allc:
        comps = {"jsd": (w_jsd, c["stage1_jsd_reward"], fj), "judge": (w_judge, c["stage2_judge"], fu),
                 "ground": (w_ground, c["grounding"], fg), "outcome": (w_outcome, c["outcome"], fo)}
        num = den = 0.0
        used = []
        for name, (w, v, f) in comps.items():
            c[f"{name}_norm"] = f(v) if v is not None else None
            if v is not None and w > 0:
                num += w * f(v)
                den += w
                used.append(name)
        c["score"] = round(num / den, 6) if den else None
        c["stages_used"] = "+".join(used)
        c["score_detail"] = {"total": c["score"], "steps": {k: c.get(k) for k in
                             ("jsd", "stage1_jsd_reward", "stage2_judge", "grounding", "outcome")},
                             "n_units": len(K.split_bullets(c.get("output", "")))}
    allc = [c for c in allc if c["score"] is not None]
    by_paper = defaultdict(list)
    for c in allc:
        by_paper[c["paper_idx"]].append(c)
    shards = {}
    for pid, cands in by_paper.items():
        paper = pinfo.get(pid, {"paper_idx": pid})
        for c in cands:
            if c["agent_role"] == "worker" and paper.get("rollouts"):
                ro = next((r for r in paper["rollouts"] if r.get("rollout_id") == c["rollout_id"]), None)
                if ro:
                    c["_tr"] = K.extract_worker_final(ro).get(c["agent_name"])
        clean, sft, dpo = S.select_from_candidates(paper, cands, sft_min, 0.0, min_gap)
        K.write_shard(out_dir / K.SHARD_SUBDIR, pid, clean, sft, dpo)
        shards[pid] = {"candidates": clean, "sft": sft, "dpo": dpo}
    cands, sft, dpo = K.merge_shards(shards)
    print(f"Pearson r(JSD, judge) = {pearson([c['stage1_jsd_reward'] for c in cands], [c['stage2_judge'] for c in cands]):+.3f}"
          "  (near 0: complementary; near 1: redundant; negative: they disagree)")
    return cands, sft, dpo
