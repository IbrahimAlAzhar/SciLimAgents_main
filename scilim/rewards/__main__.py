"""
Score rollouts and build SFT / DPO datasets.

    python -m scilim.rewards --reward stepwise --rollout_json R --out_dir D [--judge_model gpt-4o-mini]
    python -m scilim.rewards --reward outcome  ...
    python -m scilim.rewards --reward jsd      ... --student BASE --teacher TEACHER|self [--student_adapter A]
    python -m scilim.rewards --reward combined ... --jsd_scores J --stepwise_scores S [--outcome_scores O]
    python -m scilim.rewards --reward synthetic --natural_dpo D/dpo_dataset_worker.json --rollout_json R
    python -m scilim.rewards --jsd_selftest

Every reward writes the same layout into --out_dir:
    stepwise_candidate_scores.{json,csv}   per-candidate scores
    sft_dataset_{worker,leader,master}.json
    dpo_dataset_worker.json  (+ dpo_dataset_worker_augmented.json via --reward synthetic)
    per_paper/               resume shards
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import common as K


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reward", choices=["stepwise", "outcome", "jsd", "combined", "synthetic"])
    p.add_argument("--rollout_json")
    p.add_argument("--out_dir")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no_resume", action="store_true")
    # judge
    p.add_argument("--judge_model", default="gpt-4o-mini", help="gpt-4o-mini | served name | none")
    p.add_argument("--judge_base_url", default=None)
    p.add_argument("--no_llm_judge", action="store_true")
    # thresholds
    p.add_argument("--sft_min", type=float, default=None)
    p.add_argument("--master_sft_min", type=float, default=0.0)
    p.add_argument("--dpo_min_gap", type=float, default=None)
    p.add_argument("--stepwise_weight", type=float, default=0.70)
    p.add_argument("--gt_weight", type=float, default=0.30)
    p.add_argument("--leader_feedback_sft", action="store_true", help="outcome: add best MODE-A feedback to leader SFT")
    # jsd
    p.add_argument("--student")
    p.add_argument("--teacher", default=None, help="teacher path | self")
    p.add_argument("--student_adapter", default="")
    p.add_argument("--teacher_4bit", action="store_true")
    p.add_argument("--max_prompt_tokens", type=int, default=3072)
    p.add_argument("--max_completion_tokens", type=int, default=512)
    p.add_argument("--blend_stepwise", type=float, default=0.0)
    p.add_argument("--stepwise_scores", default=None)
    p.add_argument("--jsd_selftest", action="store_true")
    # combined
    p.add_argument("--jsd_scores", default=None)
    p.add_argument("--outcome_scores", default=None)
    p.add_argument("--w_jsd", type=float, default=0.25)
    p.add_argument("--w_judge", type=float, default=0.50)
    p.add_argument("--w_ground", type=float, default=0.25)
    p.add_argument("--w_outcome", type=float, default=0.0)
    p.add_argument("--normalize", default="minmax", choices=["minmax", "zscore", "none"])
    p.add_argument("--cascade_top_k", type=int, default=None)
    p.add_argument("--no_judge_calls", action="store_true")
    # synthetic
    p.add_argument("--natural_dpo", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--natural_frac", type=float, default=0.70)
    p.add_argument("--gen_model", default="gpt-4o-mini")
    p.add_argument("--gen_base_url", default=None)
    p.add_argument("--no_api", action="store_true")
    p.add_argument("--allow_fallback", action="store_true")
    a = p.parse_args(argv)

    if a.jsd_selftest:
        from .jsd import selftest
        selftest()
        return
    if a.reward == "synthetic":
        from .synthetic import build_augmented
        out = a.out or str(Path(a.natural_dpo).with_name("dpo_dataset_worker_augmented.json"))
        build_augmented(a.natural_dpo, a.rollout_json, out, a.natural_frac, a.dpo_min_gap, a.gen_model,
                        a.gen_base_url, a.no_api, a.allow_fallback)
        return

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a.cache_dir = a.cache_dir or str(out_dir.parent / "reward_cache")
    papers = json.loads(Path(a.rollout_json).read_text())
    if a.limit:
        papers = papers[: a.limit]
    print(f"[{a.reward}] {len(papers)} papers from {a.rollout_json} -> {out_dir}")
    cols = ["paper_idx", "agent_role", "agent_name", "mode", "rollout_id", "stepwise_score", "gt_similarity",
            "final_score", "search", "retrieve", "synthesize", "n_units"]

    if a.reward == "stepwise":
        from .stepwise import StepwiseScorer
        sc = StepwiseScorer(K.make_judge(a), use_llm=not a.no_llm_judge, stepwise_weight=a.stepwise_weight,
                            gt_weight=a.gt_weight, sft_min=0.30 if a.sft_min is None else a.sft_min,
                            master_sft_min=a.master_sft_min,
                            dpo_min_gap=0.05 if a.dpo_min_gap is None else a.dpo_min_gap)
        cands, sft, dpo = K.run_sharded(papers, sc.score_paper, out_dir, not a.no_resume)
    elif a.reward == "outcome":
        from .outcome import OutcomeScorer
        judge = K.make_judge(a)
        if not judge.available:
            print("[WARN] outcome reward needs an LLM judge; only cached judgments will be used.")
        sc = OutcomeScorer(judge, sft_min=0.15 if a.sft_min is None else a.sft_min,
                           dpo_min_gap=0.05 if a.dpo_min_gap is None else a.dpo_min_gap,
                           leader_feedback_sft=a.leader_feedback_sft)
        cands, sft, dpo = K.run_sharded(papers, sc.score_paper, out_dir, not a.no_resume)
        cols = ["paper_idx", "agent_role", "agent_name", "mode", "rollout_id", "final_score", "f1", "precision",
                "recall", "grounding", "specificity", "severity_avg", "rouge_l_f1", "length_penalty",
                "redundancy_penalty", "boilerplate_ratio", "n_units"]
    elif a.reward == "jsd":
        from .jsd import JSDScorer, JSDSelector
        lookup = {}
        if a.stepwise_scores and Path(a.stepwise_scores).exists():
            for c in json.loads(Path(a.stepwise_scores).read_text()):
                lookup[(c.get("paper_idx"), c.get("agent_role"), c.get("agent_name"), c.get("rollout_id"))] = \
                    c.get("stepwise_score")
        holder = {}

        def score_paper(paper):
            if "sel" not in holder:
                holder["sel"] = JSDSelector(
                    JSDScorer(a.student, a.teacher, a.student_adapter, a.teacher_4bit, a.max_prompt_tokens,
                              a.max_completion_tokens),
                    lookup, a.blend_stepwise, 0.0 if a.sft_min is None else a.sft_min,
                    0.02 if a.dpo_min_gap is None else a.dpo_min_gap)
            return holder["sel"].score_paper(paper)
        cands, sft, dpo = K.run_sharded(papers, score_paper, out_dir, not a.no_resume)
        cols = ["paper_idx", "agent_role", "agent_name", "mode", "rollout_id", "jsd", "jsd_reward", "teacher_nll",
                "student_nll", "n_tokens", "stepwise_score", "final_score"]
    else:
        from .combined import run_combined
        cands, sft, dpo = run_combined(papers, a.jsd_scores, a.stepwise_scores, out_dir, K.make_judge(a),
                                       a.w_jsd, a.w_judge, a.w_ground, a.normalize, a.cascade_top_k,
                                       0.05 if a.dpo_min_gap is None else a.dpo_min_gap,
                                       0.0 if a.sft_min is None else a.sft_min, a.no_judge_calls,
                                       a.outcome_scores, a.w_outcome)
        cols = ["paper_idx", "agent_role", "agent_name", "mode", "rollout_id", "jsd", "stage1_jsd_reward",
                "stage2_judge", "grounding", "outcome", "jsd_norm", "judge_norm", "ground_norm", "outcome_norm",
                "final_score", "stages_used", "cascade_selected"]
    K.write_outputs(out_dir, cands, sft, dpo, cols)
    K.diagnostics(f"{a.reward.upper()} REWARD DIAGNOSTICS", cands, sft, dpo)
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
