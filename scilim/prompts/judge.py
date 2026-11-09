"""
Reward-model / judge prompts.

* Outcome reward (LLM-as-judge, reward_model_no_need.py + worker_grpo.py):
  GT categorisation, bipartite matching, per-item quality, leader-feedback
  quality, master organisation.
* Stepwise reward GT-alignment judge (stepwise_reward_select.py).
* Synthetic "weak" rejection generator (build_synthetic_dpo.py).

The schema hints in reward_model_no_need.py used doubled braces ("{{") that were
never .format()-ed, so the judge literally saw "{{". worker_grpo.py fixed that;
the fixed (single-brace) versions are the defaults. The RM_* originals remain
importable from ._raw for exact reproduction.
"""
from ._raw import (  # noqa: F401
    GRPO_GT_CAT_PROMPT as GT_CAT_PROMPT,
    GRPO_MATCH_PROMPT as MATCH_PROMPT,
    GRPO_GT_CAT_SCHEMA as GT_CAT_SCHEMA,
    GRPO_MATCH_SCHEMA as MATCH_SCHEMA,
    GRPO_QUALITY_SCHEMA as QUALITY_SCHEMA,
    RM_QUALITY_PROMPT as QUALITY_PROMPT,          # em-dash wording (reward_model.py)
    GRPO_QUALITY_PROMPT as QUALITY_PROMPT_GRPO,   # colon wording (worker_grpo.py)
    RM_FEEDBACK_QUALITY_PROMPT as FEEDBACK_QUALITY_PROMPT,
    RM_ORG_PROMPT as ORG_PROMPT,
    SYNTH_WEAK_SYSTEM,
)

FEEDBACK_SCHEMA = ('Return JSON: {"actionable": <float 0..1>, '
                   '"accurate": <float 0..1>, "targeted": <float 0..1>}')
ORG_SCHEMA = ('Return JSON: {"organization": <float 0..1>, '
              '"dedup": <float 0..1>, "coherence": <float 0..1>}')

JUDGE_JSON_SYSTEM = ("You are a careful, strict scientific reviewer. "
                     "Output valid JSON only, no prose, no code fences.")

# ---------------------------------------------------------------------------
# Stepwise reward: worker <-> gold alignment judge
# ---------------------------------------------------------------------------
ALIGN_SYSTEM = "You are a strict, fair scientific reviewer. JSON only."


def align_prompt(specialty: str, gt_slice_text: str, worker_output: str) -> str:
    return (
        f"You are judging a specialist reviewer whose area is: {specialty}.\n\n"
        f"GOLD limitations (reference, for this area):\n{gt_slice_text[:2000]}\n\n"
        f"REVIEWER's limitations:\n{worker_output[:4000]}\n\n"
        "Rate how well the reviewer's limitations ALIGN with the gold ones — do they "
        "capture the same substantive flaws (regardless of wording)? Penalize misses "
        "and off-target/unsupported points. Return ONLY JSON: "
        '{"alignment": <float 0..1>}'
    )


# ---------------------------------------------------------------------------
# Synthetic weak rejection (TYPE B augmented DPO)
# ---------------------------------------------------------------------------
def synth_weak_user(paper_excerpt: str, specialty: str, chosen: str,
                    n_bullets: int, uses_evidence_ids: bool) -> str:
    style = (
        "Cite invented evidence IDs in the form [E##] to match the good answer's "
        "style — the numbers must NOT correspond to anything real in the paper.\n"
        if uses_evidence_ids else
        "Do not use evidence-ID markers.\n"
    )
    return (
        f"SPECIALTY AREA: {specialty}\n\n"
        f"PAPER EXCERPT (the only real source material):\n{paper_excerpt}\n\n"
        f"A STRONG, WELL-GROUNDED review of this paper in that area looks like:\n{chosen}\n\n"
        "Now write a WEAK version: it must read as fluent, confident, on-topic "
        "peer review, but must fail on substance. Requirements:\n"
        f"- Exactly {n_bullets} bullets, same markdown bullet format, similar length "
        "per bullet as the strong version.\n"
        "- Stay inside the specialty area above so it looks like the right kind of critique.\n"
        "- Make the criticisms GENERIC — they should apply to almost any paper in the "
        "field and not to this paper's actual content.\n"
        "- FABRICATE specifics: dataset names, benchmark names, baseline methods, "
        "percentages, and table/section references that do NOT appear in the excerpt.\n"
        "- Where you do reference the paper, misattribute it: claim it did something it "
        "did not, or omitted something it actually reported.\n"
        "- Provide no real quotes or verifiable anchors from the excerpt.\n"
        f"{style}"
        "Return ONLY the bullet list."
    )
