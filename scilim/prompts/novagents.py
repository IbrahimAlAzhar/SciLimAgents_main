"""
NovAgents: novelty & significance specialists (Paper A vs retrieved Paper B
summaries) -> Master. (source: novagents/novagents.py)
"""
from ._novagents_raw import (  # noqa: F401  (verbatim strings)
    HARSH_REVIEWER_POLICY, WORKER_FEEDBACK_INSTRUCTION,
    PROMPT_1_NOVELTY_TECH, PROMPT_2_EXPERIMENTS, PROMPT_3_LIT_REVIEW,
    PROMPT_4_SCOPE_GENERALIZABILITY, PROMPT_5_CLAIMS_OVERCLAIMING,
    PROMPT_6_METHOD_CLARITY_RIGOR, MASTER_AGENT_PROMPT,
)

# Agent name -> specialist prompt (+ feedback instruction), original order.
SPECIALIST_AGENTS = [
    ("Literature_Review_and_Data_Analysis_Agent",
     PROMPT_3_LIT_REVIEW + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Hypothesis_Refinement_and_Critical_Reflection_Agent",
     PROMPT_1_NOVELTY_TECH + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Methodological_Novelty_Agent",
     PROMPT_6_METHOD_CLARITY_RIGOR + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Experimental_Novelty_Agent",
     PROMPT_2_EXPERIMENTS + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Problem_Formulation_Novelty_Agent",
     PROMPT_4_SCOPE_GENERALIZABILITY + "\n" + WORKER_FEEDBACK_INSTRUCTION),
    ("Writing_Claim_Novelty_Agent",
     PROMPT_5_CLAIMS_OVERCLAIMING + "\n" + WORKER_FEEDBACK_INSTRUCTION),
]
SPECIALIST_BY_NAME = dict(SPECIALIST_AGENTS)

# Which LimAgents specialty each NovAgent maps to (used to route ground truth
# when scoring NovAgent rollouts). New: needed for NovAgents DPO/GRPO.
NOV_SPECIALTY = {
    "Literature_Review_and_Data_Analysis_Agent": "novelty_significance",
    "Hypothesis_Refinement_and_Critical_Reflection_Agent": "novelty_significance",
    "Methodological_Novelty_Agent": "theoretical_methodological",
    "Experimental_Novelty_Agent": "experimental_evaluation",
    "Problem_Formulation_Novelty_Agent": "generalization_robustness_efficiency",
    "Writing_Claim_Novelty_Agent": "novelty_significance",
}
NOV_SPECIALTY_LABEL = {
    "Literature_Review_and_Data_Analysis_Agent": "literature review & contextualization",
    "Hypothesis_Refinement_and_Critical_Reflection_Agent": "technical contributions & incremental nature",
    "Methodological_Novelty_Agent": "methodological clarity & rigor",
    "Experimental_Novelty_Agent": "experimental validation & comparative analysis",
    "Problem_Formulation_Novelty_Agent": "scope of analysis & generalizability",
    "Writing_Claim_Novelty_Agent": "claim accuracy & overclaiming",
}

SUMMARISER_SYSTEM = ("You are a concise academic summariser. Output a short summary "
                     "suitable for limitations comparison.")


def summariser_user(raw_text: str) -> str:
    return f"Summarize the following paper:\n{raw_text}"


MISSING_B_SUMMARY = "Paper B Summary:\n(Missing item.)"


def specialist_system(agent_name: str) -> str:
    return HARSH_REVIEWER_POLICY + "\n\n" + SPECIALIST_BY_NAME[agent_name]


def specialist_user(combined_input: str) -> str:
    return (
        f"Analyze the following paper for limitations in your domain. "
        f"Follow the output format strictly.\n\n{combined_input}"
    )


def build_dual_paper_input(main_text: str, b_summaries_text: str) -> str:
    """Inputs must already be truncated by the caller."""
    return (
        f"=== MAIN PAPER (A) ===\n{main_text}\n\n"
        f"=== RELEVANT PAPERS (B) ===\n{b_summaries_text}"
    ).strip()


def combine_b_summaries(summaries) -> str:
    return "\n\n".join(
        [f"--- Retrieved Paper #{i+1} ---\n{s}" for i, s in enumerate(summaries)]
    ).strip()


def handoff_block(specialist_outputs) -> str:
    parts = [f"[{name}]\n{out}" for name, out in specialist_outputs]
    return ("=== MASTER_HANDOFF_START ===\n" + "\n\n".join(parts)
            + "\n=== MASTER_HANDOFF_END ===")


def master_user(handoff: str) -> str:
    return (
        f"Here are the specialist limitation analyses:\n\n"
        f"{handoff}\n\n"
        f"Produce the final consolidated novelty & significance limitations report."
    )
