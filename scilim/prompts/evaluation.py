"""Evaluation prompts (pairwise GT-vs-LLM matching + false-positive classifier)."""
from ._raw import FP_SYSTEM_PROMPT  # noqa: F401


def pair_eval_prompt(gt_text: str, llm_text: str) -> str:
    """Identical to every eval.py / merge_and_eval.py in the original code."""
    return (
        "Check whether 'list2' contains a topic or limitation from 'list1' "
        "or 'list1' contains a topic or limitation from 'list2'.\n\n"
        "Your answer should be \"Yes\" or \"No\".\n"
        f"List 1: ground truth limitations: {gt_text}\n"
        f"List 2: llm generated limitations: {llm_text}\n"
    )


def fp_user_prompt(llm_limitation: str, manuscript_text: str) -> str:
    return f"""You are given the FULL TEXT of a paper (MANUSCRIPT) and a single generated
LIMITATION UNIT (LU) that was NOT matched to any reference limitation.

Classify the LU into EXACTLY ONE of these categories:
1 = valid and grounded but missing from the reference (clearly supported by the manuscript)
2 = valid but excessively broad (defensible, but too generic / not specific to this paper)
3 = partially supported (some grounding in the manuscript, but incomplete or overstated)
4 = unsupported or hallucinated (not supported by the manuscript; fabricated)
5 = irrelevant / off-target (does not pertain to this manuscript's content)
6 = redundant with another limitation (essentially restates a point already covered)

Return ONLY a JSON object with EXACTLY these keys:
- "category": integer 1-6
- "valid": "Yes" if category is 1 or 2, otherwise "No"
- "evidence": a direct quote or specific reference from the MANUSCRIPT supporting the LU
             (REQUIRED and non-empty when category is 1 or 2; empty string otherwise)
- "reasoning": one brief sentence justifying the category

LIMITATION UNIT (LU):
{llm_limitation}

MANUSCRIPT (input_text_cleaned):
{manuscript_text}
"""


CATEGORY_LABELS = {
    1: "valid_grounded_missing",
    2: "valid_excessively_broad",
    3: "partially_supported",
    4: "unsupported_hallucinated",
    5: "irrelevant_offtarget",
    6: "redundant",
}
VALID_CATEGORIES = {1, 2}
