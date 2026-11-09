"""
Single-call baselines (NEW — no counterpart existed in the original code).

zero_shot : one LLM call, paper text only.
rag       : one LLM call, paper text + retrieved/cited-paper evidence block
            (built by scilim.utils.retrieval exactly like the shared-memory /
            retrieval_heavy context).
rag_only  : one LLM call, retrieved/cited evidence + paper ABSTRACT only
            (tests how much the retrieved text alone carries).

Output format mirrors the Master prompt so the same evaluation parser applies.
"""

ZERO_SHOT_SYSTEM = "You are an expert scientific reviewer."

_FORMAT = (
    "OUTPUT FORMAT:\n"
    "Start with: \"Here is the consolidated list of key limitations identified in the paper:\"\n"
    "Then a bulleted list:\n"
    "- **Category:** Specific limitation (with brief explanation and evidence)."
)


def zero_shot_prompt(paper_content: str) -> str:
    return f"""Identify the key limitations of the scientific paper below.
Consider novelty and significance, theoretical and methodological soundness, experimental evaluation (baselines, metrics, statistical rigor), generalization/robustness/efficiency, clarity/interpretability/reproducibility, and data/ethics.
Each limitation must be specific to this paper and grounded in its content. Aim for 10-20 strong limitations.

{_FORMAT}

PAPER CONTENT:
{paper_content}"""


def rag_prompt(paper_content: str, retrieved_block: str) -> str:
    return f"""Identify the key limitations of the scientific paper below.
Consider novelty and significance, theoretical and methodological soundness, experimental evaluation (baselines, metrics, statistical rigor), generalization/robustness/efficiency, clarity/interpretability/reproducibility, and data/ethics.
Use the RETRIEVED / CITED CONTEXT (abstracts and introductions of related and cited papers) to test novelty claims, check whether the right baselines were used, and spot claims prior work already established or contradicts. When a limitation rests on this evidence, cite it as (Ref: PaperN). The retrieved papers are NOT the paper under review; do not list limitations of the retrieved papers themselves.
Each limitation must be specific to this paper and grounded in its content. Aim for 10-20 strong limitations.

{_FORMAT}

PAPER CONTENT:
{paper_content}
{retrieved_block}"""


def rag_only_prompt(abstract: str, retrieved_block: str) -> str:
    return f"""You are given only the ABSTRACT of a scientific paper and the retrieved / cited context (abstracts and introductions of related papers).
Using only this information, identify the most likely limitations of the paper, especially regarding novelty relative to prior work, choice of baselines, and claims that prior work already established or contradicts. When a limitation rests on the retrieved evidence, cite it as (Ref: PaperN).
Aim for 8-15 limitations.

{_FORMAT}

PAPER ABSTRACT:
{abstract}
{retrieved_block}"""
