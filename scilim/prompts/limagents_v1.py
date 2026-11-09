"""
LimAgents v1 prompts — the ORIGINAL 7-specialist + Master pipeline
(source: limagents/part_1/limagents.py, ablation/leave_one_out_limagents.py).

All prompt text is copied verbatim. Do not edit wording here: results in the
paper depend on it. New/experimental prompts live in their own modules.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# System prompts used for each call (verbatim from limagents.py)
# ---------------------------------------------------------------------------
SUMMARIZER_SYSTEM = "You are an expert scientific summarizer."

AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "Novelty_Significance_Agent": "You are an expert reviewer.",
    "Citation_Agent": "You are an expert citation analyst.",
    "Theoretical_Methodological_Agent": "You are an expert methodologist.",
    "Experimental_Evaluation_Agent": "You are an expert experimentalist.",
    "Generalization_Robustness_Efficiency_Agent": "You are an expert on robustness and efficiency.",
    "Clarity_Interpretability_Reproducibility_Agent": "You are an expert on clarity and reproducibility.",
    "Data_Ethics_Agent": "You are an expert on data and ethics.",
}
MASTER_SYSTEM = "You are the Master Agent."


# ---------------------------------------------------------------------------
# Relevant-paper summarisation (novelty context)
# ---------------------------------------------------------------------------
def summarization_prompt_for_relevant_paper(item_text: str, idx: int) -> str:
    return f"""You will summarize one relevant paper for novelty comparison. Produce a compact, structured summary focusing ONLY on the following dimensions:

1) Literature analysis
2) Data analysis
3) Hypothesis refinement and critical reflection
4) Methodological novelty
5) Experimental novelty
6) Problem formulation novelty
7) Writing/claim novelty

Rules:
- Use plain text only.
- Keep it concise and information-dense.
- If a dimension is not stated in the text, write "Not stated".
- Do not invent details.

Relevant paper #{idx} text:
{item_text}

Output format (STRICT):
Relevant Paper #{idx} Summary:
- Literature analysis: ...
- Data analysis: ...
- Hypothesis refinement and critical reflection: ...
- Methodological novelty: ...
- Experimental novelty: ...
- Problem formulation novelty: ...
- Writing/claim novelty: ...
""".strip()


# ---------------------------------------------------------------------------
# Specialist prompts
# ---------------------------------------------------------------------------
def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are a highly skeptical expert focused exclusively on limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for issues like rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters beyond a niche setting. Identify any unaddressed alternatives or ignored related problems that diminish the perceived impact.
The review_leader will ask for your feedback; respond thoroughly and ask clarifying questions if needed. When finished, inform the review_leader and provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
PAPER CONTENT:
{paper_content}"""


def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Citation Agent**.
Task: Compare Main Article to 'CITED PAPERS INFO'.
- Did the article fail to address insights from its citations?
- Check if the paper misinterprets or selectively cites prior work to make its own contribution look stronger.
- Output: "- [Limitation]: Explanation (Ref: Paper X)"
=== MAIN ARTICLE ===
{paper_content}
=== CITED PAPERS INFO ===
{citation_content}"""


def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws, unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, or failure to explain why the method works and which parts are critical.
Identify issues like unstated or overly strong assumptions, incomplete theoretical analysis, errors in derivations, methods that only work under restricted conditions not clearly acknowledged, missing ablations, lack of isolation of individual contributions, or ablations that do not convincingly attribute performance gains.
The review_leader will consult you; provide detailed critique and ask follow-up questions when necessary. When done, inform the review_leader and deliver a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""


def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in experimental evaluation, including validation, rigor, comparisons, baselines, and metrics. Find weaknesses in empirical support, such as insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate baselines, incomplete comparisons, misleading metrics, superficial analysis, or failure to validate claims comprehensively.
Highlight issues like small-scale experiments, missing error bars or confidence intervals, unreported failed experiments, outdated or weak baselines, missing key competitors, unfair hyperparameter tuning, reliance on misleading metrics, missing standard metrics, or overemphasis on minor gains without practical or statistical significance.
The review_leader will interact with you; respond critically and seek clarification if needed. When finished, inform the review_leader and provide a bullet list of experimental evaluation-related limitations, including validation, comparisons, baselines, and metrics.
PAPER CONTENT:
{paper_content}"""


def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings (e.g., different datasets, domains, noise, adversarial conditions), is practical in terms of resources (time, memory, hardware, scalability), and addresses genuine deployment needs without ignoring real-world constraints.
Point out limitations like overfitting to benchmarks, lack of out-of-distribution testing, sensitivity to hyperparameters, poor performance under shifts, excessive training/inference demands, high resource needs restricting deployment, reliance on synthetic data, ignoring constraints like cost or latency, lack of user studies or field tests, or over-optimistic assumptions about environments.
The review_leader will seek your input; respond thoroughly and clarify ambiguities. When finished, inform the review_leader and provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations.
PAPER CONTENT:
{paper_content}"""


def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You focus on clarity, interpretability, and reproducibility. Scrutinize for unclear explanations of methods, settings, concepts, or organization hindering understanding; lack of explainability or insights into decisions; and insufficient details for replication, such as code, data, hyperparameters, or protocols.
Identify issues like ambiguities, unstated assumptions, vague terms undermining comprehension, black-box behavior without explanations, missing feature importance or mechanistic understanding, poorly organized sections, missing code/data release, unreported seeds, ambiguous procedures, or lack of open science practices.
The review_leader will ask questions; respond and ask follow-up questions if needed. When done, inform the review_leader and provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations, including suggestions for improvement where relevant.
PAPER CONTENT:
{paper_content}"""


def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are part of a group of agents identifying limitations in a scientific paper. You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for issues in collection, labeling, cleaning, representativeness, or documentation; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Point out limitations such as small or non-diverse data, labeling errors, undocumented preprocessing, data leakage, reliance on flawed datasets without validation, biased outcomes leading to discrimination, lack of fairness metrics, unreported subgroup performance, ethical oversights, or failure to discuss misuse potential.
The review_leader will consult you; provide evidence-based critique and ask clarifying questions. When done, inform the review_leader and provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations.
PAPER CONTENT:
{paper_content}"""


# (column/agent name, readable section title used inside the master prompt)
SPECIALIST_COLUMNS: List[Tuple[str, str]] = [
    ("Novelty_Significance_Agent", "Novelty & Significance"),
    ("Citation_Agent", "Citation Analysis"),
    ("Theoretical_Methodological_Agent", "Theoretical & Methodological"),
    ("Experimental_Evaluation_Agent", "Experimental Evaluation"),
    ("Generalization_Robustness_Efficiency_Agent", "Generalization / Robustness / Efficiency"),
    ("Clarity_Interpretability_Reproducibility_Agent", "Clarity / Interpretability / Reproducibility"),
    ("Data_Ethics_Agent", "Data / Ethics"),
]

AGENT_PROMPT_FUNCS = {
    "Novelty_Significance_Agent": get_novelty_significance_prompt,
    "Citation_Agent": get_citation_agent_prompt,  # takes (paper, citations)
    "Theoretical_Methodological_Agent": get_theoretical_methodological_prompt,
    "Experimental_Evaluation_Agent": get_experimental_evaluation_prompt,
    "Generalization_Robustness_Efficiency_Agent": get_generalization_robustness_efficiency_prompt,
    "Clarity_Interpretability_Reproducibility_Agent": get_clarity_interpretability_reproducibility_prompt,
    "Data_Ethics_Agent": get_data_ethics_prompt,
}


# ---------------------------------------------------------------------------
# Master synthesis (full pipeline): ALWAYS lists all 7 sections (empty allowed)
# ---------------------------------------------------------------------------
def get_master_synthesis_prompt(paper_content: str, specialist_outputs: dict) -> str:
    sections = [
        ("Novelty & Significance",                   specialist_outputs.get("Novelty_Significance_Agent", "")),
        ("Citation Analysis",                         specialist_outputs.get("Citation_Agent", "")),
        ("Theoretical & Methodological",              specialist_outputs.get("Theoretical_Methodological_Agent", "")),
        ("Experimental Evaluation",                   specialist_outputs.get("Experimental_Evaluation_Agent", "")),
        ("Generalization / Robustness / Efficiency",  specialist_outputs.get("Generalization_Robustness_Efficiency_Agent", "")),
        ("Clarity / Interpretability / Reproducibility", specialist_outputs.get("Clarity_Interpretability_Reproducibility_Agent", "")),
        ("Data / Ethics",                             specialist_outputs.get("Data_Ethics_Agent", "")),
    ]

    joined = [f"=== {title} ===\n{content}".strip() for title, content in sections]
    all_reports = "\n\n".join(joined)

    return f"""You are the **Master Agent**. Your role is to receive limitation analyses from multiple specialist agents and produce a single, final, high-quality, consolidated list of limitations for the scientific paper.

    TASK:
    - Carefully read and integrate all provided specialist outputs below.
    - Remove redundancies (merge similar limitations).
    - Prioritize the most severe and well-justified limitations.
    - Preserve specificity and evidence from the original analyses.
    - Organize the final list logically by category.
    - Ensure each limitation is clearly stated, concise, and grounded in the paper.
    - Avoid introducing new limitations not raised by the specialists.

    OUTPUT FORMAT:
    Start with: "Here is the consolidated list of key limitations identified in the paper:"
    Then bullets:
    - **Category:** Specific limitation statement (brief explanation / evidence if useful)
    If specialists found only minor issues, say so and list them.

    PAPER CONTENT (context):
    {paper_content}

    SPECIALIST OUTPUTS (ONLY use these; do not invent new limitations):
    {all_reports}
    """


# ---------------------------------------------------------------------------
# Master prompt for leave-one-out / agent-subset ablations (skips empty sections)
# ---------------------------------------------------------------------------
def build_master_prompt_subset(paper_content: str, included_sections: List[tuple]) -> str:
    """
    included_sections: list of (title, content) tuples — already excludes the
    ablated agent. Falsy contents are skipped (so a row with a missing
    specialist output still produces a clean prompt).
    """
    joined = []
    for title, content in included_sections:
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        content = content.strip()
        if not content:
            continue
        joined.append(f"=== {title} ===\n{content}")
    all_reports = "\n\n".join(joined)

    return f"""You are the **Master Agent**. Your role is to receive limitation analyses from multiple specialist agents and produce a single, final, high-quality, consolidated list of limitations for the scientific paper.

    TASK:
    - Carefully read and integrate all provided specialist outputs below.
    - Remove redundancies (merge similar limitations).
    - Prioritize the most severe and well-justified limitations.
    - Preserve specificity and evidence from the original analyses.
    - Organize the final list logically by category.
    - Ensure each limitation is clearly stated, concise, and grounded in the paper.
    - Avoid introducing new limitations not raised by the specialists.

    OUTPUT FORMAT:
    Start with: "Here is the consolidated list of key limitations identified in the paper:"
    Then bullets:
    - **Category:** Specific limitation statement (brief explanation / evidence if useful)
    If specialists found only minor issues, say so and list them.

    PAPER CONTENT (context):
    {paper_content}

    SPECIALIST OUTPUTS (ONLY use these; do not invent new limitations):
    {all_reports}
    """


def build_v1_combined_input(main_text_tr: str, rel_sum_tr: str) -> str:
    """combined_for_novelty block (main paper + summarized relevant papers)."""
    return (
        "=== INPUT PAPER ===\n"
        f"{main_text_tr}\n\n"
        "=== RELEVANT PAPERS (SUMMARIZED) ===\n"
        f"{rel_sum_tr}"
    ).strip()
