"""
LimAgents "sequential-aligned" prompts: 7 workers (6 specialists + Citation)
-> Leader consolidation -> Master synthesis.
(source: dpo_limagents/inference/inference.py, grpo_limagents/.../inference.py,
 dpo_limagents/inference/inference_with_hop_no_need.py)

Prompt text is verbatim. The HOP variants of the Leader/Master prompts and the
3-hop scaffolding from inference_with_hop_no_need.py are kept as well.
"""


def get_novelty_significance_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is exclusively in **novelty and significance**.

Your task: Analyze the paper below and identify all limitations related to novelty and significance. Scrutinize whether the contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.

Look for issues such as:
- Rebranding existing ideas without substantial improvement
- Lack of clear differentiation from prior work
- Exaggerated claims of breakthrough
- Narrow scope that limits broader significance
- Failure to articulate why the work matters beyond a niche setting
- Unaddressed alternatives or ignored related problems that diminish perceived impact

OUTPUT FORMAT:
Provide a concise bullet list of novelty- and significance-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""


def get_citation_agent_prompt(paper_content: str, citation_content: str) -> str:
    return f"""You are the **Citation Agent** in a multi-agent system for identifying limitations in scientific papers.

Your task: Compare the main article against the cited papers' information below. Identify limitations related to how the paper uses, interprets, or fails to engage with its citations.

Specifically check:
- Did the article fail to address key insights from its citations?
- Does the paper misinterpret or selectively cite prior work to make its own contribution look stronger?
- Are important related works missing from the citation landscape?

OUTPUT FORMAT:
Provide a bullet list of citation-related limitations. Each bullet should follow this format:
- [Limitation]: Explanation (Ref: Paper X)

=== MAIN ARTICLE ===
{paper_content}

=== CITED PAPERS INFO ===
{citation_content}"""


def get_theoretical_methodological_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **theoretical and methodological soundness**, including ablations and component analysis.

Your task: Analyze the paper below and identify all limitations related to theoretical foundations, methodology, and component analysis.

Look for issues such as:
- Unstated or overly strong assumptions
- Incomplete theoretical analysis or errors in derivations
- Methods that only work under restricted conditions not clearly acknowledged
- Missing ablations or lack of isolation of individual contributions
- Ablations that do not convincingly attribute performance gains
- Logical gaps between claims and supporting evidence

OUTPUT FORMAT:
Provide a concise bullet list of theoretical, methodological, and ablation-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""


def get_experimental_evaluation_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **experimental evaluation**, including validation, rigor, comparisons, baselines, and metrics.

Your task: Analyze the paper below and identify all limitations in the empirical evaluation.

Look for issues such as:
- Insufficient runs or lack of statistical significance testing
- Cherry-picked results or narrow experimental conditions
- Outdated, weak, or missing baselines and key competitors
- Unfair hyperparameter tuning or comparison setups
- Missing error bars, confidence intervals, or variance reporting
- Reliance on misleading metrics or missing standard metrics
- Overemphasis on minor gains without practical or statistical significance

OUTPUT FORMAT:
Provide a concise bullet list of experimental evaluation-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""


def get_generalization_robustness_efficiency_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise covers **generalization, robustness, computational efficiency, and real-world applicability**.

Your task: Analyze the paper below and identify all limitations related to how well the method generalizes, how robust it is, how efficient it is, and how applicable it is to real-world scenarios.

Look for issues such as:
- Overfitting to benchmarks or lack of out-of-distribution testing
- Sensitivity to hyperparameters or poor performance under distribution shifts
- Excessive training/inference demands or high resource requirements
- Reliance on synthetic data without real-world validation
- Ignoring deployment constraints like cost, latency, or hardware limitations
- Missing user studies or field tests

OUTPUT FORMAT:
Provide a concise bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""


def get_clarity_interpretability_reproducibility_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **clarity, interpretability, and reproducibility**.

Your task: Analyze the paper below and identify all limitations related to how clearly the work is presented, how interpretable the method/results are, and whether the work can be reproduced.

Look for issues such as:
- Unclear explanations of methods, settings, or key concepts
- Ambiguities or unstated assumptions that hinder comprehension
- Black-box behavior without explanations or mechanistic understanding
- Missing code, data, or hyperparameter details needed for replication
- Unreported random seeds or ambiguous experimental procedures

OUTPUT FORMAT:
Provide a concise bullet list of clarity-, interpretability-, and reproducibility-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""


def get_data_ethics_prompt(paper_content: str) -> str:
    return f"""You are a specialist agent in a multi-agent system for identifying limitations in scientific papers. Your expertise is in **data integrity, bias, fairness, and ethical considerations**.

Your task: Analyze the paper below and identify all limitations related to data quality, potential biases, fairness issues, and ethical concerns.

Look for issues such as:
- Small or non-diverse datasets, labeling errors, or undocumented preprocessing
- Data leakage or reliance on flawed/biased datasets without validation
- Biased outcomes that could lead to discrimination
- Lack of fairness metrics or unreported subgroup performance
- Privacy risks, dual-use concerns, or failure to discuss misuse potential

OUTPUT FORMAT:
Provide a concise bullet list of data integrity-, bias-, fairness-, and ethics-related limitations. Each bullet should include a clear limitation statement and a brief explanation with specific evidence from the paper.

PAPER CONTENT:
{paper_content}"""


def get_leader_agent_prompt(paper_content: str, hops: bool = False) -> str:
    if hops:
        context = ("CONTEXT: Multiple specialist worker agents have independently analyzed the paper below. "
                   "Each worker ran a 3-hop process (SEARCH -> QUERY -> DRAFT) and produced an evidence-backed "
                   "draft of limitations in its domain (novelty, methodology, experiments, generalization, "
                   "clarity, data/ethics, citations). Their drafts have been collected and will be provided to you.")
        step1 = "1. Read all specialist drafts carefully."
    else:
        context = ("CONTEXT: Multiple specialist worker agents have independently analyzed the paper below, each "
                   "focusing on a different aspect (novelty, methodology, experiments, generalization, clarity, "
                   "data/ethics, citations). Their outputs have been collected and will be provided to you.")
        step1 = "1. Read all specialist outputs carefully."
    return f"""You are the **Leader Agent** in a sequential multi-agent pipeline for identifying limitations in scientific papers.

{context}

YOUR TASK:
{step1}
2. Review each for quality, relevance, and completeness.
3. Flag any weak or unsupported claims that lack evidence from the paper.
4. Note redundancies across specialists (similar limitations raised by multiple agents).
5. Organize all limitations into a well-structured consolidated summary, grouped by category.
6. Preserve the specificity and evidence from the original analyses.

OUTPUT FORMAT:
Produce a structured summary organized by category (e.g., Novelty & Significance, Methodology, Experimental Evaluation, etc.). For each category:
- List the relevant limitations with their supporting evidence
- Mark any limitations that appear across multiple specialists as [CROSS-VALIDATED]
- Mark any limitations that seem weakly supported as [NEEDS EVIDENCE]

This consolidated summary will be passed to the Master Agent for final synthesis.

PAPER CONTENT (for reference):
{paper_content}"""


def get_master_agent_prompt(paper_content: str, hops: bool = False) -> str:
    if hops:
        context = ("CONTEXT: The Leader Agent has reviewed and consolidated limitation analyses from multiple "
                   "specialist workers (each of which ran a SEARCH -> QUERY -> DRAFT process). The Leader Agent's "
                   "consolidated summary will be provided to you.")
    else:
        context = ("CONTEXT: The Leader Agent has reviewed and consolidated limitation analyses from multiple "
                   "specialist workers. The Leader Agent's consolidated summary will be provided to you.")
    return f"""You are the **Master Agent** in a sequential multi-agent pipeline for identifying limitations in scientific papers.

{context}

YOUR TASK:
- Carefully read and integrate the Leader Agent's consolidated summary.
- Remove redundancies by merging similar limitations into single, well-stated points.
- Prioritize the most severe and well-justified limitations.
- Preserve specificity and evidence from the original analyses.
- Organize the final list logically, grouped by category.
- Ensure each limitation is clearly stated, concise, and grounded in the paper.
- Do NOT introduce new limitations not raised by the specialists.
- Aim for 10-20 strong limitations (adjust based on paper quality).

OUTPUT FORMAT:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then provide a bulleted list:
- **Category:** Specific limitation statement (with brief explanation and evidence reference if it adds value).

If no major limitations were found, state: "The paper appears methodologically sound with only minor limitations: [list them]."

PAPER CONTENT (for reference):
{paper_content}"""


# (output column, prompt builder, needs_citation_context)
SPECIALIST_CONFIG = [
    ("lim_novelty_significance", get_novelty_significance_prompt, False),
    ("lim_theoretical_methodological", get_theoretical_methodological_prompt, False),
    ("lim_experimental_evaluation", get_experimental_evaluation_prompt, False),
    ("lim_generalization_robustness_efficiency", get_generalization_robustness_efficiency_prompt, False),
    ("lim_clarity_interpretability_reproducibility", get_clarity_interpretability_reproducibility_prompt, False),
    ("lim_data_ethics", get_data_ethics_prompt, False),
    ("lim_citation", get_citation_agent_prompt, True),
]

WORKER_USER_MSG = (
    "Analyze the paper provided in your instructions. "
    "Identify all limitations in your domain of expertise. "
    "Provide a concise bullet list of limitations with explanations and evidence."
)
CITATION_USER_MSG = "Analyze citation-related limitations as described in your instructions."


def leader_input(combined_specialist: str, hops: bool = False) -> str:
    if hops:
        head = (f"Here are the DRAFT limitation analyses from the specialist worker "
                f"agents (each produced via a SEARCH -> QUERY -> DRAFT process):\n\n")
    else:
        head = f"Here are the limitation analyses from the specialist worker agents:\n\n"
    return (
        head
        + f"{combined_specialist}\n\n"
        f"Please review, organize, flag any weak claims, and produce a consolidated "
        f"summary of all limitations grouped by category for the Master Agent."
    )


def master_input(leader_output_truncated: str) -> str:
    return (
        f"The Leader Agent has reviewed and consolidated the specialist analyses. "
        f"Here is the Leader Agent's consolidated summary:\n\n"
        f"{leader_output_truncated}\n\n"
        f"Please synthesize this into a single, final, high-quality, non-redundant "
        f"list of limitations, grouped by category."
    )


# ---------------------------------------------------------------------------
# 3-hop scaffolding (inference_with_hop_no_need.py)
# ---------------------------------------------------------------------------
MODES = {
    "strict_grounding": (
        "Be conservative. Prefer fewer limitation units with direct evidence "
        "over broader critique."
    ),
    "critique_heavy": (
        "Search harder for high-impact limitations, but keep every claim tied "
        "to evidence."
    ),
    "retrieval_heavy": (
        "Use cited/RAG/OpenAlex context aggressively to test novelty, baselines, "
        "and unsupported claims."
    ),
    "repair_heavy": (
        "Expect verifier criticism. Make statements precise, evidence-backed, "
        "nonredundant, and easy to repair."
    ),
}

HOP_MODE_MAP = [
    ("search", ["retrieval_heavy"]),
    ("query",  ["critique_heavy"]),
    ("draft",  ["strict_grounding", "repair_heavy"]),
]

HOP_MAX_TOKENS = {"search": 1024, "query": 1280, "draft": 2048}


def build_mode_block(mode_keys) -> str:
    lines = []
    for k in mode_keys:
        if k in MODES:
            lines.append(f"- {k}: {MODES[k]}")
    return "\n".join(lines)


def build_hop_system_prompt(base_system_prompt: str, hop_name: str, mode_keys) -> str:
    mode_block = build_mode_block(mode_keys)
    return f"""{base_system_prompt}

============================================================
OPERATING MODE(S) FOR THIS STEP:
{mode_block}

CURRENT HOP: {hop_name.upper()}
============================================================
Follow the hop-specific instruction in the user message. Stay strictly within
your domain of expertise and keep every claim anchored to concrete evidence
(a quote, section, table, figure, or number) from the paper/context."""


HOP_USER_INSTRUCTIONS = {
    "search": (
        "HOP 1 — SEARCH.\n"
        "Cast a wide net over the paper (and any provided context). List every "
        "plausible CANDIDATE limitation in your domain of expertise. For each "
        "candidate, attach a concrete evidence anchor (a short quote, section "
        "name, table/figure, or number). Prioritize recall here — you will "
        "filter later. Output a raw candidate list; do not finalize."
    ),
    "query": (
        "HOP 2 — QUERY.\n"
        "Below are the candidate limitations from your SEARCH hop. Interrogate "
        "each one against the paper's actual evidence. For each candidate: "
        "(a) CONFIRM or REJECT it, (b) state the direct evidence anchor, "
        "(c) drop anything speculative, redundant, or unsupported. Keep ONLY "
        "limitation units that survive scrutiny.\n\n"
        "=== SEARCH CANDIDATES ===\n{search_out}"
    ),
    "draft": (
        "HOP 3 — DRAFT.\n"
        "Using the VERIFIED limitation units below, write the final output for "
        "your domain. Each bullet must be precise, evidence-backed, "
        "nonredundant, and phrased to withstand a verifier's criticism. Follow "
        "the OUTPUT FORMAT specified in your system instructions.\n\n"
        "=== VERIFIED UNITS (from QUERY hop) ===\n{query_out}"
    ),
}
