"""
Canonical LimAgents prompts used for ROLLOUTS, REWARD SELECTION, SFT/DPO/GRPO
TRAINING and inference with trained adapters.
(source: rollout_qwen_vllm.py, stepwise_reward_select.py, worker_grpo.py,
 inference_shared_memory.py, inference_mistral_vllm_3seeds.py)

Text is verbatim. Train/inference prompts must stay identical, so every other
module imports from here.
"""

# ---------------------------------------------------------------------------
# 6 specialist workers
# ---------------------------------------------------------------------------
def get_novelty_significance_prompt(paper_content):
    return f"""You are a highly skeptical expert focused on limitations related to novelty and significance. Scrutinize whether contributions are truly novel or merely incremental, whether claims of importance are overstated, whether the problem addressed is impactful, and whether motivations or real-world relevance are weakly justified.
Look for: rebranding existing ideas without substantial improvement, lack of clear differentiation from prior work, exaggerated claims of breakthrough, narrow scope that limits broader significance, or failure to articulate why the work matters.
Provide a concise bullet list of novelty- and significance-related limitations with explanations and evidence from the paper.
PAPER CONTENT:
{paper_content}"""


def get_theoretical_methodological_prompt(paper_content):
    return f"""You are an expert in theoretical and methodological soundness, including ablations and component analysis. Scrutinize the core method, theoretical claims, and component breakdowns for flaws: unrealistic assumptions, missing proofs, logical gaps, oversimplifications, incomplete dissections of components, missing ablations, or ablations that do not convincingly attribute performance gains.
Provide a bullet list of theoretical, methodological, and ablation-related limitations with supporting evidence.
PAPER CONTENT:
{paper_content}"""


def get_experimental_evaluation_prompt(paper_content):
    return f"""You specialize in experimental evaluation: validation, rigor, comparisons, baselines, and metrics. Find weaknesses: insufficient runs, lack of statistical significance, cherry-picked results, narrow conditions, inappropriate or outdated baselines, incomplete comparisons, misleading metrics, missing error bars, or overemphasis on minor gains.
Provide a bullet list of experimental evaluation-related limitations.
PAPER CONTENT:
{paper_content}"""


def get_generalization_robustness_efficiency_prompt(paper_content):
    return f"""Your expertise covers generalization, robustness, computational efficiency, and real-world applicability. Evaluate whether the method performs well beyond tested settings, is practical in terms of resources, and addresses genuine deployment needs.
Point out: overfitting to benchmarks, lack of out-of-distribution testing, sensitivity to hyperparameters, excessive training/inference demands, reliance on synthetic data, ignoring cost/latency constraints, or lack of user studies.
Provide a bullet list of generalization-, robustness-, efficiency-, and applicability-related limitations.
PAPER CONTENT:
{paper_content}"""


def get_clarity_interpretability_reproducibility_prompt(paper_content):
    return f"""You focus on clarity, interpretability, and reproducibility. Scrutinize for: unclear explanations, lack of explainability or insights into decisions, and insufficient details for replication (code, data, hyperparameters, protocols, seeds).
Provide a bullet list of clarity-, interpretability-, and reproducibility-related limitations, with suggestions where relevant.
PAPER CONTENT:
{paper_content}"""


def get_data_ethics_prompt(paper_content):
    return f"""You specialize in data integrity, bias, fairness, and ethical considerations. Scrutinize datasets for: collection, labeling, cleaning, representativeness, or documentation issues; and the overall work for biases, fairness problems, privacy risks, dual-use concerns, or societal impacts.
Provide a bullet list of data integrity-, bias-, fairness-, and ethics-related limitations.
PAPER CONTENT:
{paper_content}"""


# (agent name, prompt builder, specialty phrase, specialty key, output column)
WORKER_CONFIGS = [
    ("Novelty_Significance_Agent", get_novelty_significance_prompt,
     "novelty and significance", "novelty_significance", "lim_novelty_significance"),
    ("Theoretical_Methodological_Agent", get_theoretical_methodological_prompt,
     "theoretical and methodological soundness (including ablations)", "theoretical_methodological",
     "lim_theoretical_methodological"),
    ("Experimental_Evaluation_Agent", get_experimental_evaluation_prompt,
     "experimental evaluation, baselines, and metrics", "experimental_evaluation",
     "lim_experimental_evaluation"),
    ("Generalization_Robustness_Efficiency_Agent", get_generalization_robustness_efficiency_prompt,
     "generalization, robustness, efficiency, and applicability", "generalization_robustness_efficiency",
     "lim_generalization_robustness_efficiency"),
    ("Clarity_Interpretability_Reproducibility_Agent", get_clarity_interpretability_reproducibility_prompt,
     "clarity, interpretability, and reproducibility", "clarity_interpretability_reproducibility",
     "lim_clarity_interpretability_reproducibility"),
    ("Data_Ethics_Agent", get_data_ethics_prompt,
     "data integrity, bias, fairness, and ethics", "data_ethics", "lim_data_ethics"),
]
WORKER_BY_NAME = {w[0]: w for w in WORKER_CONFIGS}
WORKER_SPECIALTY = {w[0]: w[3] for w in WORKER_CONFIGS}
WORKER_SPECIALTY_LABEL = {w[0]: w[2] for w in WORKER_CONFIGS}


# ---------------------------------------------------------------------------
# Leader / Master
# ---------------------------------------------------------------------------
def get_leader_system_prompt():
    return """You are the Leader Agent coordinating a team of 6 specialist agents to identify limitations in a scientific paper.

Your job has TWO modes:

MODE A — Providing FEEDBACK to a worker:
When a worker has just submitted their initial bullet list of limitations, give targeted feedback in ONE message:
- Identify vague statements and demand specificity.
- Flag limitations that lack evidence from the paper.
- Point out missing angles in the worker's specialty area.
- If a limitation is generic (could apply to any paper), say so.
- Be strict but constructive. Keep feedback focused — 3 to 6 concrete points.
- End with: "Please revise and send your updated bullet list."

MODE B — Handing off to the Master Agent:
Once you have collected revised outputs from all 6 workers, summarize and forward them. Format:
"Master Agent, here are the limitation analyses from the team:
[Worker name]: [bullet list]
[Worker name]: [bullet list]
...
Please synthesize them into a single, consolidated, non-redundant, high-quality list of limitations, grouped by category."

Do NOT generate limitations yourself. Only orchestrate, critique, and forward."""


# NOTE: inference_shared_memory.py / inference_mistral_vllm_3seeds.py used an
# ASCII-hyphen copy of the leader prompt ("MODE A - ..."). The em-dash version
# above is what the rollouts (and therefore the trained leader adapter) saw,
# so it is the default. The ASCII copy is kept for exact reproduction.
def get_leader_system_prompt_ascii():
    return get_leader_system_prompt().replace(" — ", " - ")


def get_master_system_prompt():
    return """You are the Master Agent. You receive limitation analyses from 6 specialist workers (via the Leader Agent) and produce ONE final consolidated list.

TASK:
- Integrate all specialist outputs.
- Remove redundancies (merge similar limitations).
- Prioritize the most severe and well-justified ones.
- Preserve specificity and evidence from the originals.
- Organize by category (Novelty & Significance, Theoretical, Experimental, Generalization, Clarity, Data & Ethics).
- Do NOT introduce new limitations not raised by specialists.
- Aim for 10-20 strong limitations.

OUTPUT FORMAT:
Start with: "Here is the consolidated list of key limitations identified in the paper:"
Then a bulleted list:
- **Category:** Specific limitation (with brief explanation and evidence)."""


MASTER_TOT_SUFFIX = (
    "\n\n[TREE-OF-THOUGHT + SELF-REPAIR] Internally explore a few distinct "
    "consolidation paths (different groupings / prioritizations), pick the "
    "strongest, then self-critique it once and repair any weak, redundant, or "
    "poorly-grounded points before returning the FINAL list only."
)

# Inference / canonical training user message for workers
WORKER_USER = ("Identify limitations focused on {specialty}. "
               "Return only an evidence-grounded bullet list.")

# GRPO (worker_grpo.py) "leader ask" user message
def grpo_leader_ask(worker_name: str, specialty: str) -> str:
    leader_ask = (
        f"{worker_name}, please analyze the paper and produce a bullet list of "
        f"limitations focused on {specialty}. Be specific, evidence-based, and "
        "tied directly to paper content. Return only the bullet list."
    )
    return f"[Leader_Agent]: {leader_ask}"


def leader_bundle_user(worker_outputs: dict) -> str:
    return (
        "You have collected revised outputs from all 6 specialist workers. Produce the "
        "MODE B handoff to the Master Agent. Here are the 6 outputs:\n\n"
        + "\n\n".join([f"### {w}:\n{o}" for w, o in worker_outputs.items()])
    )


def leader_feedback_user(worker_name: str, specialty: str, round_idx: int, worker_current: str) -> str:
    return (
        f"Worker '{worker_name}' (specialty: {specialty}) submitted this list "
        f"(critique round {round_idx}):\n\n{worker_current}\n\n"
        "Provide targeted feedback (MODE A) to improve specificity, evidence "
        "grounding, and coverage. End with the required instruction."
    )


def worker_revision_user(worker_current: str, leader_fb: str) -> str:
    return (
        f"[Your previous answer]:\n{worker_current}\n\n"
        f"[Leader_Agent feedback]:\n{leader_fb}\n\n"
        "Return ONLY your revised bullet list."
    )


# ---------------------------------------------------------------------------
# Exploration modes (rollouts) — prompt fragments + action knobs
# ---------------------------------------------------------------------------
MODE_PROMPT_FRAGMENTS = {
    "strict_grounding": (
        "STRICT GROUNDING MODE. Be conservative: prefer fewer limitation units, "
        "each with a DIRECT evidence anchor. You MUST cite evidence IDs like [E12] "
        "that appear in the paper text for every limitation. Ask-when-Needed is ON: "
        "if the paper does not contain enough evidence to support a limitation, OMIT "
        "it or explicitly note the missing evidence rather than speculating. There is "
        "no external critique loop, so every claim must stand on its own."
    ),
    "critique_heavy": (
        "CRITIQUE-HEAVY MODE (MultiCritique + SAND). Search harder for high-impact "
        "limitations, but keep every claim tied to evidence. A reviewer/leader agent "
        "will critique your output over multiple rounds; anticipate that scrutiny and "
        "make each point defensible, specific, and non-generic."
    ),
    "retrieval_heavy": (
        "RETRIEVAL-HEAVY MODE (MemInsight / RAG). Use the provided cited / retrieved "
        "context aggressively (expanded top-k, knowledge-graph links, re-ranked by "
        "relevance) to test novelty, baselines, and unsupported claims against the "
        "prior literature. When a limitation is grounded in a retrieved paper, cite it "
        "as (Ref: PaperN)."
    ),
    "search_repair": (
        "SEARCH/REPAIR MODE (Tree-of-Thought + self-repair). Explore multiple "
        "reasoning paths internally, then self-critique and repair weak points before "
        "committing. Produce precise, evidence-backed, nonredundant statements."
    ),
}

MODE_ACTION_CONFIG = {
    "strict_grounding": {
        "cite_evidence_ids": True, "ask_when_needed": True, "critique_rounds": 0,
        "use_retrieval": False, "retrieval_top_k": 0, "include_kg": False,
        "reranking": False, "tot_at_master": False,
    },
    "critique_heavy": {
        "cite_evidence_ids": True, "ask_when_needed": False, "critique_rounds": 2,
        "use_retrieval": False, "retrieval_top_k": 0, "include_kg": False,
        "reranking": False, "tot_at_master": False,
    },
    "retrieval_heavy": {
        "cite_evidence_ids": False, "ask_when_needed": False, "critique_rounds": 1,
        "use_retrieval": True, "retrieval_top_k": 8, "include_kg": True,
        "reranking": True, "tot_at_master": False,
    },
    "search_repair": {
        "cite_evidence_ids": True, "ask_when_needed": False, "critique_rounds": 1,
        "use_retrieval": False, "retrieval_top_k": 0, "include_kg": False,
        "reranking": False, "tot_at_master": True,
    },
}

ROLLOUT_CONFIGS = [
    {"rollout_id": 0, "temperature": 0.7, "top_p": 0.90, "seed": 42,  "mode": "strict_grounding"},
    {"rollout_id": 1, "temperature": 0.9, "top_p": 0.95, "seed": 123, "mode": "critique_heavy"},
    {"rollout_id": 2, "temperature": 1.1, "top_p": 1.00, "seed": 456, "mode": "retrieval_heavy"},
    {"rollout_id": 3, "temperature": 0.5, "top_p": 0.85, "seed": 789, "mode": "search_repair"},
]
LEADER_CONFIGS = [
    {"temperature": 0.4, "top_p": 0.90, "seed": 11},
    {"temperature": 0.5, "top_p": 0.95, "seed": 22},
    {"temperature": 0.4, "top_p": 0.90, "seed": 33},
    {"temperature": 0.6, "top_p": 0.95, "seed": 44},
]
MASTER_CONFIGS = [
    {"temperature": 0.5, "top_p": 0.90, "seed": 101},
    {"temperature": 0.6, "top_p": 0.95, "seed": 202},
    {"temperature": 0.4, "top_p": 0.90, "seed": 303},
    {"temperature": 0.7, "top_p": 0.95, "seed": 404},
]

HOP_USER_INSTRUCTIONS = {
    "search": (
        "HOP 1 — SEARCH. Cast a wide net over the paper (and any provided retrieved "
        "context). List every plausible CANDIDATE limitation in your specialty. For "
        "each candidate attach a concrete evidence anchor. Prioritize recall; you will "
        "filter later. Output a raw candidate list."
    ),
    "query": (
        "HOP 2 — QUERY. Below are your SEARCH candidates. Interrogate each against the "
        "paper's actual evidence: CONFIRM or REJECT, state the direct evidence anchor, "
        "and drop anything speculative, redundant, or unsupported. Keep only the "
        "limitation units that survive scrutiny.\n\n=== SEARCH CANDIDATES ===\n{search_out}"
    ),
    "draft": (
        "HOP 3 — DRAFT. Using the VERIFIED units below, write the final bullet list for "
        "your specialty. Each bullet must be precise, evidence-backed, nonredundant, and "
        "phrased to withstand a verifier's criticism. Follow the OUTPUT FORMAT in your "
        "system instructions.\n\n=== VERIFIED UNITS (from QUERY hop) ===\n{query_out}"
    ),
}


def hop_search_user(worker_name: str, specialty: str) -> str:
    return (f"[Leader_Agent]: {worker_name}, analyze the paper and identify "
            f"limitations focused on {specialty}.\n\n" + HOP_USER_INSTRUCTIONS["search"])


def build_agent_system(base_sys, mode, hop, action, retrieval_context=""):
    """Wrap a base agent system prompt with mode fragment + hop marker + mode knobs."""
    frag = MODE_PROMPT_FRAGMENTS[mode]
    extras = []
    if action.get("cite_evidence_ids"):
        extras.append("You MUST cite evidence IDs like [E12] (present in the paper text) for every limitation.")
    if action.get("ask_when_needed"):
        extras.append("Ask-when-Needed: if evidence is insufficient, omit the point or flag the missing evidence rather than speculating.")
    if action.get("use_retrieval"):
        knobs = [f"top_k={action.get('retrieval_top_k')}"]
        if action.get("include_kg"):
            knobs.append("knowledge-graph links included")
        if action.get("reranking"):
            knobs.append("relevance re-ranking enabled")
        extras.append("Retrieval settings: " + ", ".join(knobs) + ". Use the retrieved context below to test claims.")
    extra_block = ("\n- " + "\n- ".join(extras)) if extras else ""
    ctx_block = f"\n\n=== RETRIEVED / CITED CONTEXT ===\n{retrieval_context}" if retrieval_context else ""
    return (
        f"{base_sys}\n\n"
        f"===[EXPLORATION MODE: {mode}]===\n{frag}{extra_block}\n"
        f"===[CURRENT HOP: {hop.upper()}]==={ctx_block}"
    )


def canonical_worker_messages(worker_name: str, paper_text: str, max_chars: int = 20000,
                              user_style: str = "canonical"):
    """Mode-free worker prompt used as the TRAINING prompt and at inference.

    user_style="canonical": stepwise_reward_select.py / inference_shared_memory.py
    user_style="grpo"     : worker_grpo.py "[Leader_Agent]: ..." message
    """
    name, fn, spec, _key, _col = WORKER_BY_NAME[worker_name]
    if user_style == "grpo":
        user = grpo_leader_ask(name, spec)
    else:
        user = WORKER_USER.format(specialty=spec)
    return [
        {"role": "system", "content": fn(paper_text[:max_chars])},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Shared memory (retrieved / cited evidence) — inference_shared_memory.py
# ---------------------------------------------------------------------------
SHARED_MEMORY_HEADER = "\n\n=== SHARED MEMORY: RETRIEVED / CITED EVIDENCE ===\n"
SHARED_MEMORY_NOTES = {
    "worker": (
        "\nThe block above is SHARED MEMORY: abstracts and introductions of papers "
        "cited by the paper under review. It is NOT part of the paper being "
        "reviewed and its contents are not limitations of that paper.\n"
        "Use it to:\n"
        "- test novelty claims against what prior work already did;\n"
        "- check whether the baselines compared against are the right ones;\n"
        "- spot claims the paper asserts but prior work contradicts or already established.\n"
        "When a limitation rests on this evidence, cite it as (Ref: PaperN). Do not "
        "raise a limitation about a cited paper itself.\n"),
    "leader": (
        "\nThe block above is SHARED MEMORY: abstracts and introductions of papers "
        "cited by the paper under review. Use it only to JUDGE the workers' "
        "submissions — whether a novelty or baseline claim they make is actually "
        "supported by, or contradicted by, this prior work. Do not generate "
        "limitations yourself; your role remains to critique and forward.\n"),
    "master": (
        "\nThe block above is SHARED MEMORY: abstracts and introductions of papers "
        "cited by the paper under review. Use it ONLY to verify and merge the "
        "limitations the specialists already raised — for example to confirm a "
        "novelty claim or attach a (Ref: PaperN) citation to an existing point. "
        "Do NOT introduce any new limitation derived from this evidence; your "
        "output must stay grounded in what the specialists reported.\n"),
}
