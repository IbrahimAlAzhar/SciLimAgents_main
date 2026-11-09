"""Project-wide defaults: paths, model registry, column names, ablation registry."""
from __future__ import annotations

import os

PROJECT_ROOT = os.environ.get(
    "SCILIM_ROOT",
    "/lstr/sahara/datalab-ml/ibrahim/limagents_update/limagents_acl/scilimagents_main")
MODELS_DIR = os.environ.get("MODELS_DIR", "/lstr/sahara/datalab-ml/ibrahim/models")
OUTPUT_ROOT = os.environ.get("SCILIM_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs"))

DEFAULT_INPUT_CSV = os.environ.get(
    "INPUT_CSV",
    "/lstr/sahara/datalab-ml/ibrahim/limagents_update/rl_llm_agents/data/"
    "balanced_data/main/df_with_retrieved_sections_main.csv")

# ---------------------------------------------------------------------------
# Columns of the input CSV
# ---------------------------------------------------------------------------
COLS = dict(
    id="id",
    text="input_text_cleaned",                 # main paper text (all agents)
    novelty_text="input_text_for_novelty",     # LimAgents v1 novelty input
    text_without_lim="input_text_without_lim", # GRPO original CSV
    cited="cited_in",                          # LimAgents v1 citation agent
    relevant_list="relevant_papers_list",      # retrieved related papers (NovAgents / v1 summaries)
    retrieval_cols="retrieved_abs_int,ret_abs_int_cit",   # RAG / shared memory
    citation_rollout="ret_abs_int_cit",        # rollout retrieval_heavy context
    gt="ground_truth_lim_peer",                # gold limitations (one per line)
)

# ---------------------------------------------------------------------------
# Model registry. `family` drives chat-template quirks (Qwen3 thinking flag,
# system-role folding). jsd_teacher must share the student's vocabulary.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "qwen": dict(path=f"{MODELS_DIR}/qwen3_8b", family="qwen3", tag="qwen3_8b",
                 max_model_len=32768, jsd_teacher="/lstr/sahara/datalab-ml/ibrahim/qwen_3_32b_awq"),
    "mistral": dict(path=f"{MODELS_DIR}/mistral_small_3_1_24b_instruct", family="mistral",
                    tag="mistral_small_3_1_24b", max_model_len=32768, jsd_teacher=""),
    "gemma": dict(path=f"{MODELS_DIR}/gemma3_27b_it", family="gemma", tag="gemma3_27b_it",
                  max_model_len=32768, jsd_teacher=f"{MODELS_DIR}/gemma4_26b_a4b_it"),
    "llama": dict(path=f"{MODELS_DIR}/llama3_1_8b_instruct", family="llama", tag="llama3_1_8b",
                  max_model_len=32768, jsd_teacher=""),
    "gpt": dict(path="gpt-4o-mini", family="openai", tag="gpt_4o_mini", max_model_len=128000,
                jsd_teacher=""),
}

# Served LoRA names on the vLLM server: <system>-<reward>-<role>-<algo>
def lora_name(system: str, reward: str, role: str, algo: str) -> str:
    return f"{system}-{reward}-{role}-{algo}"


def adapter_dir(out_root: str, system: str, reward: str, role: str, algo: str) -> str:
    """outputs/<model>/adapters/<system>/<reward>/<role>_<algo>/final"""
    return os.path.join(out_root, "adapters", system, reward, f"{role}_{algo}", "final")


SYSTEMS = ("limagents", "novagents")
REWARDS = ("outcome", "stepwise", "jsd", "combined")
ALGOS = ("sft", "dpo", "dpo_aug", "grpo")
