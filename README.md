# scilimagents_main — unified LimAgents / NovAgents / RL pipeline

One code base for every experiment that used to live in `qwen_scilim/` and
`scilimagent_jsd/`: multi-agent limitation generation (LimAgents, NovAgents,
merge), rollouts, three reward models (outcome / stepwise / JSD) plus their
combination, SFT + DPO + GRPO training, vLLM inference with the trained LoRA
adapters, and evaluation. Every stage is driven from a PBS file whose variables
act as the argparse interface.

```
scilimagents_main/
├── scilim/
│   ├── prompts/          every original prompt, verbatim (+ new zero-shot / RAG baselines)
│   ├── agents/           LimAgents (v1, seq, canonical), NovAgents, single-call baselines, merge
│   ├── llm/client.py     one chat client: vLLM server (open models + LoRA names) | OpenAI GPT | mock
│   ├── generate.py       ablation runner -> new column per ablation, CSV saved every 10 rows
│   ├── rollout/          rollouts for LimAgents (3-hop x 4 modes) and NovAgents; teacher = self|GPT|open model
│   ├── rewards/          outcome | stepwise | jsd | combined | synthetic (augmented DPO)
│   ├── train/            sft.py, dpo.py, grpo.py  (LoRA / QLoRA, multimodal-safe)
│   ├── serve/            stack SFT+DPO LoRAs for vLLM, --lora-modules builder, vocab checker
│   ├── evaluation/       P/R/F1 + cosine/Jaccard/ROUGE-L/BERTScore (+ false-positive analysis)
│   └── collect.py        all ablation columns -> one CSV
├── pbs/
│   ├── run_pipeline.sh   shared driver (vLLM start/stop, stage order)
│   └── models/{qwen,mistral,gemma,llama,gpt}.pbs
└── tests/                prompt-fidelity check, fake data, dry-run script
```

## Quick start (cluster)

```bash
cd /lstr/sahara/datalab-ml/ibrahim/limagents_update/limagents_acl
unzip scilimagents_main.zip            # creates scilimagents_main/
cd scilimagents_main
echo "sk-..." > ~/.openai_key && chmod 600 ~/.openai_key     # GPT judge / merger / teacher
mkdir -p outputs/qwen3_8b/logs
qsub pbs/models/qwen.pbs                                     # everything, default settings
```

Typical narrower jobs (any variable can be overridden with `qsub -v`):

```bash
# base ablations only, 50 rows
qsub -v STAGES="generate eval collect",ABLATIONS="zero_shot rag limagents novagents merge",N_ROWS=50 pbs/models/qwen.pbs
# every single LimAgents agent on its own + leave-one-out masters
qsub -v STAGES="generate eval",ABLATIONS="limagents_each limagents_loo" pbs/models/gemma.pbs
# RL with the JSD reward only, DPO + GRPO, GPT-4o-mini as rollout teacher
qsub -v STAGES="rollout reward train prepare infer eval",REWARDS="jsd",ALGOS="dpo grpo",TEACHER=gpt-4o-mini pbs/models/qwen.pbs
# only the trained ablations, for the stepwise reward
qsub -v STAGES="prepare infer eval",REWARDS="stepwise",TRAINED_ABLATIONS="limagents_dpo merge:limagents_dpo+novagents" pbs/models/mistral.pbs
```

Outputs land in `outputs/<model_tag>/`:

| path | contents |
|---|---|
| `generations/<ablation>.csv` | input rows + **new column `<ablation>`** (final limitations) + `<ablation>__<agent>` intermediates; saved every `SAVE_EVERY`=10 rows, resumable |
| `rollouts/<system>/` | `rollout_data_full.json` (+ worker/leader/master splits, per-paper checkpoints) |
| `select/<system>/<reward>/` | scored candidates, `sft_dataset_{worker,leader,master}.json`, `dpo_dataset_worker.json`, `dpo_dataset_worker_augmented.json` |
| `adapters/<system>/<reward>/<role>_<algo>/final` | LoRA adapters (worker_sft, leader_sft, master_sft, worker_dpo, worker_dpo_aug, worker_grpo) |
| `serve_adapters/` | vLLM-ready adapters (SFT+DPO stacked) named `<system>-<reward>-<role>-<algo>` |
| `eval/summary.csv` | one row per evaluated column: P/R/F1 (all + valid rows, + `_fp`), cosine, Jaccard, ROUGE-L, BERTScore |
| `all_ablations.csv` | every ablation column side by side |

## Ablations

Set `ABLATIONS` (no training) and `TRAINED_ABLATIONS` (after training) in the PBS file.

| name | what runs |
|---|---|
| `zero_shot` | one call, paper only (new baseline) |
| `rag` | one call, paper + retrieved/cited evidence (`retrieved_abs_int`, `ret_abs_int_cit`) |
| `rag_only` | one call, abstract + retrieved evidence only |
| `limagents` | **original LimAgents v1**: summarize 3 relevant papers → 7 specialists (incl. Citation) → Master |
| `limagents_agent:<Agent>` | one v1 specialist alone (its list is the output); `limagents_each` = all 7 |
| `limagents_loo` | v1 + 7 leave-one-agent-out Master syntheses (`limagents_loo__without_<Agent>` columns) |
| `limagents_seq`, `limagents_seq_hops` | 7 workers → Leader consolidation → Master ("sequential" prompts), optional 3-hop workers |
| `limagents_canonical` | 6 canonical workers → Leader MODE B → Master + Tree-of-Thought (the prompts the adapters are trained on) |
| `novagents`, `novagents_agent:<Agent>`, `novagents_each` | NovAgents: Paper-B summaries → 6 novelty specialists → Master |
| `merge` | LimAgents (v1) + NovAgents merged (Report A anchor, 10–14 limitations) |
| `limagents_{sft,dpo,dpo_aug,grpo}` | canonical LimAgents with trained worker (+ SFT leader/master) adapters |
| `novagents_{sft,dpo,dpo_aug,grpo}` | NovAgents with trained specialist (+ SFT master) adapters |
| `merge:<A>+<B>` | merge any LimAgents variant with any NovAgents variant, e.g. `merge:limagents_grpo+novagents_dpo` |
| modifiers `@mem` `@hops` `@critique` `@workeronly` | shared-memory RAG for all roles; 3-hop workers; 2 Leader critique rounds; train only the worker (base Leader/Master) |

Trained ablations are generated once per reward in `REWARDS`, so the column is
`<ablation>__<reward>` (e.g. `limagents_dpo__jsd`). Merge components that were
already generated are reused instead of being regenerated.

## Stages, teacher, reward, algorithm

| PBS variable | values | meaning |
|---|---|---|
| `STAGES` | `generate rollout reward train prepare infer eval collect` | run any subset, in this order |
| `START_ROW`, `N_ROWS` | default `0`, `200` | held-out generation/eval rows |
| `TRAIN_START`, `TRAIN_END` | default `300`, `450` | rollout / RL rows (never overlap eval) |
| `SYSTEMS` | `limagents novagents` | which agent systems get RL training |
| `TEACHER` | `self` · `gpt-4o-mini` · `/path/model` | who generates rollouts (on-policy vs distillation) |
| `JUDGE` | `gpt-4o-mini` · `self` · `/path/model` | LLM judge for outcome/stepwise rewards and synthetic rejections |
| `REWARDS` | `outcome stepwise jsd combined` | each gives its own SFT/DPO data and adapters |
| `JSD_TEACHER` | same-family bigger model · `self` | teacher for the JSD reward (vocab checked first; skipped if incompatible) |
| `ALGOS` | `sft dpo dpo_aug grpo` | `dpo_aug` = DPO whose bottom 30 % pairs get synthetic ungrounded rejections |
| `MERGER`, `MERGE_STYLE` | `gpt-4o-mini`/`self`; `anchor`/`anchor_v0`/`simple` | merge model and prompt |
| `EVAL_JUDGE`, `EVAL_FP`, `FP_MODEL` | | eval pair judge; false-positive classification of unmatched limitations |
| `FOUR_BIT`, `VLLM_TP`, `MAX_MODEL_LEN` | | QLoRA; tensor parallel; context |
| `SAVE_EVERY` | `10` | rows between CSV saves |

**Reward models**

* **outcome** — LLM-as-judge hybrid (from `reward_model_no_need.py` / `worker_grpo.py`): bipartite matching to the gold limitations (F-β, β=1.5, novelty bonus), per-item grounded/specific/valid/severity, ROUGE-L cross-check, length/redundancy/boilerplate guards; master adds organisation/dedup; leader handoff + feedback scoring.
* **stepwise** — process reward (`stepwise_reward_select.py`): SEARCH 0.22 / RETRIEVE 0.30 / SYNTHESIZE 0.48 grounding scores blended 0.7/0.3 with GT alignment (0.7 judge + 0.15 ROUGE-L + 0.15 lexical).
* **jsd** — 1 − JSD(teacher ‖ student) under teacher forcing (`jsd_reward_select.py`).
* **combined** — min-max-normalized 0.25 JSD + 0.50 judge + 0.25 grounding (+ optional outcome weight) (`two_stage_reward.py`).

Selection is identical for all rewards: SFT = best candidate per (paper, agent), DPO = best vs worst worker when the gap clears the threshold, with the mode-free canonical prompt so exploration-mode text never leaks into training.

**GRPO** accepts any of the four rewards (`--reward` in `scilim grpo`; the PBS uses the same `REWARDS` loop). Online outcome/stepwise rewards need an API judge while the GPU trains; with a local judge, stepwise falls back to ROUGE + lexical alignment and outcome is skipped.

## Models (`pbs/models/*.pbs`)

| file | model | GPUs | notes |
|---|---|---|---|
| `qwen.pbs` | `models/qwen3_8b` | 1 | thinking disabled; JSD teacher `qwen_3_32b_awq` |
| `mistral.pbs` | `models/mistral_small_3_1_24b_instruct` | 2 (TP=2) | QLoRA; no same-vocab teacher in `models/` → JSD off unless `JSD_TEACHER` set |
| `gemma.pbs` | `models/gemma3_27b_it` | 2 (TP=2) | multimodal checkpoint, LoRA restricted to `language_model.*`; JSD teacher `gemma4_26b_a4b_it` only if vocab matches |
| `llama.pbs` | `models/llama3_1_8b_instruct` | 1 | **no Llama in `models/` yet** — download one or set `MODEL_PATH` |
| `gpt.pbs` | `gpt-4o-mini` (API) | 0 | inference ablations only; use GPT as `TEACHER`/`JUDGE` in the others |

All jobs activate the `image_lim_vllm_final` conda env and start/stop a local vLLM
OpenAI-compatible server per stage (base model + every trained LoRA via `--lora-modules`).

## Direct CLI (outside PBS)

```bash
python -m scilim generate --ablations zero_shot limagents --llm_model student --llm_base_url http://127.0.0.1:8000/v1 \
       --tokenizer /path/model --output_root outputs/x --n_rows 20
python -m scilim rollout  --system limagents --teacher_model gpt-4o-mini --out_dir outputs/x/rollouts/limagents
python -m scilim reward   --reward stepwise --rollout_json .../rollout_data_full.json --out_dir outputs/x/select/limagents/stepwise
python -m scilim sft      --role worker --base_model M --sft_data .../sft_dataset_worker.json --out_dir .../worker_sft
python -m scilim dpo      --base_model M --sft_adapter .../worker_sft/final --dpo_data .../dpo_dataset_worker.json --out_dir .../worker_dpo
python -m scilim grpo     --base_model M --sft_adapter .../worker_sft/final --rollout_json R --reward jsd --jsd_teacher T --out_dir .../worker_grpo
python -m scilim prepare  --output_root outputs/x --base_model M
python -m scilim eval     --gen_dir outputs/x/generations --out_dir outputs/x/eval
```

Every sub-command has `-h`. `--llm_model mock` runs the whole thing offline with a deterministic fake model.

## Prompt inventory

All prompts are copied verbatim; `tests/check_prompts_against_originals.py` renders
every original prompt function and checks byte-for-byte equality (0 differences).

| module | source |
|---|---|
| `prompts/limagents_v1.py` | `limagents/part_*/limagents.py` (7 specialists, summarizer, master, per-agent system prompts), `ablation/leave_one_out_limagents.py` |
| `prompts/limagents_seq.py` | `dpo_limagents/inference/inference.py`, `grpo_limagents/.../inference.py`, `inference_with_hop_no_need.py` (hops, modes) |
| `prompts/canonical.py` | `rollout_qwen_vllm.py`, `stepwise_reward_select.py`, `worker_grpo.py`, `inference_shared_memory.py`, `inference_mistral_vllm_3seeds.py` |
| `prompts/novagents.py` | `novagents/novagents.py` |
| `prompts/merge.py` | `*/merge.py`, `merge_and_eval.py`, `limagents_dpo+novagents_dpo/lim_nov.py` |
| `prompts/judge.py` | `reward_model_no_need.py`, `worker_grpo.py`, `stepwise_reward_select.py`, `build_synthetic_dpo.py` |
| `prompts/evaluation.py` | all `eval*.py` (pair judge), `eval_fp.py` (false-positive classifier) |
| `prompts/baselines.py` | **new**: zero-shot, RAG, RAG-only |

## What changed vs. the original scripts

Behaviour-preserving consolidation plus these fixes/improvements:

* **Secrets removed.** Several scripts and PBS files hard-coded OpenAI API keys and a Hugging Face token. They are gone; keys are read from `OPENAI_API_KEY` or `~/.openai_key`. **Rotate the leaked keys** — they were in the files you shared.
* **Every output is resumable** and saved every 10 rows (atomic writes, SIGTERM-safe); reruns skip finished rows and retry `ERROR` rows. If any agent call fails, the row is marked `ERROR` instead of letting the Master silently paper over it.
* **One vLLM path for all models**, including trained adapters: DPO adapters are trained on the SFT-merged base, so `scilim prepare` stacks SFT ⊕ DPO into one exact LoRA (as in `prepare_vllm_adapters.py`) so vLLM serves base+SFT+DPO. Trained ablations check their adapters are actually served before running.
* **Merge prompt bug fixed**: `merge.py` fused items 1 and 2 ("…14 limitations total 2. Report A…"). `MERGE_STYLE=anchor_v0` reproduces the old text exactly.
* **Judge schema bug fixed**: `reward_model_no_need.py` sent literal `{{…}}` braces to the judge; the single-brace versions from `worker_grpo.py` are used.
* **NovAgents RL**: rollouts, rewards, SFT/DPO/GRPO now also work for NovAgents (needed for `limagents_dpo+novagents_dpo` style ablations).
* **GRPO**: any reward; reference policy = a frozen copy of the SFT adapter on the same base (no second full model); log-probs computed only on completion positions (the original materialized full-vocab log-softmax over the 6k-token prompt); KV cache on during sampling.
* **DPO**: QLoRA path (SFT merged once on CPU, trained in NF4) for 24B/27B models; no double EOS; seed control.
* **Multimodal checkpoints** (Gemma-3-27B, Mistral-Small-3.1): loaded via the image-text class when needed and LoRA restricted to `language_model.*`.
* **Prompt truncation keeps instructions**: context is fitted by shrinking the paper, never by cutting the tail (which deleted the instruction and generation header in the old HF inference).
* **Qwen3 thinking off** everywhere and `<think>` blocks stripped; templates without a system role are handled automatically.

## Testing

```bash
python tests/check_prompts_against_originals.py <dir containing qwen_scilim/ and scilimagent_jsd/>
bash tests/run_dry_tests.sh            # whole pipeline on fake data with the mock model (CPU, no GPU/API)
```

The dry run covers generation of every ablation, rollouts, outcome/stepwise/combined
rewards, synthetic DPO, eval (incl. false-positive stage) and collect. Training and
JSD need GPUs and were statically checked only — run a short job first, e.g.
`qsub -v N_ROWS=10,TRAIN_END=310,GRPO_MAX_STEPS=5 pbs/models/qwen.pbs`.
