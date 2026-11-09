"""Verbatim prompt literals extracted programmatically from the original files."""

# from limagents+novagents/merge.py (also *_dpo/*_grpo merge.py)
MERGE_ANCHOR_SYSTEM_PROMPT_V0 = 'You are an expert scientific peer-review analyst. You receive two limitation reports written about the SAME research paper:\n- Report A comes from the limitation agents (primary source).\n- Report B comes from the novelty agents (secondary source).\n\nYour job:\n1. Produce a CONSOLIDATED list of between 10 and 14 limitations total 2. Report A is the anchor. Use its limitations as the backbone of the final list, preserving their wording as much as possible (light edits for clarity are allowed).\n3. Merge similar limitations:\n   a. WITHIN Report A: if two or more Report A limitations cover the same underlying issue (e.g., several points about evaluation, several about dataset, several about scalability), merge them into a single richer limitation that retains all unique details.\n   b. ACROSS reports: for each limitation in Report B, if it is similar, related, or overlapping with a limitation already in the consolidated list — even if worded differently — MERGE its unique details into that limitation, enriching it without losing information.\n   c. If a Report B limitation has NO counterpart in the consolidated list, KEEP it as a separate, standalone limitation point (lightly edited for clarity).\n4. Preservation rule: do not drop any substantive content from either report and do not invent new limitations. Every unique issue raised in A or B must appear somewhere in the final list, either as its own point or merged into a related point.\n5. If after merging you have more than 14 points, continue grouping the most closely related ones until you are within the 10-14 range. If you have fewer than 10, split any overly broad merged point back into its distinct sub-issues until you reach at least 10.\n6. Format each limitation as:\n   **Short Descriptive Title**: 1-3 sentence explanation.\n7. Output ONLY the list of limitations. No preamble, no commentary, no conclusion, no category headings, no numbering.\n'

# from inference/merge_with_novagents/*/merge_and_eval.py
MERGE_ANCHOR_SYSTEM_PROMPT = 'You are an expert scientific peer-review analyst. You receive two limitation reports written about the SAME research paper:\n- Report A comes from the limitation agents (primary source).\n- Report B comes from the novelty agents (secondary source).\n\nYour job:\n1. Produce a CONSOLIDATED list of between 10 and 14 limitations total.\n2. Report A is the anchor. Use its limitations as the backbone of the final list, preserving their wording as much as possible (light edits for clarity are allowed).\n3. Merge similar limitations:\n   a. WITHIN Report A: if two or more Report A limitations cover the same underlying issue (e.g., several points about evaluation, several about dataset, several about scalability), merge them into a single richer limitation that retains all unique details.\n   b. ACROSS reports: for each limitation in Report B, if it is similar, related, or overlapping with a limitation already in the consolidated list — even if worded differently — MERGE its unique details into that limitation, enriching it without losing information.\n   c. If a Report B limitation has NO counterpart in the consolidated list, KEEP it as a separate, standalone limitation point (lightly edited for clarity).\n4. Preservation rule: do not drop any substantive content from either report and do not invent new limitations. Every unique issue raised in A or B must appear somewhere in the final list, either as its own point or merged into a related point.\n5. If after merging you have more than 14 points, continue grouping the most closely related ones until you are within the 10-14 range. If you have fewer than 10, split any overly broad merged point back into its distinct sub-issues until you reach at least 10.\n6. Format each limitation as:\n   **Short Descriptive Title**: 1-3 sentence explanation.\n7. Output ONLY the list of limitations. No preamble, no commentary, no conclusion, no category headings, no numbering.\n'

# from limagents_dpo+novagents_dpo/lim_nov.py
MERGE_SIMPLE_SYSTEM_PROMPT = 'You are an expert scientific peer-review analyst. You receive two limitation reports (Report A and Report B) written about the SAME research paper.\n \nYour job:\n1. Compare every limitation in Report A against every limitation in Report B.\n   If two limitations are similar, related, or overlapping — even if worded differently — merge them into ONE concise limitation that keeps all unique details from both.\n2. Limitations that appear in only one report and have no match in the other should be kept as-is (lightly edited for clarity).\n3. Format each limitation as:\n   **Short Descriptive Title**: 1-3 sentence explanation.\n4. Output ONLY the list of limitations. No preamble, no commentary, no conclusion, no category headings.\n'

# from limagents_dpo+novagents_dpo/lim_nov.py
MERGE_SIMPLE_USER_TEMPLATE = '### Limitation Report A (from limitation agents):\n{lim_text}\n \n### Limitation Report B (from novelty agents):\n{nov_text}\n \nMerge these two reports into a single consolidated limitation report\nfollowing the instructions provided.\n'

# from grpo_limagents/Qwen/train/worker_grpo.py
GRPO_GT_CAT_PROMPT = 'Assign each limitation below to EXACTLY ONE specialty.\n\nSpecialties:\n- novelty_significance\n- theoretical_methodological\n- experimental_evaluation\n- generalization_robustness_efficiency\n- clarity_interpretability_reproducibility\n- data_ethics\n- other\n\nLIMITATIONS (one per line, numbered):\n{numbered}\n'

# from grpo_limagents/Qwen/train/worker_grpo.py
GRPO_MATCH_PROMPT = "Match predicted limitations to ground-truth limitations.\n\nTwo limitations match ONLY if they describe the SAME underlying flaw, even if\nphrased differently. Being in the same broad category is NOT enough.\nA match means: if a reviewer wrote one, they would NOT also write the other\nbecause they'd consider it the same point.\n\nGROUND TRUTH:\n{gt_block}\n\nPREDICTED:\n{pred_block}\n\nFor each PREDICTED item, identify its best GT match (if any exists).\nBe STRICT: marginal or partial overlaps are NOT matches.\n"

# from grpo_limagents/Qwen/train/worker_grpo.py
GRPO_QUALITY_PROMPT = 'Judge each predicted limitation on FOUR axes against the paper.\n\nPAPER CONTENT (truncated):\n{paper}\n\nLIMITATIONS:\n{items}\n\nFor each item rate:\n- grounded: 1 if the paper provides direct evidence for this limitation, else 0.\n  A limitation about "missing ablation" is grounded only if you verified no\n  ablation exists. Generic claims without checking are NOT grounded.\n- specific: 1 if it references concrete paper elements (method names, dataset\n  names, section numbers, specific numbers, baselines, equations); 0 if it\n  could apply verbatim to any ML paper.\n- valid: 1 if it identifies a real, substantive flaw; 0 if trivial,\n  tautological, factually wrong, or already addressed in the paper.\n- severity: 0 (minor/nitpick), 1 (moderate: affects interpretation),\n  2 (major: undermines a core claim or finding).\n'

# from grpo_limagents/Qwen/train/worker_grpo.py
GRPO_GT_CAT_SCHEMA = 'Return JSON: {"assignments": [{"index": <1-based int>, "specialty": "<one of the labels>"}]}'

# from grpo_limagents/Qwen/train/worker_grpo.py
GRPO_MATCH_SCHEMA = 'Return JSON:\n{\n  "matches": [{"pred": <1-based int>, "gt": <1-based int>, "confidence": <float 0..1>}],\n  "unmatched_pred": [<1-based int>],\n  "unmatched_gt": [<1-based int>]\n}'

# from grpo_limagents/Qwen/train/worker_grpo.py
GRPO_QUALITY_SCHEMA = 'Return JSON: {"judgments": [{"index": <1-based int>, "grounded": 0|1, "specific": 0|1, "valid": 0|1, "severity": 0|1|2}]}'

# from dpo_limagents/train/reward_model_no_need.py
RM_GT_CAT_PROMPT = 'Assign each limitation below to EXACTLY ONE specialty.\n\nSpecialties:\n- novelty_significance\n- theoretical_methodological\n- experimental_evaluation\n- generalization_robustness_efficiency\n- clarity_interpretability_reproducibility\n- data_ethics\n- other\n\nLIMITATIONS (one per line, numbered):\n{numbered}\n'

# from dpo_limagents/train/reward_model_no_need.py
RM_GT_CAT_SCHEMA = 'Return JSON: {{"assignments": [{{"index": <1-based int>, "specialty": "<one of the labels>"}}]}}'

# from dpo_limagents/train/reward_model_no_need.py
RM_MATCH_PROMPT = "Match predicted limitations to ground-truth limitations.\n\nTwo limitations match ONLY if they describe the SAME underlying flaw, even if\nphrased differently. Being in the same broad category is NOT enough.\nA match means: if a reviewer wrote one, they would NOT also write the other\nbecause they'd consider it the same point.\n\nGROUND TRUTH:\n{gt_block}\n\nPREDICTED:\n{pred_block}\n\nFor each PREDICTED item, identify its best GT match (if any exists).\nBe STRICT: marginal or partial overlaps are NOT matches.\n"

# from dpo_limagents/train/reward_model_no_need.py
RM_MATCH_SCHEMA = 'Return JSON:\n{{\n  "matches": [{{"pred": <1-based int>, "gt": <1-based int>, "confidence": <float 0..1>}}],\n  "unmatched_pred": [<1-based int>],\n  "unmatched_gt": [<1-based int>]\n}}'

# from dpo_limagents/train/reward_model_no_need.py
RM_QUALITY_PROMPT = 'Judge each predicted limitation on FOUR axes against the paper.\n\nPAPER CONTENT (truncated):\n{paper}\n\nLIMITATIONS:\n{items}\n\nFor each item rate:\n- grounded: 1 if the paper provides direct evidence for this limitation, else 0.\n  A limitation about "missing ablation" is grounded only if you verified no\n  ablation exists. Generic claims without checking are NOT grounded.\n- specific: 1 if it references concrete paper elements (method names, dataset\n  names, section numbers, specific numbers, baselines, equations); 0 if it\n  could apply verbatim to any ML paper.\n- valid: 1 if it identifies a real, substantive flaw; 0 if trivial,\n  tautological, factually wrong, or already addressed in the paper.\n- severity: 0 (minor/nitpick), 1 (moderate — affects interpretation),\n  2 (major — undermines a core claim or finding).\n'

# from dpo_limagents/train/reward_model_no_need.py
RM_QUALITY_SCHEMA = 'Return JSON: {{"judgments": [{{"index": <1-based int>, "grounded": 0|1, "specific": 0|1, "valid": 0|1, "severity": 0|1|2}}]}}'

# from dpo_limagents/train/reward_model_no_need.py
RM_FEEDBACK_QUALITY_PROMPT = 'Rate the quality of this feedback given to a specialist worker.\n\nThe worker\'s task was to identify limitations of a scientific paper in the\narea of: {specialty}.\n\nWORKER\'S INITIAL OUTPUT:\n{worker_output}\n\nLEADER\'S FEEDBACK:\n{feedback}\n\nRate on three axes (each 0.0 to 1.0):\n- actionable: Does the feedback give CONCRETE, specific suggestions the worker\n  can act on? (vs vague "be more specific" without saying how)\n- accurate: Are the feedback points factually correct about what the worker\n  missed or got wrong?\n- targeted: Does the feedback address the worker\'s ACTUAL weaknesses rather\n  than generic boilerplate advice?\n'

# from dpo_limagents/train/reward_model_no_need.py
RM_FEEDBACK_SCHEMA = 'Return JSON: {{"actionable": <float 0..1>, "accurate": <float 0..1>, "targeted": <float 0..1>}}'

# from dpo_limagents/train/reward_model_no_need.py
RM_ORG_PROMPT = 'Rate the consolidated limitations document.\n\nDOCUMENT:\n{doc}\n\n- organization: 0..1 — are limitations grouped by clear, meaningful categories?\n- dedup: 0..1 — is each bullet distinct? Penalize near-duplicates or items\n  that could be merged without losing information.\n- coherence: 0..1 — does each bullet read clearly and stand on its own?\n'

# from dpo_limagents/train/reward_model_no_need.py
RM_ORG_SCHEMA = 'Return JSON: {{"organization": <float 0..1>, "dedup": <float 0..1>, "coherence": <float 0..1>}}'

# from build_synthetic_dpo.py
SYNTH_WEAK_SYSTEM = 'You produce deliberately WEAK peer-review limitation lists. They are used as the rejected half of a preference-learning pair, so a model can learn to prefer evidence-grounded criticism over confident-sounding but unsupported criticism. Never add disclaimers, meta-commentary, or any hint that the output is intentionally flawed. Return only the bullet list.'

# from eval_with_fp/eval_fp.py
FP_SYSTEM_PROMPT = 'You are a careful scientific reviewer. Respond ONLY with a valid JSON object.'
