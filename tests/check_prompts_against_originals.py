"""Byte-for-byte check: every original prompt function/constant vs scilim.prompts.
usage: python tests/check_prompts_against_originals.py <dir with original qwen_scilim/ and scilimagent_jsd/>"""
import ast, sys, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scilim.prompts import limagents_v1 as V1, limagents_seq as Q, canonical as C, novagents as N, merge as M, judge as J, evaluation as E

root = Path(sys.argv[1])
def funcs(path, names):
    src = (root / path).read_text(); tree = ast.parse(src); ns = {"List": list, "Optional": object, "Dict": dict}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            try: ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(path), "eval"), ns)
            except Exception: pass
    return ns
P, CIT = "PAPER_TEXT_XYZ", "CITATIONS_ABC"
fails = 0
def eq(name, a, b):
    global fails
    ok = a == b
    if not ok:
        fails += 1
        import difflib
        print("MISMATCH", name); print("\n".join(list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))[:20]))
    else:
        print("ok      ", name)

lim = "qwen_scilim/limagents/part_1/limagents.py"
o = funcs(lim, ["summarization_prompt_for_relevant_paper","get_novelty_significance_prompt","get_citation_agent_prompt",
  "get_theoretical_methodological_prompt","get_experimental_evaluation_prompt","get_generalization_robustness_efficiency_prompt",
  "get_clarity_interpretability_reproducibility_prompt","get_data_ethics_prompt","get_master_synthesis_prompt"])
eq("v1 summarizer", o["summarization_prompt_for_relevant_paper"]("ITEM", 2), V1.summarization_prompt_for_relevant_paper("ITEM", 2))
for n in ["get_novelty_significance_prompt","get_theoretical_methodological_prompt","get_experimental_evaluation_prompt",
          "get_generalization_robustness_efficiency_prompt","get_clarity_interpretability_reproducibility_prompt","get_data_ethics_prompt"]:
    eq("v1 "+n, o[n](P), getattr(V1, n)(P))
eq("v1 citation", o["get_citation_agent_prompt"](P, CIT), V1.get_citation_agent_prompt(P, CIT))
so = {k: f"OUT_{k}" for k in V1.AGENT_PROMPT_FUNCS}
eq("v1 master", o["get_master_synthesis_prompt"](P, so), V1.get_master_synthesis_prompt(P, so))
loo = funcs("qwen_scilim/limagents/ablation/leave_one_out_limagents.py", ["build_master_prompt"])
inc = [("A", "x"), ("B", ""), ("C", "y")]
eq("v1 LOO master", loo["build_master_prompt"](P, inc), V1.build_master_prompt_subset(P, inc))

seq = "qwen_scilim/dpo_limagents/inference/inference.py"
o = funcs(seq, ["get_novelty_significance_prompt","get_citation_agent_prompt","get_theoretical_methodological_prompt",
  "get_experimental_evaluation_prompt","get_generalization_robustness_efficiency_prompt","get_clarity_interpretability_reproducibility_prompt",
  "get_data_ethics_prompt","get_leader_agent_prompt","get_master_agent_prompt"])
for n in ["get_novelty_significance_prompt","get_theoretical_methodological_prompt","get_experimental_evaluation_prompt",
          "get_generalization_robustness_efficiency_prompt","get_clarity_interpretability_reproducibility_prompt","get_data_ethics_prompt"]:
    eq("seq "+n, o[n](P), getattr(Q, n)(P))
eq("seq citation", o["get_citation_agent_prompt"](P, CIT), Q.get_citation_agent_prompt(P, CIT))
eq("seq leader", o["get_leader_agent_prompt"](P), Q.get_leader_agent_prompt(P))
eq("seq master", o["get_master_agent_prompt"](P), Q.get_master_agent_prompt(P))
hop = funcs("qwen_scilim/dpo_limagents/inference/inference_with_hop_no_need.py",
            ["get_leader_agent_prompt","get_master_agent_prompt","build_mode_block","build_hop_system_prompt","MODES","HOP_USER_INSTRUCTIONS"])
eq("seq leader (hops)", hop["get_leader_agent_prompt"](P), Q.get_leader_agent_prompt(P, hops=True))
eq("seq master (hops)", hop["get_master_agent_prompt"](P), Q.get_master_agent_prompt(P, hops=True))
eq("seq hop system", hop["build_hop_system_prompt"]("BASE", "query", ["critique_heavy"]), Q.build_hop_system_prompt("BASE", "query", ["critique_heavy"]))
for k in ("search","query","draft"): eq("seq hop user "+k, hop["HOP_USER_INSTRUCTIONS"][k], Q.HOP_USER_INSTRUCTIONS[k])

ro = "scilimagent_jsd/rollout_qwen_vllm.py"
o = funcs(ro, ["get_novelty_significance_prompt","get_theoretical_methodological_prompt","get_experimental_evaluation_prompt",
  "get_generalization_robustness_efficiency_prompt","get_clarity_interpretability_reproducibility_prompt","get_data_ethics_prompt",
  "get_leader_system_prompt","get_master_system_prompt","MODE_PROMPT_FRAGMENTS","MODE_ACTION_CONFIG","HOP_USER_INSTRUCTIONS","build_agent_system"])
for n in ["get_novelty_significance_prompt","get_theoretical_methodological_prompt","get_experimental_evaluation_prompt",
          "get_generalization_robustness_efficiency_prompt","get_clarity_interpretability_reproducibility_prompt","get_data_ethics_prompt"]:
    eq("canonical "+n, o[n](P), getattr(C, n)(P))
eq("canonical leader", o["get_leader_system_prompt"](), C.get_leader_system_prompt())
eq("canonical master", o["get_master_system_prompt"](), C.get_master_system_prompt())
for k in o["MODE_PROMPT_FRAGMENTS"]: eq("mode "+k, o["MODE_PROMPT_FRAGMENTS"][k], C.MODE_PROMPT_FRAGMENTS[k])
eq("mode actions", repr(o["MODE_ACTION_CONFIG"]), repr(C.MODE_ACTION_CONFIG))
for k in ("search","query","draft"): eq("rollout hop "+k, o["HOP_USER_INSTRUCTIONS"][k], C.HOP_USER_INSTRUCTIONS[k])
for mode in C.MODE_ACTION_CONFIG:
    eq("agent system "+mode, o["build_agent_system"]("B", mode, "search", C.MODE_ACTION_CONFIG[mode], "CTX"),
       C.build_agent_system("B", mode, "search", C.MODE_ACTION_CONFIG[mode], "CTX"))
sm = funcs("scilimagent_jsd/inference/inference_shared_memory.py", ["SHARED_MEMORY_NOTES","MASTER_TOT_SUFFIX","WORKER_USER","get_leader_system_prompt"])
for r in ("worker","leader","master"): eq("shared mem note "+r, sm["SHARED_MEMORY_NOTES"][r], C.SHARED_MEMORY_NOTES[r])
eq("ToT suffix", sm["MASTER_TOT_SUFFIX"], C.MASTER_TOT_SUFFIX)
eq("worker user", sm["WORKER_USER"], C.WORKER_USER)
eq("leader ascii", sm["get_leader_system_prompt"](), C.get_leader_system_prompt_ascii())
gr = funcs("qwen_scilim/grpo_limagents/Qwen/train/worker_grpo.py", ["get_novelty_significance_prompt","build_worker_messages"])
eq("grpo worker msgs", repr(gr["build_worker_messages"]("Novelty_Significance_Agent", gr["get_novelty_significance_prompt"], "novelty and significance", P)),
   repr([{"role":"system","content":C.get_novelty_significance_prompt(P)},{"role":"user","content":C.grpo_leader_ask("Novelty_Significance_Agent","novelty and significance")}]))
mg = funcs("qwen_scilim/limagents+novagents/merge.py", ["get_merge_user_prompt"])
eq("merge user", mg["get_merge_user_prompt"]("A","B"), M.get_merge_user_prompt("A","B"))
sy = funcs("scilimagent_jsd/build_synthetic_dpo.py", ["_gen_user_prompt"])
eq("synthetic user", sy["_gen_user_prompt"]("EX","spec","CH",5,True), J.synth_weak_user("EX","spec","CH",5,True))
fp = funcs("qwen_scilim/limagents_dpo+novagents/eval_with_fp/eval_fp.py", ["_make_user_prompt"])
eq("fp user", fp["_make_user_prompt"]("LU","MS"), E.fp_user_prompt("LU","MS"))
print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
