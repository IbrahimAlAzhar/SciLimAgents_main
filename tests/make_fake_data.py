"""Builds a small synthetic CSV with every column the pipelines read (for dry runs)."""
import random, sys
import pandas as pd
random.seed(0)
words = ("transformer baseline dataset ablation robustness evaluation benchmark metric seed reproducibility "
         "fairness bias annotation generalization latency compute novelty contribution assumption proof").split()
def para(n): return " ".join(random.choice(words) for _ in range(n)) + "."
rows = []
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 24):
    paper = " ".join(para(25) for _ in range(40))
    cited = {f"p{j}": {"title": f"Cited paper {j}", "abstractText": para(40),
                       "sections": [{"heading": "Introduction", "text": para(60)}]} for j in range(3)}
    rows.append({"id": f"paper_{i}", "input_text_cleaned": "Abstract: " + paper,
                 "input_text_for_novelty": paper, "cited_in": repr(cited),
                 "relevant_papers_list": repr([para(80) for _ in range(3)]),
                 "retrieved_abs_int": repr(cited), "ret_abs_int_cit": repr(cited),
                 "ground_truth_lim_peer": "\n".join(f"The paper lacks {random.choice(words)} analysis of {para(8)}" for _ in range(4))})
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
print("wrote", sys.argv[1], len(rows))
