"""Fake RAG corpus (rag_db dicts, like rag_data_source.csv) + target CSV with pdf_text / references."""
import random, sys
import pandas as pd
random.seed(1)
out_dir, n_targets = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12
topics = {"vision": "image segmentation pose depth convolution detection pixel camera",
          "nlp": "language model token transformer translation text corpus summarization",
          "rl": "reinforcement policy reward agent environment exploration value",
          "graph": "graph node edge message passing molecule network link"}
def para(t, n):
    w = topics[t].split() + "method results dataset evaluation baseline ablation proposed approach".split()
    return " ".join(random.choice(w) for _ in range(n)) + "."
rows, titles = [], []
for j in range(80):
    t = list(topics)[j % 4]
    title = f"A study of {t} number {j} with {random.choice(topics[t].split())}"
    titles.append((title, t))
    for sec in ["Abstract", "Introduction", "Method", "Experiments", "Conclusion"]:
        rows.append({"rag_db": repr({"chunk_key": f"P{j:03d}_{sec.lower()}", "text": para(t, 60), "section_name": sec,
                                     "paper_title": title, "year": 2010 + j % 12})})
pd.DataFrame(rows).to_csv(f"{out_dir}/rag_data_source.csv", index=False)
base = pd.read_csv(f"{out_dir}/data.csv")
pdf = []
for i in range(len(base)):
    t = list(topics)[i % 4]
    refs = [{"title": titles[j][0]} for j in random.sample([k for k in range(80) if titles[k][1] == t], 4)]
    refs += [{"title": "Some paper that is not in the corpus"}]
    pdf.append(repr({"title": f"Target paper {i}", "abstractText": para(t, 80),
                     "sections": [{"heading": "1 Introduction", "text": para(t, 150)},
                                  {"heading": "3 Methodology", "text": para(t, 150)},
                                  {"heading": "5 Experiments", "text": para(t, 150)}],
                     "references": refs}))
base["pdf_text"] = pdf
base["_source_file"] = [f"ICLR_{2018 + i % 6}" for i in range(len(base))]
base.to_csv(f"{out_dir}/data.csv", index=False)
print("fake corpus + targets written")
