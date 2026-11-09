"""
python -m rag_ablation <stage> [options]

  build      retrieve with every configuration -> <work>/configs/<cfg>/input.csv
  summarize  6-field LLM summaries for the summary6 memory format (needs a served model)
  generate   run scilimagents_main on one configuration (extra args after `--` go to scilim generate)
  ir         retrieval quality vs each paper's cited papers (P/R/nDCG/Hit@k, MRR)
  irjudge    LLM-judge graded relevance on a sample + blind human sheet + agreement
  compare    downstream F1 + paired tests (Holm) + IR link + year + cost -> report/RAG_ABLATION_REPORT.txt
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(prog="rag_ablation", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["build", "summarize", "generate", "ir", "irjudge", "compare"])
    p.add_argument("--work_dir", required=True)
    g = p.add_argument_group("build")
    g.add_argument("--input_csv", default="")
    g.add_argument("--start_row", type=int, default=0)
    g.add_argument("--n_rows", type=int, default=0)
    g.add_argument("--rag_csv", default="")
    g.add_argument("--rag_db_col", default="rag_db")
    g.add_argument("--doc_id_field", default="auto")
    g.add_argument("--title_field", default="auto")
    g.add_argument("--year_field", default="auto")
    g.add_argument("--doc_id_regex", default="")
    g.add_argument("--corpus_limit", type=int, default=0, help="debug: read only the first N corpus rows")
    g.add_argument("--configs", default="baseline")
    g.add_argument("--base_alpha", type=float, default=0.5)
    g.add_argument("--sections", default="abstract introduction method")
    g.add_argument("--k", type=int, default=3)
    g.add_argument("--item_tokens", type=int, default=600)
    g.add_argument("--cand", type=int, default=100)
    g.add_argument("--section_filter", type=int, default=1)
    g.add_argument("--keep_citation_col", type=int, default=1)
    g.add_argument("--embed_models", default="minilm=/lstr/sahara/datalab-ml/ibrahim/models/all-MiniLM-L6-v2")
    g.add_argument("--pdf_col", default="pdf_text")
    g.add_argument("--text_col", default="input_text_cleaned")
    g.add_argument("--year_col", default="")
    g.add_argument("--overwrite", type=int, default=0)
    m = p.add_argument_group("models")
    m.add_argument("--llm_model", default="mock")
    m.add_argument("--llm_base_url", default=None)
    m.add_argument("--tokenizer", default=None)
    m.add_argument("--family", default="auto")
    m.add_argument("--max_model_len", type=int, default=32768)
    m.add_argument("--workers", type=int, default=16)
    m.add_argument("--config_dir", default="")
    m.add_argument("--ablations", default="novagents")
    m.add_argument("--keep_system_defaults", type=int, default=0)
    j = p.add_argument_group("ir / judge / compare")
    j.add_argument("--ks", default="1 3 5 9 15")
    j.add_argument("--rel_source", default="both", choices=["references", "cited_in", "both"])
    j.add_argument("--judge_model", default="gpt-4o-mini")
    j.add_argument("--judge_base_url", default=None)
    j.add_argument("--judge_configs", default="baseline")
    j.add_argument("--judge_n", type=int, default=100)
    j.add_argument("--human_labels", default="")
    j.add_argument("--link_metric", default="nDCG@9")
    j.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.stage == "build":
        from .build import run_build
        run_build(a)
    elif a.stage == "summarize":
        from .summarize import run_summarize
        run_summarize(a)
    elif a.stage == "generate":
        from .generate import run_generate
        run_generate(a, extra)
    elif a.stage == "ir":
        from .ir_eval import run_ir
        run_ir(a)
    elif a.stage == "irjudge":
        from .ir_judge import run_irjudge
        run_irjudge(a)
    elif a.stage == "compare":
        from .compare import run_compare
        run_compare(a)


if __name__ == "__main__":
    main()
