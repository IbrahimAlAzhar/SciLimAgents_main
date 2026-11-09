"""
STAGE generate — run scilimagents_main's generator on one configuration's input.csv.

scilimagents_main reads only the first 3 relevant papers (NovAgents k_relevant, LimAgents v1
summaries) and 8 records for RAG / shared memory. To make DEPTH and FORMAT ablations reach the
agents, the generator is run with these knobs set from the configuration (no file in
scilimagents_main is modified; the configs are patched in-process):
    NovAgents  k_relevant = n_papers, per_paper_input_tokens = item_tokens,
               citation_tokens = max(default, n_papers * summary_max_new)
    LimAgents  v1_summary_items = n_papers, v1_summary_item_tokens = item_tokens,
               retrieval_top_k = n_papers, retrieval_max_tokens = n_papers * item_tokens   (@mem)
    RAG        top_k = n_papers, retrieval_max_tokens = n_papers * item_tokens
Set KEEP_SYSTEM_DEFAULTS=1 to run the systems exactly as in scilimagents_main instead.
Writes <cfg>/out/generations/<ablation>.csv and <cfg>/timing.json.
"""
from __future__ import annotations

import functools
import json
import time
from pathlib import Path


def patch(cfg: dict, keep_defaults=False):
    import scilim.generate as G
    if keep_defaults or not cfg.get("memory", True):
        return
    n = max(1, int(cfg["n_papers"]))
    it = int(cfg["item_tokens"])
    Nov, Lim, single = G.NovAgentsConfig, G.LimAgentsConfig, G.run_single

    def nov(*a, **k):
        c = Nov(*a, **k)
        c.k_relevant = n
        c.per_paper_input_tokens = it
        c.citation_tokens = max(c.citation_tokens, n * c.summary_max_new)
        return c

    def lim(*a, **k):
        c = Lim(*a, **k)
        c.v1_summary_items = n
        c.v1_summary_item_tokens = it
        c.retrieval_top_k = n
        c.retrieval_max_tokens = n * it
        return c
    G.NovAgentsConfig, G.LimAgentsConfig = nov, lim
    G.run_single = functools.partial(single, top_k=n, retrieval_max_tokens=n * it)
    print(f"[generate] systems read {n} retrieved papers, {it} tokens each")


def run_generate(a, extra_args):
    d = Path(a.config_dir)
    cfg = json.loads((d / "config.json").read_text())
    patch(cfg, a.keep_system_defaults)
    import scilim.generate as G
    argv = ["--ablations", *a.ablations.split(), "--input_csv", str(d / "input.csv"),
            "--output_root", str(d / "out"), "--llm_model", a.llm_model, *extra_args]
    if a.llm_base_url:
        argv += ["--llm_base_url", a.llm_base_url]
    if cfg.get("seed"):
        argv += ["--seed", str(cfg["seed"])]
    t0 = time.time()
    G.main(argv)
    dt = time.time() - t0
    tp = d / "timing.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    old[a.ablations] = old.get(a.ablations, 0) + dt
    tp.write_text(json.dumps(old, indent=2))
    print(f"[generate] {cfg['name']}: {dt / 60:.1f} min")
