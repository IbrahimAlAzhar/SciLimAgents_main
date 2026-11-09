"""
Ablation configurations. Each changes ONE factor relative to `baseline`.

  factor            values                                  baseline
  ---------------   -------------------------------------   ---------------------------------
  hybrid weight     alpha (dense share) 1.0 0.7 0.5 0.3 0.0  BASE_ALPHA (0.5 = retrieval_from_rag.py)
  query design      per_section | whole_paper | abstract     per_section (SECTIONS, each filtered
                                                             to the same section of the corpus)
  depth             k papers per section query: 1 3 5        3  (x 3 sections = 9 papers)
  memory format     abs_intro | abstract | chunks | summary6 abs_intro  (same per-paper token budget)
  embedding model   minilm | gte_small | <extra name=path>   minilm (all-MiniLM-L6-v2)
  temporal          only corpus papers published BEFORE      off
                    the target paper
  memory on/off     mem_off: no retrieved context at all     on
  original          the retrieval columns already in the input CSV (sanity check vs. rebuilt baseline)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class RagConfig:
    name: str = "baseline"
    alpha: float = 0.5
    query: str = "per_section"             # per_section | whole_paper | abstract
    sections: tuple = ("abstract", "introduction", "method")
    k: int = 3                             # papers per section query (per_section) / k*len(sections) otherwise
    fmt: str = "abs_intro"                 # abs_intro | abstract | chunks | summary6
    embed: str = "minilm"
    temporal: bool = False
    memory: bool = True
    original: bool = False
    item_tokens: int = 600                 # per retrieved paper (same budget for every memory format)
    cand: int = 100
    section_filter: bool = True
    keep_citation_col: bool = True         # keep ret_abs_int_cit (citation-based context) unchanged
    seed: int = 0                          # generation seed (repeat baseline for seed variance)
    extra: dict = field(default_factory=dict)

    @property
    def n_papers(self):
        return self.k * len(self.sections)

    def to_json(self):
        d = asdict(self)
        d["sections"] = list(self.sections)
        d["n_papers"] = self.n_papers
        return d

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text())
        d.pop("n_papers", None)
        d["sections"] = tuple(d["sections"])
        return RagConfig(**d)


FACTORS = {
    "alpha": ["alpha_1.0", "alpha_0.7", "alpha_0.5", "alpha_0.3", "alpha_0.0"],
    "query": ["query_whole", "query_abstract"],
    "depth": ["k1", "k5"],
    "format": ["fmt_abstract", "fmt_chunks", "fmt_summary6"],
    "embed": ["emb_gte_small"],
    "temporal": ["temporal"],
    "memory": ["mem_off"],
    "original": ["original"],
}
ALL = ["baseline"] + [c for v in FACTORS.values() for c in v]


def make(name, base: RagConfig) -> RagConfig:
    """'alpha_0.7', 'k5', 'fmt_chunks', 'emb_<name>', 'query_whole', 'baseline@seed43', ..."""
    seed = None
    if "@seed" in name:
        name_, s = name.split("@seed", 1)
        seed = int(s)
    else:
        name_ = name
    if name_ == "baseline":
        c = replace(base)
    elif name_.startswith("alpha_"):
        c = replace(base, alpha=float(name_.split("_", 1)[1]))
    elif name_ == "query_whole":
        c = replace(base, query="whole_paper")
    elif name_ == "query_abstract":
        c = replace(base, query="abstract")
    elif name_.startswith("k") and name_[1:].isdigit():
        c = replace(base, k=int(name_[1:]))
    elif name_.startswith("fmt_"):
        c = replace(base, fmt=name_[4:])
    elif name_.startswith("emb_"):
        c = replace(base, embed=name_[4:])
    elif name_ == "temporal":
        c = replace(base, temporal=True)
    elif name_ == "mem_off":
        c = replace(base, memory=False)
    elif name_ == "original":
        c = replace(base, original=True)
    else:
        raise SystemExit(f"unknown config '{name}'. Known: {ALL} (+ emb_<name>, k<int>, alpha_<x>, <cfg>@seed<n>)")
    c.name = name
    if seed is not None:
        c.seed = seed
    return c


def is_duplicate_of_baseline(c: RagConfig, base: RagConfig) -> bool:
    a, b = c.to_json(), base.to_json()
    for d in (a, b):
        d.pop("name", None)
    return a == b
