"""
RAG knowledge source -> chunks + paper-level records.

Reads the same file as retrieval_from_rag.py (rag_data_source.csv, column `rag_db` holding one
python-literal dict per chunk with at least `chunk_key`, `text`, `section_name`). Because the
ablations need PAPER-level units (for relevant_papers_list / retrieved_abs_int and for nDCG
against each paper's reference list), chunks are grouped into papers. The paper id, title and
year fields are auto-detected and printed; override with --doc_id_field / --title_field /
--year_field / --doc_id_regex if the detection is wrong.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DOC_ID_KEYS = ["paper_id", "doc_id", "paperId", "pid", "source_id", "corpus_id", "arxiv_id", "openreview_id",
               "paper", "id", "title", "paper_title"]
TITLE_KEYS = ["title", "paper_title", "Title", "doc_title"]
YEAR_KEYS = ["year", "pub_year", "publication_year", "Year", "date", "published"]
SECTION_KEYS = ["section_name", "section", "heading", "section_title"]
TEXT_KEYS = ["text", "chunk_text", "content", "chunk"]
SEC_WORDS = r"(abstract|introduction|intro|method|methods|methodology|approach|experiments?|experimental|results?|" \
            r"evaluation|conclusions?|discussion|related[_ ]?work|background|limitations?)"


def coerce_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:  # noqa: BLE001
        pass
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, (list, tuple, set)):
        return " ".join(s for s in (coerce_text(i) for i in x) if s).strip()
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, np.generic):
        x = x.item()
    return str(x).strip()


def is_bad_text(s: Any) -> bool:
    return not isinstance(s, str) or s.strip().lower() in {"", "none", "nan", "<na>", "null"}


def parse_obj(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str) or not x.strip():
        return None
    for f in (ast.literal_eval, json.loads):
        try:
            return f(x.strip())
        except Exception:  # noqa: BLE001
            continue
    return None


def norm_title(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", coerce_text(t).lower())
    return re.sub(r"\s+", " ", t).strip()


def canon_section(s: str) -> str:
    s = re.sub(r"[^a-z]", "", coerce_text(s).lower())
    for k, pats in (("abstract", ("abstract",)), ("introduction", ("introduction", "intro")),
                    ("method", ("method", "approach")), ("experiment", ("experiment", "evaluation", "result")),
                    ("conclusion", ("conclusion", "discussion"))):
        if any(p in s for p in pats):
            return k
    return s or "other"


def parse_year(v) -> float:
    m = re.search(r"(19|20)\d{2}", coerce_text(v))
    return float(m.group(0)) if m else np.nan


class Corpus:
    def __init__(self):
        self.keys: List[str] = []
        self.texts: List[str] = []
        self.sections: List[str] = []        # raw section name
        self.csec: List[str] = []            # canonical section
        self.doc_of: List[str] = []
        self.docs: Dict[str, dict] = {}
        self.title_index: Dict[str, set] = defaultdict(set)
        self.mapping = {}

    def __len__(self):
        return len(self.texts)


def _pick(records, keys, min_share=0.9):
    for k in keys:
        n = sum(1 for r in records if not is_bad_text(coerce_text(r.get(k, ""))))
        if records and n / len(records) >= min_share:
            return k
    return None


def load_corpus(rag_csv, rag_db_col="rag_db", doc_id_field="auto", title_field="auto", year_field="auto",
                doc_id_regex="", max_chars=20000, limit=0) -> Corpus:
    df = pd.read_csv(rag_csv, nrows=limit or None)
    recs = []
    if rag_db_col in df.columns:
        for s in df[rag_db_col].tolist():
            d = parse_obj(s)
            if isinstance(d, dict):
                recs.append(d)
    else:                                                # already one column per field
        recs = df.to_dict("records")
    if not recs:
        raise SystemExit(f"no usable records in {rag_csv} (column '{rag_db_col}')")
    sample = recs[:2000]
    tkey = _pick(sample, TEXT_KEYS) or "text"
    skey = _pick(sample, SECTION_KEYS, 0.5) or "section_name"
    dkey = (None if doc_id_field == "auto" else doc_id_field) or _pick(sample, DOC_ID_KEYS)
    ttl = (None if title_field == "auto" else title_field) or _pick(sample, TITLE_KEYS, 0.5)
    yk = (None if year_field == "auto" else year_field) or _pick(sample, YEAR_KEYS, 0.3)
    rx = re.compile(doc_id_regex) if doc_id_regex else None

    def doc_id(r):
        if rx is not None:
            m = rx.search(coerce_text(r.get("chunk_key", "")))
            if m:
                return m.group(1) if m.groups() else m.group(0)
        if dkey:
            v = coerce_text(r.get(dkey, ""))
            if v:
                return norm_title(v) if dkey in TITLE_KEYS else v
        ck = coerce_text(r.get("chunk_key", ""))
        ck = re.sub(r"([_:#|\-](chunk|sec|section|part|c)?\d+)+$", "", ck, flags=re.I)
        ck = re.sub(r"[_:#|\-]+" + SEC_WORDS + r"$", "", ck, flags=re.I)
        return ck

    C = Corpus()
    C.mapping = {"text": tkey, "section": skey, "doc_id": dkey or ("regex" if rx else "chunk_key prefix"),
                 "title": ttl, "year": yk}
    seen = set()
    for r in recs:
        txt = coerce_text(r.get(tkey, ""))[:max_chars]
        if is_bad_text(txt):
            continue
        key = coerce_text(r.get("chunk_key", "")) or f"chunk{len(C.keys)}"
        if key in seen:
            continue
        seen.add(key)
        did = doc_id(r) or key
        sec = coerce_text(r.get(skey, ""))
        cs = canon_section(sec)
        C.keys.append(key); C.texts.append(txt); C.sections.append(sec); C.csec.append(cs); C.doc_of.append(did)
        d = C.docs.setdefault(did, {"title": "", "year": np.nan, "sections": {}, "chunks": []})
        if ttl and not d["title"]:
            d["title"] = coerce_text(r.get(ttl, ""))
        if yk and np.isnan(d["year"]):
            d["year"] = parse_year(r.get(yk, ""))
        d["chunks"].append(len(C.keys) - 1)
        if cs not in d["sections"]:
            d["sections"][cs] = txt
    for did, d in C.docs.items():
        if d["title"]:
            C.title_index[norm_title(d["title"])].add(did)
    n_title = sum(1 for d in C.docs.values() if d["title"])
    n_year = sum(1 for d in C.docs.values() if not np.isnan(d["year"]))
    secs = Counter(C.csec).most_common(8)
    print(f"[corpus] {len(C)} chunks -> {len(C.docs)} papers | fields {C.mapping}")
    print(f"[corpus] papers with title: {n_title}  with year: {n_year} | sections: {secs}")
    ex = list(zip(C.keys[:3], C.doc_of[:3]))
    print(f"[corpus] chunk_key -> paper id examples: {ex}")
    if len(C.docs) > 0.9 * len(C):
        print("[corpus] WARNING: almost one paper per chunk — the paper id was probably not detected. "
              "Set DOC_ID_FIELD / DOC_ID_REGEX (see the examples above).")
    return C
