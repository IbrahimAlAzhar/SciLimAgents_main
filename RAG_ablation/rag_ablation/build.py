"""
STAGE build — retrieve with every configuration and write one input CSV per configuration.

For each target paper (row of INPUT_CSV) and each configuration:
  * queries are built from the paper (`pdf_text` science-parse dict: abstract + section texts,
    exactly like retrieval_from_rag.py; falls back to `input_text_cleaned`)
  * papers are retrieved (self-excluded; temporal filter if enabled)
  * the three columns scilimagents_main reads are REWRITTEN in their usual formats:
      relevant_papers_list  python list of strings, one per retrieved paper   (NovAgents, LimAgents v1)
      retrieved_abs_int     {doc_id: {title, abstractText, sections:[Introduction]}}  (RAG, @mem)
      ret_abs_int_cit       citation-based context: kept unchanged (KEEP_CITATION_COL=1) or cleared
  * rag_abl__retrieved  JSON ranked list [{doc_id, title, year, score, rank, query}] (for IR metrics)
    rag_abl__target_year, rag_abl__context_chars, rag_abl__n_retrieved
Output: <work>/configs/<name>/input.csv + config.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .configs import RagConfig, is_duplicate_of_baseline, make
from .corpus import coerce_text, is_bad_text, norm_title, parse_obj, parse_year

SEC_ALIASES = {"abstract": ("abstract",), "introduction": ("introduction", "intro"),
               "method": ("method", "methodology", "approach"), "experiments": ("experiment", "experimental"),
               "conclusion": ("conclusion", "conclusions", "discussion"), "related": ("related",)}
FILTER_KEY = {"abstract": "abstract", "introduction": "introduction", "method": "method",
              "experiments": "experiment", "conclusion": "conclusion", "related": "related"}


def target_parts(row, pdf_col="pdf_text", text_col="input_text_cleaned"):
    """-> (title, {section: text}, whole_text)"""
    d = parse_obj(row.get(pdf_col)) if pdf_col in row else None
    secs, title = {}, ""
    if isinstance(d, dict):
        title = coerce_text(d.get("title", ""))
        a = coerce_text(d.get("abstractText", ""))
        if not is_bad_text(a):
            secs["abstract"] = a
        for s in d.get("sections", []) or []:
            if not isinstance(s, dict):
                continue
            h = re.sub(r"\s+", " ", coerce_text(s.get("heading", "")).lower())
            t = coerce_text(s.get("text", ""))
            if is_bad_text(h) or is_bad_text(t):
                continue
            for canon, al in SEC_ALIASES.items():
                if any(x in h for x in al):
                    secs[canon] = (secs.get(canon, "") + "\n\n" + t).strip()
    if not secs:
        txt = coerce_text(row.get(text_col, ""))
        m = re.search(r"abstract[:\s]*(.{200,2500}?)(?:\n\s*\n|introduction)", txt, re.I | re.S)
        secs["abstract"] = (m.group(1) if m else txt[:2500]).strip()
        m = re.search(r"introduction(.{200,6000}?)(?:\n\s*\n\d|\n\s*\d+\s+[A-Z])", txt, re.I | re.S)
        if m:
            secs["introduction"] = m.group(1).strip()
    if not title:
        title = coerce_text(row.get("title", "")) or ""
    whole = "\n\n".join(secs.values()) if isinstance(d, dict) else coerce_text(row.get(text_col, ""))
    return title, secs, whole


def target_year(row, year_col=""):
    for c in ([year_col] if year_col else []) + ["year", "pub_year", "_source_file", "submission", "id", "venue"]:
        if c and c in row and not is_bad_text(coerce_text(row.get(c))):
            y = parse_year(row.get(c))
            if not np.isnan(y):
                return y
    return np.nan


def approx_trunc(text, tokens):
    w = str(text or "").split()
    n = int(tokens / 1.3)                       # ~1.3 tokens per word
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def doc_item(C, did, fmt, item_tokens, hit_chunks, summaries=None):
    d = C.docs[did]
    t = d["title"]
    a = d["sections"].get("abstract", "")
    intro = d["sections"].get("introduction", "")
    if fmt == "abstract":
        body = f"Abstract: {a}" if a else f"Text: {C.texts[d['chunks'][0]]}"
    elif fmt == "chunks":
        body = "\n".join(f"[{C.sections[i] or 'chunk'}] {C.texts[i]}" for i in hit_chunks)
    elif fmt == "summary6":
        body = (summaries or {}).get(did) or "SUMMARY_PENDING"
    else:                                                   # abs_intro (original layout)
        body = f"Abstract: {a}\nIntroduction: {intro}" if (a or intro) else f"Text: {C.texts[d['chunks'][0]]}"
    head = f"Title: {t}\n" if t else ""
    return head + approx_trunc(body, max(50, item_tokens - len(t.split())))


def abs_int_record(C, did, item_text, fmt):
    d = C.docs[did]
    rec = {"title": d["title"], "abstractText": item_text.split("\n", 1)[-1] if d["title"] else item_text}
    if fmt == "abs_intro":
        a = d["sections"].get("abstract", "")
        i = d["sections"].get("introduction", "")
        rec = {"title": d["title"], "abstractText": a or rec["abstractText"],
               "sections": [{"heading": "Introduction", "text": i}] if i else []}
    return rec


def retrieve_row(R, C, cfg: RagConfig, title, secs, whole, ty):
    excl = frozenset(C.title_index.get(norm_title(title), set())) if title else frozenset()
    max_year = ty if (cfg.temporal and not np.isnan(ty)) else None
    if cfg.query == "per_section":
        lists = []
        for s in cfg.sections:
            q = secs.get(s, "")
            if not q:
                continue
            # retrieve deeper than k so the fused list can reach k x sections DISTINCT papers
            hits = R.search_docs(q, cfg.n_papers, alpha=cfg.alpha, cand=cfg.cand,
                                 section_filter=FILTER_KEY.get(s, s) if cfg.section_filter else None,
                                 exclude=excl, max_year=max_year)
            lists.append((s, hits))
        out, seen = [], set()                               # round-robin fusion of the section rankings
        for r in range(max((len(h) for _, h in lists), default=0)):
            for s, hits in lists:
                if r < len(hits) and hits[r][0] not in seen:
                    seen.add(hits[r][0])
                    out.append({"doc_id": hits[r][0], "score": hits[r][1], "chunk": hits[r][2], "query": s})
        return out[:cfg.n_papers]   # round-robin: rank 1 of every section first, then rank 2, ...
    q = secs.get("abstract", "") if cfg.query == "abstract" else whole
    hits = R.search_docs(q[:20000], cfg.n_papers, alpha=cfg.alpha, cand=cfg.cand, section_filter=None,
                         exclude=excl, max_year=max_year)
    return [{"doc_id": h[0], "score": h[1], "chunk": h[2], "query": cfg.query} for h in hits]


def build_config(cfg, df, C, R, out_dir, summaries=None, pdf_col="pdf_text", text_col="input_text_cleaned",
                 year_col=""):
    out = df.copy()
    rel, abs_int, meta, ty_col, chars = [], [], [], [], []
    for _, row in df.iterrows():
        ty = target_year(row, year_col)
        ty_col.append(ty)
        if cfg.original:
            rel.append(row.get("relevant_papers_list")); abs_int.append(row.get("retrieved_abs_int"))
            meta.append("[]"); chars.append(len(coerce_text(row.get("relevant_papers_list"))))
            continue
        if not cfg.memory:
            rel.append("[]"); abs_int.append("{}"); meta.append("[]"); chars.append(0)
            continue
        title, secs, whole = target_parts(row, pdf_col, text_col)
        hits = retrieve_row(R, C, cfg, title, secs, whole, ty)
        items, recs, m = [], {}, []
        for rank, h in enumerate(hits, 1):
            did = h["doc_id"]
            hit_chunks = [x["chunk"] for x in hits if x["doc_id"] == did]
            it = doc_item(C, did, cfg.fmt, cfg.item_tokens, hit_chunks, summaries)
            items.append(it)
            recs[str(did)] = abs_int_record(C, did, it, cfg.fmt)
            d = C.docs[did]
            m.append({"doc_id": str(did), "title": d["title"], "year": None if np.isnan(d["year"]) else int(d["year"]),
                      "score": round(float(h["score"]), 5), "rank": rank, "query": h["query"]})
        rel.append(repr(items)); abs_int.append(repr(recs)); meta.append(json.dumps(m))
        chars.append(sum(len(x) for x in items))
    out["relevant_papers_list"] = rel
    out["retrieved_abs_int"] = abs_int
    if (not cfg.memory or not cfg.keep_citation_col) and "ret_abs_int_cit" in out.columns:
        out["ret_abs_int_cit"] = "{}"                     # memory off = no retrieved / cited context at all
    out["rag_abl__retrieved"] = meta
    out["rag_abl__target_year"] = ty_col
    out["rag_abl__context_chars"] = chars
    out["rag_abl__n_retrieved"] = [len(json.loads(x)) for x in meta]
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "input.csv", index=False)
    (d / "config.json").write_text(json.dumps(cfg.to_json(), indent=2))
    n = out["rag_abl__n_retrieved"]
    print(f"[build] {cfg.name:16s} papers/row mean={n.mean():.1f} min={n.min()}  context chars mean="
          f"{np.mean(chars):.0f}  -> {d / 'input.csv'}")
    return out


def run_build(a):
    from .corpus import load_corpus
    from .retriever import BM25, DenseIndex, HybridRetriever, tokenize
    work = Path(a.work_dir)
    df = pd.read_csv(a.input_csv)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))
    df = df.iloc[a.start_row:(a.start_row + a.n_rows) if a.n_rows else None].reset_index(drop=True)
    print(f"[build] {len(df)} target papers from {a.input_csv}")
    base = RagConfig(alpha=a.base_alpha, sections=tuple(a.sections.split()), k=a.k, item_tokens=a.item_tokens,
                     cand=a.cand, keep_citation_col=bool(a.keep_citation_col), section_filter=bool(a.section_filter))
    cfgs = []
    for name in a.configs.split():
        c = make(name, base)
        if name != "baseline" and "@seed" not in name and is_duplicate_of_baseline(c, base):
            print(f"[build] {name} == baseline -> skipped")
            continue
        cfgs.append(c)
    need_retrieval = [c for c in cfgs if c.memory and not c.original]
    if need_retrieval:
        C = load_corpus(a.rag_csv, a.rag_db_col, a.doc_id_field, a.title_field, a.year_field, a.doc_id_regex,
                        limit=a.corpus_limit)
        bm = BM25([tokenize(t) for t in C.texts])
        embeds = dict(x.split("=", 1) for x in a.embed_models.split())
        dense = {}
    else:
        C = None
    for c in cfgs:
        d = work / "configs" / c.name
        if (d / "input.csv").exists() and not a.overwrite and c.fmt != "summary6":
            print(f"[build] {c.name}: exists -> kept (OVERWRITE=1 to rebuild)")
            continue
        R = None
        if c.memory and not c.original:
            if c.embed not in embeds:
                raise SystemExit(f"embedding '{c.embed}' not in EMBED_MODELS ({list(embeds)})")
            if c.embed not in dense:
                dense[c.embed] = DenseIndex(embeds[c.embed], C.texts, work / "emb_cache")
            R = HybridRetriever(C, bm, dense[c.embed])
        summ = None
        if c.fmt == "summary6":
            sp = work / "summary6.json"
            summ = json.loads(sp.read_text()) if sp.exists() else {}
        build_config(c, df, C, R, d, summ, a.pdf_col, a.text_col, a.year_col)
        if c.fmt == "summary6":                     # remember which papers need an LLM summary
            pend = set()
            for m in pd.read_csv(d / "input.csv")["rag_abl__retrieved"]:
                pend |= {x["doc_id"] for x in json.loads(m)}
            todo = sorted(p for p in pend if p not in (summ or {}))
            (d / "summary6_todo.json").write_text(json.dumps(todo))
            print(f"[build] {c.name}: {len(todo)} retrieved papers need a 6-field summary (stage summarize)")
    if C is not None:
        import pickle
        with open(work / "corpus_docs.pkl", "wb") as f:
            pickle.dump({"docs": {k: {kk: v[kk] for kk in ("title", "year", "sections")} for k, v in C.docs.items()},
                         "title_index": dict(C.title_index), "mapping": C.mapping}, f)
