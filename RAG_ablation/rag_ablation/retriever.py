"""
Hybrid dense + BM25 retriever over the RAG chunks (generalises HybridRetriever in
retrieval_from_rag.py so every factor can be ablated).

Scoring is identical to the original script:
    candidates = top-`cand` dense  U  top-`cand` BM25
    score      = alpha * cosine  +  (1 - alpha) * bm25 / max(bm25)
(the original script used alpha = 0.5). Then the section filter (substring of the cleaned
section name, as before), self-exclusion of the target paper, optional temporal filter and
de-duplication to distinct PAPERS are applied; if fewer than k papers survive, the candidate
pool is enlarged (x10) instead of silently returning fewer results.
BM25 reproduces rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25) with sparse matrices.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75, epsilon=0.25):
        from scipy.sparse import csc_matrix
        self.k1, self.b = k1, b
        vocab, rows, cols, vals = {}, [], [], []
        dl = np.zeros(len(docs_tokens), dtype=np.float32)
        for i, toks in enumerate(docs_tokens):
            dl[i] = len(toks)
            for w, c in Counter(toks).items():
                j = vocab.setdefault(w, len(vocab))
                rows.append(i); cols.append(j); vals.append(c)
        self.vocab, self.dl = vocab, dl
        self.avgdl = float(dl.mean()) if len(dl) else 1.0
        self.tf = csc_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(len(docs_tokens), len(vocab)))
        n = len(docs_tokens)
        df = np.diff(self.tf.indptr)
        idf = np.log(n - df + 0.5) - np.log(df + 0.5)
        eps = epsilon * float(idf.mean()) if len(idf) else 0.0
        self.idf = np.where(idf < 0, eps, idf).astype(np.float32)
        self.norm = (k1 * (1 - b + b * dl / self.avgdl)).astype(np.float32)

    def scores(self, query_tokens):
        s = np.zeros(self.tf.shape[0], dtype=np.float32)
        for w, qc in Counter(query_tokens).items():
            j = self.vocab.get(w)
            if j is None:
                continue
            lo, hi = self.tf.indptr[j], self.tf.indptr[j + 1]
            ix, f = self.tf.indices[lo:hi], self.tf.data[lo:hi]
            s[ix] += qc * self.idf[j] * f * (self.k1 + 1) / (f + self.norm[ix])
        return s


class DenseIndex:
    """sentence-transformers encoder (falls back to transformers mean pooling); embeddings cached."""

    def __init__(self, model_path, texts, cache_dir, batch=64, device=None):
        self.model_path = model_path
        self.device = device
        self._load()
        h = hashlib.sha1((model_path + str(len(texts)) + "".join(t[:50] for t in texts[::max(1, len(texts) // 500)])
                          ).encode()).hexdigest()[:16]
        p = Path(cache_dir) / f"emb_{Path(model_path).name}_{h}.npy"
        if p.exists():
            self.X = np.load(p)
        else:
            self.X = self.encode(texts, batch)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, self.X)
        print(f"[dense] {Path(model_path).name}: {self.X.shape}")

    def _load(self):
        self.st = self.hf = None
        if self.model_path == "mock":
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(self.model_path, device=self.device)
        except Exception as e:  # noqa: BLE001
            import torch
            from transformers import AutoModel, AutoTokenizer
            print(f"[dense] sentence-transformers failed ({e}); using transformers mean pooling")
            self.tok = AutoTokenizer.from_pretrained(self.model_path)
            self.hf = AutoModel.from_pretrained(self.model_path).eval()
            self.dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.hf.to(self.dev)

    def encode(self, texts, batch=64):
        texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        if self.model_path == "mock":                       # tests: hashed bag of words
            from sklearn.feature_extraction.text import HashingVectorizer
            X = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2").transform(texts).toarray()
            return X.astype(np.float32)
        if self.st is not None:
            X = self.st.encode(texts, batch_size=batch, convert_to_numpy=True, show_progress_bar=False,
                               normalize_embeddings=True)
            return np.asarray(X, np.float32)
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                             return_tensors="pt").to(self.dev)
                h = self.hf(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = ((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy()
                out.append(v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None))
        return np.concatenate(out).astype(np.float32)


def tokenize(t):
    return str(t).lower().split()          # identical to the original BM25 tokenisation


class HybridRetriever:
    def __init__(self, corpus, bm25: BM25, dense: DenseIndex):
        self.C, self.bm25, self.dense = corpus, bm25, dense
        self.clean_sec = [re.sub(r"[^a-zA-Z]", "", s).lower() for s in corpus.sections]

    def search_docs(self, query, k, alpha=0.5, cand=100, section_filter=None, exclude=frozenset(),
                    max_year=None, keep_unknown_year=True):
        """-> ranked list of (doc_id, score, chunk_idx) with k distinct papers."""
        q = str(query or "").strip()
        if not q or k <= 0:
            return []
        qv = self.dense.encode([q])[0]
        d = self.dense.X @ qv
        s = self.bm25.scores(tokenize(q))
        max_s = float(s.max()) if len(s) and s.max() > 0 else 1.0
        flt = re.sub(r"[^a-zA-Z]", "", str(section_filter)).lower() if section_filter else ""
        n = len(d)
        c = cand
        while True:
            di = np.argpartition(-d, min(c, n) - 1)[:min(c, n)]
            si = np.argpartition(-s, min(c, n) - 1)[:min(c, n)]
            comb = {}
            for i in di:
                comb[int(i)] = comb.get(int(i), 0.0) + float(d[i]) * alpha
            for i in si:
                comb[int(i)] = comb.get(int(i), 0.0) + float(s[i]) / max_s * (1 - alpha)
            out, seen = [], set()
            for i, sc in sorted(comb.items(), key=lambda x: -x[1]):
                if flt and flt not in self.clean_sec[i]:
                    continue
                did = self.C.doc_of[i]
                if did in seen or did in exclude:
                    continue
                if max_year is not None:
                    y = self.C.docs[did]["year"]
                    if (np.isnan(y) and not keep_unknown_year) or (not np.isnan(y) and y >= max_year):
                        continue
                seen.add(did)
                out.append((did, sc, i))
                if len(out) >= k:
                    return out
            if c >= n:
                return out
            c = min(n, c * 10)
