#!/usr/bin/env python
"""
eval_rag_ablation.py - evaluates every RAG-ablation configuration x every generation file x every limitation column
with YOUR pairwise LLM-as-a-judge (gpt-4o-mini, Yes/No prompt, same as eval_limitations.py) + NLP similarity, measures
retrieval quality of the same configurations, and writes one table per ablation type (P, R, F1, NLP metrics and
retrieval metrics side by side).

Input (written by `python -m rag_ablation build/generate`):
  <work>/configs/<config>/config.json                      settings of the configuration
  <work>/configs/<config>/input.csv                        rebuilt retrieval context (rag_abl__retrieved = ranked list)
  <work>/configs/<config>/out/generations/<system>.csv     limagents.csv  novagents.csv  merge.csv  rag.csv
  <work>/corpus_docs.pkl                                   RAG corpus titles (for retrieval relevance labels)

Columns evaluated (--columns):
  final    (default) the final limitations of each system: limagents, novagents, merge, rag
  agents   final + every specialist agent: limagents__<X>_Agent, novagents__<X>_Agent
  all      agents + merge__lim_report, merge__nov_report
  or an explicit list "limagents,merge,novagents__Methodological_Novelty_Agent"
Not evaluated (they are not limitations): *__relevant_paper_sum*, *__full_chat_history, rag__retrieval_*.

Per (config, column, paper):
  1. ground truth (ground_truth_lim_peer) and generated text -> units split on newline (bullets, '**', < 3 words,
     header lines such as 'Here is the consolidated list ...:' and duplicates removed)
  2. the judge answers every (GT unit, limitation unit) pair with your prompt -> Yes / No
  3. recall = GT units with >= 1 Yes / GT units; precision = limitation units with >= 1 Yes / limitation units; F1
  4. TF-IDF cosine, Jaccard, ROUGE-L F1, BERTScore F1 averaged over the Yes pairs
  Generation failures (text starting with 'ERROR') are excluded (reported as n_error), empty outputs score 0.
Retrieval (no LLM): relevant = papers cited by the target paper that exist in the RAG corpus (title match);
  P@k, R@k, nDCG@k, Hit@k, MRR, R@all (recall of everything retrieved). Not defined for mem_off / original.
Statistics: mean + 95% bootstrap CI over papers; every configuration vs baseline on the SAME papers (paired randomization
  test, Holm correction per column and metric); seed noise from baseline@seed*; Spearman link retrieval -> F1.

Modes
  --mode merge_dry  no API call: per configuration, how many 'merge' rows are ERROR / repairable + cost estimate
  --mode merge      repairs every ERROR row of configs/<cfg>/out/generations/merge.csv with gpt-4o-mini and YOUR merge
                    prompt (Report A = limagents final = anchor, Report B = novagents final, 10-14 limitations).
                    A / B are taken from merge__lim_report / merge__nov_report, or (if empty) from limagents.csv /
                    novagents.csv of the SAME configuration (matched by row_id). The original file is copied once to
                    out/generations_backup/merge.csv; repaired rows get merge__repaired=True. Resumable (merge_cache.jsonl).
  --mode dry        no API call: unit split examples + number of pairs + cost estimate
  --mode test       one paper of one column of baseline, prints every judged pair
  --mode run        everything; resumable (judge_cache.jsonl: no pair is paid twice; extra configs only judge new pairs)
  --tag _agents     suffix for every output file, so an agent-level run does not overwrite the main run
                    (the judge cache is shared: pairs judged in one run are free in the other)

Output (--out_dir, default <this folder>/<model_tag>):
  RAG_EVAL_TABLES.txt          all tables (start here)
  tables.tex                   the same tables in LaTeX (booktabs)
  summary.csv                  one row per (config, column): P/R/F1 + CI + deltas + p + NLP + retrieval
  per_paper.csv                one row per (config, column, paper)
  ir_per_paper.csv, ir_summary.csv
  ir_f1_link.csv               Spearman retrieval quality vs F1
  configs_eval/<config>/<system>_eval.csv   the generation file + <col>__precision/__recall/__f1/... per paper
  plots/*.png
  judge_cache.jsonl, bertscore_cache.jsonl, output.txt, error.txt

    python eval/eval_rag_ablation.py --mode merge_dry --work_dir outputs/qwen3_8b
    python eval/eval_rag_ablation.py --mode merge     --work_dir outputs/qwen3_8b
    python eval/eval_rag_ablation.py --mode dry  --work_dir outputs/qwen3_8b
    python eval/eval_rag_ablation.py --mode run  --work_dir outputs/qwen3_8b --columns agents --configs "baseline mem_off" --tag _agents
    python eval/eval_rag_ablation.py --mode test --work_dir outputs/qwen3_8b
    python eval/eval_rag_ablation.py --mode run  --work_dir outputs/qwen3_8b --n_rows -1
"""
import argparse
import ast
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAG_DIR = HERE.parent
sys.path.insert(0, str(RAG_DIR))
PROMPT_VERSION = "pairwise_yesno_v1"            # same cache key as eval_limitations.py (caches are interchangeable)
SYSTEM_ORDER = ["limagents", "novagents", "merge", "rag"]


class Tee:
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def write(self, s):
        with self.lock:
            self.stream.write(s)
            self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def warn(*a):
    print(time.strftime("[%H:%M:%S] WARNING:"), *a, file=sys.stderr, flush=True)


# ============================================================================= judge (your prompt, verbatim)
def judge_prompt(gt_text, llm_text):
    description1 = f"ground truth limitations: {gt_text}"
    description2 = f"llm generated limitations: {llm_text}"
    return ("Check whether 'list2' contains a topic or limitation from 'list1' "
            "or 'list1' contains a topic or limitation from 'list2'.\n\n"
            "Your answer should be \"Yes\" or \"No\".\n"
            f"List 1: {description1}\n"
            f"List 2: {description2}\n")


def is_yes(answer):
    m = re.match(r"\W*(yes|no)\b", str(answer), re.I)
    if m:
        return m.group(1).lower() == "yes"
    return "YES" in str(answer).upper()


class FatalAPIError(RuntimeError):
    pass


class Judge:
    def __init__(self, model, cache_path, backend="openai", max_tokens=10):
        self.model, self.backend, self.max_tokens = model, backend, max_tokens
        self.cache_path, self.cache, self.lock = cache_path, {}, threading.Lock()
        if os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    r = json.loads(line)
                    self.cache[r["k"]] = r["a"]
                except Exception:
                    continue
        self.client = None
        if backend == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY is not set (export it in the PBS file or keep it in ~/.openai_key)")
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=60)
            except ImportError:
                self.client = None

    def key(self, gt, llm):
        return hashlib.sha1(f"{PROMPT_VERSION}|{self.model}|{gt}\x00{llm}".encode()).hexdigest()

    def _call(self, prompt):
        if self.backend == "mock":
            a = set(re.findall(r"[a-z]{4,}", prompt.split("List 1:")[1].split("List 2:")[0].lower()))
            b = set(re.findall(r"[a-z]{4,}", prompt.split("List 2:")[1].lower()))
            return "Yes" if len(a & b) / max(1, min(len(a), len(b))) >= 0.3 else "No"
        if self.client is not None:
            r = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}],
                                                    temperature=0, max_tokens=self.max_tokens, stream=False)
            return (r.choices[0].message.content or "").strip()
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                           "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()

    def ask(self, gt, llm, tries=8):
        last = ""
        for t in range(tries):
            try:
                return self._call(judge_prompt(gt, llm)), None
            except Exception as e:
                code = getattr(e, "status_code", None) or getattr(e, "code", None)
                last = f"{type(e).__name__}: {str(e)[:200]}"
                if code in (401, 403) or "invalid_api_key" in str(e) or "insufficient_quota" in str(e):
                    raise FatalAPIError(last)
                time.sleep(min(60, 2 ** t) + random.random())
        return None, last

    def run(self, pairs, workers=16):
        todo, seen = [], set()
        for gt, llm in pairs:
            k = self.key(gt, llm)
            if k not in self.cache and k not in seen:
                seen.add(k)
                todo.append((k, gt, llm))
        log(f"[judge] {len(todo):,} new pairs to judge ({len(self.cache):,} already in the cache)")
        if not todo:
            return 0
        fh = open(self.cache_path, "a")
        t0, done, errors = time.time(), 0, 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self.ask, gt, llm): (k, gt, llm) for k, gt, llm in todo}
                for f in as_completed(futs):
                    k, gt, llm = futs[f]
                    ans, err = f.result()
                    done += 1
                    if err:
                        errors += 1
                        if errors <= 20:
                            warn(f"judge error (pair counted as No, retried on the next run): {err}")
                    else:
                        with self.lock:
                            self.cache[k] = ans
                            fh.write(json.dumps({"k": k, "a": ans, "gt": gt, "llm": llm, "m": self.model}) + "\n")
                            if done % 50 == 0:
                                fh.flush()
                    if done % 2000 == 0 or done == len(todo):
                        el = time.time() - t0
                        log(f"[judge] {done:,}/{len(todo):,}  {done / el:.1f} pairs/s  "
                            f"eta {(len(todo) - done) / max(1e-9, done / el) / 60:.1f} min  errors {errors}")
        except FatalAPIError as e:
            raise SystemExit(f"OpenAI API refused the key / quota: {e}")
        finally:
            fh.close()
        return errors

    def answer(self, gt, llm):
        return self.cache.get(self.key(gt, llm))


# ============================================================================= NLP similarity of a pair
TOK = re.compile(r"(?u)\b\w\w+\b")


def tfidf_cosine(a, b):
    ta, tb = TOK.findall(a.lower()), TOK.findall(b.lower())
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return 0.0
    idf = {w: math.log(3 / (1 + (w in sa) + (w in sb))) + 1 for w in sa | sb}
    va = {w: c * idf[w] for w, c in Counter(ta).items()}
    vb = {w: c * idf[w] for w, c in Counter(tb).items()}
    na, nb = math.sqrt(sum(v * v for v in va.values())), math.sqrt(sum(v * v for v in vb.values()))
    return sum(va[w] * vb.get(w, 0.0) for w in va) / (na * nb) if na and nb else 0.0


def jaccard(a, b):
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0


class RougeL:
    def __init__(self):
        try:
            from rouge_score import rouge_scorer
            self.sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            self.kind = "rouge_score (stemmer)"
        except ImportError:
            self.sc, self.kind = None, "own LCS implementation (rouge_score not installed: no stemmer)"

    def __call__(self, ref, cand):
        if self.sc is not None:
            return float(self.sc.score(ref, cand)["rougeL"].fmeasure)
        r, c = re.findall(r"[a-z0-9]+", ref.lower()), re.findall(r"[a-z0-9]+", cand.lower())
        if not r or not c:
            return 0.0
        prev = [0] * (len(c) + 1)
        for x in r:
            cur = [0]
            for j, y in enumerate(c):
                cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        lcs = prev[-1]
        if lcs == 0:
            return 0.0
        p, rr = lcs / len(c), lcs / len(r)
        return 2 * p * rr / (p + rr)


class BertScore:
    """BERTScore F1 (candidate = generated unit, reference = GT unit), batched, cached on disk."""

    def __init__(self, cache_path, model_type=None, num_layers=None, enabled=True, batch_size=64):
        self.cache_path, self.cache, self.kind, self.scorer = cache_path, {}, "disabled", None
        self.model_type, self.num_layers, self.bs = model_type, num_layers, batch_size
        if os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    r = json.loads(line)
                    self.cache[r["k"]] = r["f"]
                except Exception:
                    continue
        self.enabled = enabled

    def _load(self):
        if self.scorer is not None or not self.enabled:
            return
        try:
            import torch
            from bert_score import BERTScorer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            kw = {"model_type": self.model_type} if self.model_type else {"lang": "en"}
            if self.model_type and self.num_layers:
                kw["num_layers"] = self.num_layers
            self.scorer = BERTScorer(device=dev, batch_size=self.bs, **kw)
            self.kind = f"bert_score ({self.model_type or 'lang=en default: roberta-large'}, {dev})"
        except Exception as e:
            warn(f"BERTScore unavailable ({e!r}) -> BERTScore = NaN. Install bert-score and set --bertscore_model "
                 f"to a local roberta-large folder (the cluster is offline)")
            self.enabled, self.kind = False, f"unavailable ({type(e).__name__})"

    def key(self, gt, llm):
        return hashlib.sha1(f"{self.model_type}|{gt}\x00{llm}".encode()).hexdigest()

    def compute(self, pairs):
        if not self.enabled:
            return
        todo = list({self.key(g, l): (g, l) for g, l in pairs if self.key(g, l) not in self.cache}.items())
        if not todo:
            self.kind = self.kind if self.kind != "disabled" else "bert_score (all pairs cached)"
            return
        self._load()
        if not self.enabled:
            return
        log(f"[bertscore] {len(todo):,} new Yes pairs ({len(self.cache):,} cached) with {self.kind}")
        with open(self.cache_path, "a") as fh:
            for s in range(0, len(todo), 2048):
                part = todo[s:s + 2048]
                _, _, F = self.scorer.score([l for _, (g, l) in part], [g for _, (g, l) in part])
                for (k, _), f in zip(part, F.tolist()):
                    self.cache[k] = float(f)
                    fh.write(json.dumps({"k": k, "f": float(f)}) + "\n")
                log(f"[bertscore] {min(s + 2048, len(todo)):,}/{len(todo):,}")

    def get(self, gt, llm):
        return self.cache.get(self.key(gt, llm), float("nan"))


# ============================================================================= splitting into units
BULLET = re.compile(r"^\s*(?:[-*•▪●–]+|\(?\d{1,2}[.)]|\(?[a-hA-H][.)])\s*")
PREAMBLE = re.compile(r"^(here (is|are)|below (is|are)|the following|based on (the|my)|sure[,!.]|certainly[,!.]|"
                      r"(final |consolidated |key )*limitations?( of the paper| identified)?\s*:?\s*$)", re.I)
# reviewer / agent self-introductions and notes: never a limitation
INTRO = re.compile(r"^(i am reviewing|i'm reviewing|i have reviewed|i reviewed|i will review|i will now|"
                   r"as (a|an|the) (reviewer|expert|review leader)|in this review,? i|this review (focuses|examines|evaluates)|"
                   r"review(er)? (leader )?(note|summary)\b|leader note\b|note to (the )?leader)", re.I)
# category / section labels such as 'Category: Theoretical Foundation', 'Category - Data', 'Section: Ethics'
CATEGORY = re.compile(r"^(category|categories|sub-?category|section|aspect|dimension|area|theme|type)\s*(\d+)?\s*[:\-–—]", re.I)
ERROR_RE = re.compile(r"^\s*\[?\s*ERROR\b", re.I)


def _as_list(text):
    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(t)
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v]
            except Exception:
                pass
    return None


def clean_unit(raw):
    s = BULLET.sub("", raw.strip()).replace("**", "").replace("__", "").strip().strip('"').strip("'").strip()
    s = re.sub(r"^#+\s*", "", s)
    return BULLET.sub("", s).strip()


def drop_reason(s, min_words):
    """why a cleaned line is not a limitation unit (None = keep it)."""
    nw = len(s.split())
    if not re.search(r"[A-Za-z]", s):
        return "separator"
    if CATEGORY.match(s):
        return "category label"
    if INTRO.match(s):
        return "reviewer intro / note"
    if s.endswith(":") and (nw <= 8 or PREAMBLE.match(s)):
        return "header"
    if PREAMBLE.match(s) and nw <= 15 and s.endswith((":", ".")) \
            and not re.search(r"\b(lack|limit|does not|fails?|only|no )", s, re.I):
        return "preamble"
    if nw < min_words:
        return f"too short (<= {min_words - 1} words)"
    return None


def split_units(text, seps=("\n",), min_words=6):
    """Split on newline; clean bullets / '**' / '#'; drop separators, 'Category: ...' labels, reviewer intros and
    notes, headers ending with ':', preambles, units with fewer than min_words words and duplicates.
    Returns (units, dropped) where dropped = ['<reason>: <raw line>', ...]."""
    if not isinstance(text, str) or not text.strip() or text.strip().lower() in ("nan", "none", "[]"):
        return [], []
    pieces = _as_list(text) or [text]
    pieces = [p.replace("\r\n", "\n").replace("\\n", "\n") for p in pieces]
    for sep in seps:
        pieces = [q for p in pieces for q in p.split(sep)]
    units, dropped, seen = [], [], set()
    for raw in pieces:
        raw = raw.strip()
        if not raw:
            continue
        s = clean_unit(raw)
        why = drop_reason(s, min_words)
        if why:
            dropped.append(f"[{why}] {raw}")
            continue
        k = re.sub(r"\s+", " ", s.lower())
        if k in seen:
            dropped.append(f"[duplicate] {raw}")
            continue
        seen.add(k)
        units.append(s)
    return units, dropped


def is_error(text):
    return isinstance(text, str) and bool(ERROR_RE.match(text))


# ============================================================================= statistics
def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_p(a, b, n=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    sims = np.abs((d[None, :] * rng.choice([-1, 1], (n, len(d)))).mean(1))
    return float((1 + (sims >= obs - 1e-12).sum()) / (n + 1))


def holm(ps):
    """ps: dict key -> p. Returns dict key -> Holm-adjusted p."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), int(len(x))
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        return float(r.correlation if hasattr(r, "correlation") else r[0]), float(r.pvalue if hasattr(r, "pvalue") else r[1]), int(len(x))
    except Exception:
        rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan"), int(len(x))


# ============================================================================= configurations
def group_of(name):
    """ablation type of a configuration (by name; '@seed<n>' repeats belong to 'seed')."""
    base = name.split("@seed")[0]
    if "@seed" in name or name == "baseline":
        return "seed"
    if base.startswith("alpha_"):
        return "alpha"
    if base.startswith("query_"):
        return "query"
    if re.fullmatch(r"k\d+", base):
        return "depth"
    if base.startswith("fmt_"):
        return "format"
    if base.startswith("emb_"):
        return "embedding"
    if base == "temporal":
        return "temporal"
    if base in ("mem_off", "original"):
        return "memory"
    return "other"


GROUPS = [  # (key, title, what the baseline row means)
    ("memory", "1. MEMORY / SOURCE OF THE RETRIEVED CONTEXT", "rebuilt hybrid retrieval"),
    ("alpha", "2. HYBRID WEIGHT alpha (1.0 = dense only, 0.0 = BM25 only)", None),
    ("query", "3. QUERY DESIGN", None),
    ("depth", "4. RETRIEVAL DEPTH k (papers per section query; x3 sections)", None),
    ("format", "5. MEMORY FORMAT (what of each retrieved paper is shown; same token budget)", None),
    ("embedding", "6. EMBEDDING MODEL", None),
    ("temporal", "7. TEMPORAL FILTER (only papers published before the target paper)", "no filter"),
    ("seed", "8. SEED VARIANCE (same baseline, different generation seed) = noise floor", None),
    ("other", "9. OTHER CONFIGURATIONS", None),
]


def setting_label(cfg):
    """value of the factor that a configuration changes, e.g. alpha=0.7, k=5 (9 papers->15), fmt=summary6."""
    n = cfg.get("name", "")
    g = group_of(n)
    if g == "alpha" or n == "baseline_alpha":
        return f"alpha={cfg.get('alpha')}"
    if g == "query":
        return f"query={cfg.get('query')}"
    if g == "depth":
        return f"k={cfg.get('k')} ({cfg.get('n_papers')} papers)"
    if g == "format":
        return f"fmt={cfg.get('fmt')}"
    if g == "embedding":
        return f"emb={cfg.get('embed')}"
    if n == "temporal":
        return "before target year"
    if n == "mem_off":
        return "no retrieved context"
    if n == "original":
        return "retrieval columns of the input CSV"
    if "@seed" in n or n == "baseline":
        return f"seed={cfg.get('seed', 0)}"
    return ""


def baseline_label(cfg, group):
    return {"alpha": f"alpha={cfg.get('alpha')}", "query": f"query={cfg.get('query')}",
            "depth": f"k={cfg.get('k')} ({cfg.get('n_papers')} papers)", "format": f"fmt={cfg.get('fmt')}",
            "embedding": f"emb={cfg.get('embed')}", "temporal": "no filter", "memory": "rebuilt hybrid retrieval",
            "seed": f"seed={cfg.get('seed', 0)}"}.get(group, "")


def pick_columns(df, system, mode):
    cols = []
    if system in df.columns:
        cols.append(system)
    if mode in ("agents", "all"):
        cols += [c for c in df.columns if c.startswith(system + "__") and c.endswith("_Agent")]
    if mode == "all":
        cols += [c for c in df.columns if c in (f"{system}__lim_report", f"{system}__nov_report")]
    return cols


def discover(work, configs, systems, columns_mode):
    """-> dict cfg -> {"cfg": json, "input": path, "files": {system: (path, [columns])}}"""
    root = Path(work) / "configs"
    if not root.is_dir():
        raise SystemExit(f"no configs folder in {work}")
    want_cfg = None if configs in ("all", "") else set(configs.replace(",", " ").split())
    want_sys = None if systems in ("all", "") else set(systems.replace(",", " ").split())
    explicit = None if columns_mode in ("final", "agents", "all") else [c.strip() for c in re.split(r"[,+ ]", columns_mode) if c.strip()]
    out = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or (want_cfg is not None and d.name not in want_cfg):
            continue
        cj = d / "config.json"
        cfg = json.loads(cj.read_text()) if cj.exists() else {"name": d.name}
        cfg.setdefault("name", d.name)
        files = {}
        gdir = d / "out" / "generations"
        for f in sorted(gdir.glob("*.csv")) if gdir.is_dir() else []:
            sysname = f.stem
            if want_sys is not None and sysname not in want_sys:
                continue
            head = pd.read_csv(f, nrows=0).columns
            if explicit is not None:
                cols = [c for c in explicit if c in head]
            else:
                cols = pick_columns(pd.DataFrame(columns=head), sysname, columns_mode)
            if cols:
                files[sysname] = (f, cols)
        if files:
            out[d.name] = {"cfg": cfg, "input": d / "input.csv", "files": files}
        elif want_cfg is not None:
            warn(f"config {d.name}: no generation files with the requested columns in {gdir}")
    if want_cfg:
        miss = sorted(want_cfg - set(out))
        if miss:
            warn(f"requested configs without generations (skipped): {miss}")
    if not out:
        raise SystemExit(f"nothing to evaluate in {root} (configs={configs}, systems={systems}, columns={columns_mode})")
    return out


def cfg_order(name):
    order = ["baseline", "mem_off", "original", "alpha_1.0", "alpha_0.7", "alpha_0.5", "alpha_0.3", "alpha_0.0",
             "query_whole", "query_abstract", "k1", "k5", "fmt_abstract", "fmt_chunks", "fmt_summary6", "temporal"]
    if name in order:
        return (0, order.index(name), name)
    if "@seed" in name:
        return (2, 0, name)
    return (1, 0, name)


def col_order(c):
    s = c.split("__")[0]
    return (SYSTEM_ORDER.index(s) if s in SYSTEM_ORDER else 9, "__" in c, c)


# ============================================================================= retrieval quality
def compute_ir(work, D, row_ids, ks, rel_source, pdf_col, out_dir, tag=""):
    try:
        from rag_ablation.ir_eval import TitleMatcher, ref_titles, metrics as ir_metrics
        from rag_ablation.corpus import coerce_text, parse_obj
    except Exception as e:
        warn(f"cannot import rag_ablation.ir_eval ({e!r}) -> no retrieval metrics")
        return pd.DataFrame()
    pk = Path(work) / "corpus_docs.pkl"
    if not pk.exists():
        warn(f"{pk} not found -> no retrieval metrics")
        return pd.DataFrame()
    with open(pk, "rb") as f:
        cd = pickle.load(f)
    TM = TitleMatcher(cd["title_index"])
    rows, rel_cache = [], {}
    for name, info in D.items():
        cfg = info["cfg"]
        if not info["input"].exists():
            continue
        df = pd.read_csv(info["input"], low_memory=False)
        rid_col = "row_id" if "row_id" in df.columns else None
        for i, r in df.iterrows():
            rid = str(r[rid_col]) if rid_col else str(i)
            if rid not in row_ids:
                continue
            ctx = r.get("rag_abl__context_chars", np.nan)
            base = {"config": name, "row_id": rid, "context_chars": ctx, "n_retrieved": r.get("rag_abl__n_retrieved", np.nan),
                    "target_year": r.get("rag_abl__target_year", np.nan)}
            if cfg.get("original") or not cfg.get("memory", True) or "rag_abl__retrieved" not in df.columns:
                rows.append({**base, "ir_defined": False})
                continue
            if rid not in rel_cache:
                titles = ref_titles(r, rel_source, pdf_col)
                tp = parse_obj(r.get(pdf_col)) if pdf_col in r else None
                self_ids = TM.match(coerce_text(tp.get("title", ""))) if isinstance(tp, dict) else set()
                relevant = set().union(*[TM.match(t) for t in titles]) - self_ids if titles else set()
                rel_cache[rid] = (len(titles), relevant)
            n_refs, relevant = rel_cache[rid]
            try:
                ranked = [m["doc_id"] for m in json.loads(r["rag_abl__retrieved"])]
            except Exception:
                ranked = []
            m = ir_metrics(ranked, relevant, ks)
            hits = len(set(ranked) & relevant)
            m["R@all"] = hits / len(relevant) if relevant else np.nan
            m["P@all"] = hits / len(ranked) if ranked else (0.0 if relevant else np.nan)
            rows.append({**base, "ir_defined": True, "n_references": n_refs, "n_relevant_in_corpus": len(relevant), **m})
    P = pd.DataFrame(rows)
    P.to_csv(out_dir / f"ir_per_paper{tag}.csv", index=False)
    S = []
    if not P.empty:
        mcols = [c for c in P.columns if "@" in c or c == "MRR"]
        for name, d in P.groupby("config"):
            s = {"config": name, "n_papers": len(d), "context_chars": float(pd.to_numeric(d.context_chars, errors="coerce").mean()),
                 "n_retrieved": float(pd.to_numeric(d.n_retrieved, errors="coerce").mean())}
            if d.ir_defined.any():
                dd = d[d.n_relevant_in_corpus > 0]
                s["papers_with_relevant_in_corpus"] = len(dd)
                for c in mcols:
                    src = d[d.ir_defined] if c.startswith("P@") else dd
                    v = pd.to_numeric(src[c], errors="coerce").dropna()
                    s[c] = float(v.mean()) if len(v) else np.nan
                    s[c + "_lo"], s[c + "_hi"] = boot_ci(v.values)
            S.append(s)
    S = pd.DataFrame(S)
    S.to_csv(out_dir / f"ir_summary{tag}.csv", index=False)
    if not P.empty and "n_relevant_in_corpus" in P:
        cov = (P[P.ir_defined].drop_duplicates("row_id").n_relevant_in_corpus > 0).mean()
        log(f"[ir] {cov:.0%} of the evaluated papers cite at least one paper that is in the RAG corpus "
            f"(retrieval metrics use only those papers)")
    return P


# ============================================================================= merge repair (your merge prompt, verbatim)
MERGE_PROMPT_VERSION = "merge_a_anchor_10_14_v1"
MERGE_SYSTEM_PROMPT = """\
You are an expert scientific peer-review analyst. You receive two limitation \
reports written about the SAME research paper:
- Report A comes from the limitation agents (primary source).
- Report B comes from the novelty agents (secondary source).

Your job:
1. Produce a CONSOLIDATED list of between 10 and 14 limitations total \
2. Report A is the anchor. Use its limitations as the backbone of the final \
list, preserving their wording as much as possible (light edits for clarity \
are allowed).
3. Merge similar limitations:
   a. WITHIN Report A: if two or more Report A limitations cover the same \
underlying issue (e.g., several points about evaluation, several about \
dataset, several about scalability), merge them into a single richer \
limitation that retains all unique details.
   b. ACROSS reports: for each limitation in Report B, if it is similar, \
related, or overlapping with a limitation already in the consolidated list \
— even if worded differently — MERGE its unique details into that \
limitation, enriching it without losing information.
   c. If a Report B limitation has NO counterpart in the consolidated list, \
KEEP it as a separate, standalone limitation point (lightly edited for \
clarity).
4. Preservation rule: do not drop any substantive content from either report \
and do not invent new limitations. Every unique issue raised in A or B must \
appear somewhere in the final list, either as its own point or merged into a \
related point.
5. If after merging you have more than 14 points, continue grouping the most \
closely related ones until you are within the 10-14 range. If you have fewer \
than 10, split any overly broad merged point back into its distinct \
sub-issues until you reach at least 10.
6. Format each limitation as:
   **Short Descriptive Title**: 1-3 sentence explanation.
7. Output ONLY the list of limitations. No preamble, no commentary, no \
conclusion, no category headings, no numbering.
"""


def get_merge_user_prompt(lim_text, nov_text):
    return f"""\
### Report A — Limitation Agents (primary, must be kept):
{lim_text}

### Report B — Novelty Agents (merge into A when similar, otherwise keep separate):
{nov_text}

Produce the consolidated limitation report following the instructions provided.
"""


def usable(t):
    """a real report: non-empty string, not an ERROR, not SKIPPED_SHORT_TEXT."""
    return isinstance(t, str) and bool(t.strip()) and t.strip().lower() not in ("nan", "none") \
        and not is_error(t) and "SKIPPED_SHORT_TEXT" not in t


class Merger:
    def __init__(self, model, cache_path, backend="openai"):
        self.model, self.backend, self.cache_path, self.cache = model, backend, cache_path, {}
        self.lock = threading.Lock()
        if os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    r = json.loads(line)
                    self.cache[r["k"]] = r["out"]
                except Exception:
                    continue
        self.client = None
        if backend == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY is not set (export it in the PBS file or keep it in ~/.openai_key)")
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=180)

    def key(self, a, b):
        return hashlib.sha1(f"{MERGE_PROMPT_VERSION}|{self.model}|{a}\x00{b}".encode()).hexdigest()

    def _call(self, a, b):
        if self.backend == "mock":
            la = [x for x in a.splitlines() if len(x.split()) >= 3][:10]
            lb = [x for x in b.splitlines() if len(x.split()) >= 3][:4]
            return "\n".join(f"**Point {i + 1}**: {re.sub(r'^[-*• ]+', '', x)}" for i, x in enumerate(la + lb))
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.2, max_tokens=4096,
            messages=[{"role": "system", "content": MERGE_SYSTEM_PROMPT},
                      {"role": "user", "content": get_merge_user_prompt(a, b)}])
        return (r.choices[0].message.content or "").strip()

    def merge(self, a, b, tries=6):
        k = self.key(a, b)
        if k in self.cache:
            return self.cache[k], True
        last = ""
        for t in range(tries):
            try:
                out = self._call(a, b)
                if not out:
                    raise RuntimeError("empty answer")
                with self.lock, open(self.cache_path, "a") as fh:
                    self.cache[k] = out
                    fh.write(json.dumps({"k": k, "out": out, "m": self.model}) + "\n")
                return out, False
            except Exception as e:
                code = getattr(e, "status_code", None)
                last = f"{type(e).__name__}: {str(e)[:200]}"
                if code in (401, 403) or "invalid_api_key" in str(e) or "insufficient_quota" in str(e):
                    raise FatalAPIError(last)
                time.sleep(min(60, 2 ** (t + 1)) + random.random())
        return f"ERROR: merge failed after {tries} tries ({last})", False


def repair_merge(args, work, out_dir, dry=False):
    root = Path(work) / "configs"
    want = None if args.configs in ("all", "") else set(args.configs.replace(",", " ").split())
    merger = None if dry else Merger(args.merge_model, str(out_dir / "merge_cache.jsonl"), args.backend)
    rows, tot_calls, tot_chars, example = [], 0, 0, None
    for d in sorted(root.iterdir(), key=lambda x: cfg_order(x.name)):
        if not d.is_dir() or (want is not None and d.name not in want):
            continue
        g = d / "out" / "generations"
        mf = g / "merge.csv"
        if not mf.exists():
            continue
        M = pd.read_csv(mf, low_memory=False)
        if "merge" not in M.columns:
            warn(f"{mf}: no 'merge' column")
            continue
        rid = M["row_id"].astype(str) if "row_id" in M.columns else M.index.astype(str)
        side = {}
        for sysname in ("limagents", "novagents"):
            f = g / f"{sysname}.csv"
            if f.exists():
                X = pd.read_csv(f, low_memory=False)
                if sysname in X.columns:
                    xid = X["row_id"].astype(str) if "row_id" in X.columns else X.index.astype(str)
                    side[sysname] = dict(zip(xid, X[sysname]))
        bad = [i for i in M.index if not usable(M.at[i, "merge"])]
        todo, missing = [], Counter()
        for i in bad:
            a = M.at[i, "merge__lim_report"] if "merge__lim_report" in M.columns else None
            b = M.at[i, "merge__nov_report"] if "merge__nov_report" in M.columns else None
            a = a if usable(a) else side.get("limagents", {}).get(rid[i])
            b = b if usable(b) else side.get("novagents", {}).get(rid[i])
            if not usable(a):
                missing["limagents report missing/ERROR"] += 1
            elif not usable(b):
                missing["novagents report missing/ERROR"] += 1
            else:
                todo.append((i, a, b))
        chars = sum(len(MERGE_SYSTEM_PROMPT) + len(get_merge_user_prompt(a, b)) for _, a, b in todo)
        tot_calls += len(todo)
        tot_chars += chars
        if todo and example is None:
            example = (d.name, rid[todo[0][0]], todo[0][1], todo[0][2], M.at[todo[0][0], "merge"])
        rows.append({"config": d.name, "rows": len(M), "merge_ok": len(M) - len(bad), "merge_error": len(bad),
                     "repairable": len(todo), "not_repairable": dict(missing)})
        if dry or not todo:
            continue
        log(f"[merge] {d.name}: repairing {len(todo)} of {len(M)} rows with {args.merge_model}")
        bk = d / "out" / "generations_backup"
        bk.mkdir(exist_ok=True)
        if not (bk / "merge.csv").exists():
            M.to_csv(bk / "merge.csv", index=False)
            log(f"[merge] original saved -> {bk / 'merge.csv'}")
        for c in ("merge__lim_report", "merge__nov_report", "merge__repaired", "merge__merger"):
            if c not in M.columns:
                M[c] = pd.Series(dtype="object")
            M[c] = M[c].astype("object")
        M["merge"] = M["merge"].astype("object")
        done, fails = 0, 0
        try:
            with ThreadPoolExecutor(max_workers=args.merge_workers) as ex:
                futs = {ex.submit(merger.merge, a, b): (i, a, b) for i, a, b in todo}
                for f in as_completed(futs):
                    i, a, b = futs[f]
                    out, _ = f.result()
                    M.at[i, "merge"] = out
                    M.at[i, "merge__lim_report"], M.at[i, "merge__nov_report"] = a, b
                    M.at[i, "merge__repaired"] = not is_error(out)
                    M.at[i, "merge__merger"] = f"{args.merge_model} ({MERGE_PROMPT_VERSION})"
                    done += 1
                    fails += is_error(out)
                    if done % 25 == 0 or done == len(todo):
                        M.to_csv(str(mf) + ".tmp", index=False)
                        os.replace(str(mf) + ".tmp", mf)
                        log(f"[merge] {d.name}: {done}/{len(todo)} (failed {fails})")
        except FatalAPIError as e:
            M.to_csv(str(mf) + ".tmp", index=False)
            os.replace(str(mf) + ".tmp", mf)
            raise SystemExit(f"OpenAI API refused the key / quota: {e}")
        rows[-1]["repaired"] = done - fails
        rows[-1]["failed"] = fails
    R = pd.DataFrame(rows)
    L = ["=" * 110, f" MERGE {'DRY RUN' if dry else 'REPAIR'} ({args.merge_model}, your prompt: Report A = limagents anchor, "
         f"Report B = novagents, 10-14 points)   {time.strftime('%Y-%m-%d %H:%M')}", "=" * 110]
    L.append(R.to_string(index=False) if len(R) else "   no merge.csv found")
    tok = tot_chars / 4.0
    L.append(f"\n total: {tot_calls:,} merge calls, ~{tok / 1e6:.2f}M input tokens -> roughly "
             f"${tok / 1e6 * 0.15 + tot_calls * 900 / 1e6 * 0.6:.2f} with gpt-4o-mini (~900 output tokens each; check prices)"
             + ("" if dry else f"; cached calls are free ({len(merger.cache):,} in merge_cache.jsonl)"))
    if dry and example:
        n, r, a, b, old = example
        L += [f"\n EXAMPLE ({n}, row_id {r})", f"   current merge : {str(old)[:200]!r}",
              f"   Report A (limagents, {len(a):,} chars): {a[:400]!r} ...",
              f"   Report B (novagents, {len(b):,} chars): {b[:400]!r} ..."]
    L.append("=" * 110)
    txt = "\n".join(L)
    print(txt)
    (out_dir / f"merge_{'dry_run' if dry else 'repair'}.txt").write_text(txt + "\n")


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="dry", choices=["merge_dry", "merge", "dry", "test", "run"])
    ap.add_argument("--tag", default="", help="suffix of every output file, e.g. _agents")
    ap.add_argument("--merge_model", default="gpt-4o-mini")
    ap.add_argument("--merge_workers", type=int, default=8)
    ap.add_argument("--work_dir", default=str(RAG_DIR / "outputs" / "qwen3_8b"), help="RAG-ablation output folder of one model")
    ap.add_argument("--out_dir", default=None, help="default: <eval folder>/<basename of work_dir>")
    ap.add_argument("--configs", default="all", help="'all' or a list, e.g. 'baseline mem_off k1 k5'")
    ap.add_argument("--systems", default="all", help="'all' or a list of generation files, e.g. 'novagents merge'")
    ap.add_argument("--columns", default="final", help="final | agents | all | explicit list joined with ','")
    ap.add_argument("--reference", default="baseline", help="configuration every other one is compared with")
    ap.add_argument("--gt_col", default="ground_truth_lim_peer")
    ap.add_argument("--n_rows", type=int, default=None, help="papers per configuration; -1 = all (default: test 1, dry/run all)")
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--min_words", type=int, default=6,
                    help="generated units with fewer words are dropped (6 = drop units of 5 words or less)")
    ap.add_argument("--gt_min_words", type=int, default=3, help="the same for ground-truth units")
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--bertscore_model", default=None, help="local folder of roberta-large (cluster is offline)")
    ap.add_argument("--bertscore_layers", type=int, default=17, help="layer for a local model (17 = roberta-large default)")
    ap.add_argument("--ks", default="1 3 5 9 15")
    ap.add_argument("--rel_source", default="both", choices=["references", "cited_in", "both"])
    ap.add_argument("--pdf_col", default="pdf_text")
    ap.add_argument("--ir_metric", default="nDCG@9", help="retrieval metric used for the retrieval -> F1 link")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else HERE / work.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, out_dir / "output.txt")
    sys.stderr = Tee(sys.__stderr__, out_dir / "error.txt")
    print("\n" + "=" * 100)
    log(f"eval_rag_ablation.py mode={args.mode} argv={' '.join(sys.argv[1:])}")
    print(f"=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} mode={args.mode} ===", file=sys.stderr, flush=True)
    if args.mode in ("merge", "merge_dry"):
        repair_merge(args, work, out_dir, dry=args.mode == "merge_dry")
        return
    n_rows = args.n_rows if args.n_rows is not None else {"dry": -1, "test": 1, "run": -1}[args.mode]
    T = args.tag

    D = discover(work, args.configs, args.systems, args.columns)
    names = sorted(D, key=cfg_order)
    ref = args.reference if args.reference in D else None
    if not ref:
        warn(f"reference configuration '{args.reference}' has no generations -> no deltas / p-values")
    if args.mode == "test":
        names = [ref or names[0]]

    # ---------------------------------------------------------------- load generations, pick the same papers everywhere
    gens, order_ids = {}, None
    for n in names:
        for s, (f, cols) in D[n]["files"].items():
            df = pd.read_csv(f, low_memory=False)
            if args.gt_col not in df.columns:
                raise SystemExit(f"{f}: ground-truth column '{args.gt_col}' missing")
            df["_rid"] = df["row_id"].astype(str) if "row_id" in df.columns else df.index.astype(str)
            df = df.drop_duplicates("_rid").set_index("_rid", drop=False)
            gens[(n, s)] = (df, cols)
    src = D[ref]["input"] if ref and D[ref]["input"].exists() else None
    if src is not None:
        h = pd.read_csv(src, usecols=lambda c: c == "row_id", low_memory=False)
        order_ids = h["row_id"].astype(str).tolist() if "row_id" in h.columns else None
    if order_ids is None:
        first = gens[next(iter(gens))][0]
        order_ids = first["_rid"].tolist()
    present = set().union(*[set(df.index) for df, _ in gens.values()])
    order_ids = [r for r in order_ids if r in present]
    end = None if n_rows < 0 else args.start_row + n_rows
    sel = order_ids[args.start_row:end]
    sel_set = set(sel)
    log(f"{len(names)} configurations: {names}")
    log(f"papers: {len(sel):,} (positions {args.start_row}..{args.start_row + len(sel) - 1} of the reference input order)")
    gt_text = {}
    for (n, s), (df, cols) in gens.items():
        for rid in df.index:
            if rid in sel_set and rid not in gt_text:
                gt_text[rid] = df.at[rid, args.gt_col]
    gt_units = {rid: split_units(gt_text.get(rid), ("\n",), args.gt_min_words)[0] for rid in sel}

    jobs = []           # (cfg, system, column)
    units = {}          # (cfg, column, rid) -> (units, dropped) or None (= generation error / missing)
    for (n, s), (df, cols) in sorted(gens.items(), key=lambda kv: (cfg_order(kv[0][0]), col_order(kv[0][1]))):
        for c in sorted(cols, key=col_order):
            jobs.append((n, s, c))
            for rid in sel:
                if rid not in df.index:
                    units[(n, c, rid)] = None
                    continue
                t = df.at[rid, c]
                units[(n, c, rid)] = None if is_error(t) else split_units(t, ("\n",), args.min_words)
    if args.mode == "test":
        jobs = [j for j in jobs if j[2] == j[1]][:1] or jobs[:1]
    cols_all = sorted({c for _, _, c in jobs}, key=col_order)
    log(f"columns: {cols_all}")
    log(f"{len(jobs)} (configuration, column) jobs")

    if args.mode == "dry":
        dry_run(args, out_dir, names, jobs, units, gt_units, gt_text, sel)
        return

    judge = Judge(args.judge_model, str(out_dir / "judge_cache.jsonl"), args.backend)
    pairs = [(g, u) for n, s, c in jobs for rid in sel if units.get((n, c, rid)) for g in gt_units[rid]
             for u in units[(n, c, rid)][0]]
    log(f"{len(pairs):,} (GT unit, limitation unit) pairs, {len(set(pairs)):,} unique")
    n_err = judge.run(pairs, args.workers)

    rouge = RougeL()
    bs = BertScore(str(out_dir / "bertscore_cache.jsonl"), args.bertscore_model, args.bertscore_layers,
                   enabled=not args.no_bertscore)
    res, yes_all = {}, []
    for n, s, c in jobs:
        for rid in sel:
            uu = units.get((n, c, rid))
            if uu is None:
                res[(n, c, rid)] = None
                continue
            G, L = gt_units[rid], uu[0]
            yes, unk = [], 0
            for gi, g in enumerate(G):
                for li, u in enumerate(L):
                    a = judge.answer(g, u)
                    if a is None:
                        unk += 1
                    elif is_yes(a):
                        yes.append((gi, li))
            yg, yl = {g for g, _ in yes}, {l for _, l in yes}
            r = len(yg) / len(G) if G else np.nan
            p = len(yl) / len(L) if L else 0.0
            f = (2 * p * r / (p + r) if p + r else 0.0) if G else np.nan
            res[(n, c, rid)] = {"n_gt": len(G), "n_llm": len(L), "precision": p if G else np.nan, "recall": r, "f1": f,
                                "n_yes_pairs": len(yes), "judge_errors": unk, "yes": yes, "valid": bool(G and L)}
            yes_all += [(G[g], L[l]) for g, l in yes]
    bs.compute(yes_all)
    for key, r in res.items():
        if r is None:
            continue
        n, c, rid = key
        G, L = gt_units[rid], units[key][0]
        yp = [(G[g], L[l]) for g, l in r["yes"]]
        if yp:
            r["avg_cosine_sim"] = float(np.mean([tfidf_cosine(g, u) for g, u in yp]))
            r["avg_jaccard_sim"] = float(np.mean([jaccard(g, u) for g, u in yp]))
            r["avg_rougeL"] = float(np.mean([rouge(g, u) for g, u in yp]))
            b = [bs.get(g, u) for g, u in yp]
            r["avg_bertscore"] = float(np.mean(b)) if not any(math.isnan(x) for x in b) else float("nan")
        else:
            r["avg_cosine_sim"] = r["avg_jaccard_sim"] = r["avg_rougeL"] = 0.0
            r["avg_bertscore"] = 0.0 if bs.enabled else float("nan")

    if args.mode == "test":
        for n, s, c in jobs:
            for rid in sel:
                r = res[(n, c, rid)]
                if r is None:
                    print(f"\n### paper {rid} column {c}: generation error / missing")
                    continue
                G, L = gt_units[rid], units[(n, c, rid)][0]
                print(f"\n### config {n}  paper {rid}  column {c}\n    {len(G)} GT units x {len(L)} limitation units = {len(G) * len(L)} pairs")
                for gi, g in enumerate(G):
                    print(f"    GT{gi}: {g}")
                for li, u in enumerate(L):
                    print(f"    L{li}: {u}")
                print("    --- judge answers (rows = GT units, Y = Yes) ---")
                print("          " + " ".join(f"L{li:<2}" for li in range(len(L))))
                ys = set(map(tuple, r["yes"]))
                for gi in range(len(G)):
                    print(f"    GT{gi:<3}  " + " ".join(f"{'Y' if (gi, li) in ys else '.':<3}" for li in range(len(L))))
                print(f"    precision {r['precision']:.3f}  recall {r['recall']:.3f}  F1 {r['f1']:.3f}  yes pairs {r['n_yes_pairs']}"
                      f"  judge errors {r['judge_errors']}")
                print(f"    over the Yes pairs: cosine {r['avg_cosine_sim']:.3f}  jaccard {r['avg_jaccard_sim']:.3f}  "
                      f"ROUGE-L {r['avg_rougeL']:.3f}  BERTScore {r['avg_bertscore']:.3f}")
        log("test done (no tables written in test mode)")
        return

    # ---------------------------------------------------------------- per-file eval csv (your df_eval format)
    keys = ["n_gt", "n_llm", "precision", "recall", "f1", "n_yes_pairs", "avg_cosine_sim", "avg_jaccard_sim", "avg_rougeL",
            "avg_bertscore", "judge_errors"]
    for (n, s), (df, cols) in gens.items():
        sub = df.loc[[r for r in sel if r in df.index]].copy()
        sub["gt_units"] = [json.dumps(gt_units[r]) for r in sub.index]
        new = {}
        for c in cols:
            if (n, s, c) not in jobs:
                continue
            new[f"{c}__units"] = [json.dumps(units[(n, c, r)][0]) if units[(n, c, r)] else "" for r in sub.index]
            new[f"{c}__yes_pairs"] = [json.dumps(res[(n, c, r)]["yes"]) if res[(n, c, r)] else "" for r in sub.index]
            new[f"{c}__generation_error"] = [res[(n, c, r)] is None for r in sub.index]
            for k in keys:
                new[f"{c}__{k}"] = [res[(n, c, r)][k] if res[(n, c, r)] else np.nan for r in sub.index]
        sub = pd.concat([sub.drop(columns=["_rid"]), pd.DataFrame(new, index=sub.index)], axis=1)
        p = out_dir / "configs_eval" / n
        p.mkdir(parents=True, exist_ok=True)
        sub.to_csv(p / f"{s}_eval{T}.csv", index=False)
    log(f"per-file evaluations -> {out_dir / 'configs_eval'}/<config>/<system>_eval.csv")

    # ---------------------------------------------------------------- per paper table
    PP = []
    for n, s, c in jobs:
        for rid in sel:
            r = res[(n, c, rid)]
            base = {"config": n, "group": group_of(n), "system": s, "column": c, "row_id": rid}
            if r is None:
                PP.append({**base, "generation_error": True})
            else:
                PP.append({**base, "generation_error": False, **{k: r[k] for k in keys}, "valid": r["valid"]})
    PP = pd.DataFrame(PP)

    ks = [int(x) for x in args.ks.split()]
    IR = compute_ir(work, D, sel_set, ks, args.rel_source, args.pdf_col, out_dir, T)
    if not IR.empty:
        irc = [c for c in IR.columns if "@" in c or c in ("MRR", "n_relevant_in_corpus", "context_chars", "n_retrieved")]
        PP = PP.merge(IR[["config", "row_id"] + irc], on=["config", "row_id"], how="left")
    PP.to_csv(out_dir / f"per_paper{T}.csv", index=False)

    S = summarize(args, PP, IR, jobs, names, ref, D, out_dir)
    S.to_csv(out_dir / f"summary{T}.csv", index=False)
    link = ir_link(args, PP, S, ref)
    link.to_csv(out_dir / f"ir_f1_link{T}.csv", index=False)
    write_tables(args, out_dir, S, link, D, names, ref, len(sel), n_err, judge, rouge, bs, IR)
    if not args.no_plots:
        plots(out_dir, S, ref, args.ir_metric, T)


def summarize(args, PP, IR, jobs, names, ref, D, out_dir):
    ok = PP[~PP.generation_error.astype(bool)] if "generation_error" in PP else PP
    ok = ok[ok.n_gt.fillna(0) > 0]                              # papers without ground truth cannot be scored
    out = []
    irp = out_dir / f"ir_summary{args.tag}.csv"
    irsum = pd.read_csv(irp) if not IR.empty and irp.exists() else pd.DataFrame()
    for n, s, c in jobs:
        d = ok[(ok.config == n) & (ok.column == c)]
        allr = PP[(PP.config == n) & (PP.column == c)]
        row = {"config": n, "group": group_of(n), "setting": setting_label(D[n]["cfg"]), "system": s, "column": c,
               "n_papers": len(d), "n_error": int(allr.generation_error.sum()), "avg_n_gt": d.n_gt.mean(),
               "avg_n_units": d.n_llm.mean()}
        for k in ("precision", "recall", "f1", "n_yes_pairs", "avg_cosine_sim", "avg_jaccard_sim", "avg_rougeL", "avg_bertscore"):
            row[k] = float(d[k].mean()) if len(d) else np.nan
        row["f1_lo"], row["f1_hi"] = boot_ci(d.f1.values)
        row["judge_errors"] = int(d.judge_errors.sum())
        if ref and n != ref:
            b = ok[(ok.config == ref) & (ok.column == c)].set_index("row_id")
            a = d.set_index("row_id")
            common = a.index.intersection(b.index)
            row["n_paired"] = len(common)
            for k in ("precision", "recall", "f1"):
                x, y = a.loc[common, k].values, b.loc[common, k].values
                row[f"d_{k}"] = float(np.mean(x - y)) if len(common) else np.nan
                row[f"p_{k}"] = paired_p(x, y) if len(common) else np.nan
                dd = x - y
                row[f"d_{k}_lo"], row[f"d_{k}_hi"] = boot_ci(dd) if len(common) > 1 else (np.nan, np.nan)
        out.append(row)
    S = pd.DataFrame(out)
    if ref and "p_f1" in S:
        for k in ("precision", "recall", "f1"):
            S[f"p_{k}_holm"] = np.nan
            for c, d in S[S.config != ref].groupby("column"):
                ps = {i: p for i, p in d[f"p_{k}"].items() if not pd.isna(p)}
                for i, v in holm(ps).items():
                    S.at[i, f"p_{k}_holm"] = v
    if not irsum.empty:
        S = S.merge(irsum.drop(columns=["n_papers"], errors="ignore"), on="config", how="left")
    # seed noise: SD of F1 over baseline and its @seed repeats, per column
    seeds = S[S.config.map(lambda x: x == (ref or "baseline") or x.startswith((ref or "baseline") + "@seed"))]
    sd = seeds.groupby("column").f1.std() if len(seeds) else pd.Series(dtype=float)
    S["seed_sd_f1"] = S.column.map(sd)
    return S


def ir_link(args, PP, S, ref):
    """does better retrieval give better limitations? (configs with a defined retrieval metric only)"""
    m = args.ir_metric
    rows = []
    if m not in PP.columns:
        return pd.DataFrame(columns=["column", "level", "rho", "p", "n"])
    for c, d in PP[~PP.generation_error.astype(bool)].groupby("column"):
        d = d[d[m].notna() & d.f1.notna()]
        if d.empty:
            continue
        agg = S[(S.column == c) & S[m].notna()] if m in S else pd.DataFrame()
        if len(agg):
            rho, p, n = spearman(agg[m], agg.f1)
            rows.append({"column": c, "level": f"across configurations (mean {m} vs mean F1)", "rho": rho, "p": p, "n": n})
        dd = d.copy()
        dd["f1_c"] = dd.f1 - dd.groupby("row_id").f1.transform("mean")
        dd["ir_c"] = dd[m] - dd.groupby("row_id")[m].transform("mean")
        rho, p, n = spearman(dd.ir_c, dd.f1_c)
        rows.append({"column": c, "level": f"within paper (paper-centred {m} vs F1, all configs)", "rho": rho, "p": p, "n": n})
        if ref:
            b = d[d.config == ref]
            rho, p, n = spearman(b[m], b.f1)
            rows.append({"column": c, "level": f"across papers within {ref}", "rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows)


# ============================================================================= reporting
def f3(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x) else f"{x:.{nd}f}"


def table(headers, rows, left=1):
    if not rows:
        return "   (no rows)"
    w = [max(len(str(h)), *(len(str(r[k])) for r in rows)) for k, h in enumerate(headers)]
    line = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    o = [line, "| " + " | ".join(str(h).ljust(w[k]) for k, h in enumerate(headers)) + " |", line.replace("-", "=")]
    prev = None
    for r in rows:
        if prev is not None and r[0] != prev and left == 1:
            o.append(line)
        prev = r[0]
        o.append("| " + " | ".join(str(x).ljust(w[k]) if k < left + 1 else str(x).rjust(w[k]) for k, x in enumerate(r)) + " |")
    o.append(line)
    return "\n".join(o)


def delta(d, k, holm_col=True):
    v = d.get(f"d_{k}")
    if v is None or pd.isna(v):
        return "ref"
    p = d.get(f"p_{k}_holm") if holm_col and not pd.isna(d.get(f"p_{k}_holm", np.nan)) else d.get(f"p_{k}")
    star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    return f"{v:+.3f}{star}"


def ir_cells(d, metric):
    vals = []
    for m in ("R@9", "nDCG@9", "P@9", "MRR", "R@all"):
        vals.append(f3(d.get(m)))
    vals.append(f3(d.get("context_chars"), 0) if not pd.isna(d.get("context_chars", np.nan)) else "-")
    return vals


def write_tables(args, out_dir, S, link, D, names, ref, n_sel, n_err, judge, rouge, bs, IR):
    L = []
    W = L.append
    cols = sorted(S.column.unique(), key=col_order)
    finals = [c for c in cols if "__" not in c]
    W("=" * 150)
    W(f" RAG ABLATIONS - LIMITATION GENERATION ({Path(args.work_dir).resolve().name})    {time.strftime('%Y-%m-%d %H:%M')}")
    W("=" * 150)
    W(f" work dir        : {Path(args.work_dir).resolve()}")
    W(f" configurations  : {len(names)}  {names}")
    W(f" columns         : {cols}")
    W(f" papers          : {n_sel} per configuration (the same papers everywhere; paired comparisons use the papers both have)")
    W(f" ground truth    : {args.gt_col} (split on newline)")
    W(f" LLM judge       : {args.judge_model} ({args.backend}), every (GT unit, limitation unit) pair, your Yes/No prompt, T=0;"
      f" cache {len(judge.cache):,} pairs; failed calls this run {n_err} (counted as No; rerun to retry)")
    W("   P = share of generated units matched by >= 1 GT unit; R = share of GT units matched by >= 1 generated unit; F1 per paper,"
      " then averaged")
    W(f" NLP similarity  : over the Yes pairs (0 if none). ROUGE-L: {rouge.kind}; BERTScore: {bs.kind}")
    W(f" retrieval       : relevant = papers cited by the target paper that are in the RAG corpus ({args.rel_source});"
      " papers without such a citation are excluded from R/nDCG/MRR. '-' = no retrieval (mem_off) or not rebuilt (original)")
    W("   R@all = recall of everything the configuration retrieved (k1 retrieves 3 papers, baseline 9, k5 15); ctx = mean"
      " characters of retrieved context given to the generator")
    W(f" significance    : d = config - {ref} on the same papers; paired randomization test, Holm-corrected per column and"
      " metric over all configurations; * p<0.05, ** p<0.01")
    W("")

    # ---------- main wide table
    W("A. MAIN TABLE - F1 per system (rows = configuration) + retrieval quality")
    hdr = ["config", "setting"] + [f"F1 {c}" for c in finals] + ["R@9", "nDCG@9", "MRR", "ctx chars"]
    rows = []
    for n in names:
        d0 = S[S.config == n]
        if d0.empty:
            continue
        r0 = d0.iloc[0]
        cells = [n, r0.setting if n != ref else "reference"]
        for c in finals:
            x = d0[d0.column == c]
            cells.append(f"{f3(x.f1.iloc[0])} ({delta(x.iloc[0].to_dict(), 'f1')})" if len(x) else "")
        cells += [f3(r0.get("R@9")), f3(r0.get("nDCG@9")), f3(r0.get("MRR")),
                  f3(r0.get("context_chars"), 0) if not pd.isna(r0.get("context_chars", np.nan)) else "-"]
        rows.append(cells)
    W(table(hdr, rows, left=1).replace("\n| ", "\n| ", 1))
    W("   cell = F1 (difference vs the reference, Holm-corrected star)")

    # ---------- per ablation type
    W("\nB. ONE TABLE PER ABLATION TYPE (the reference row is repeated in every table)")
    for g, title, _ in GROUPS:
        members = [n for n in names if group_of(n) == g and n != ref]
        if not members:
            continue
        lst = ([ref] if ref else []) + members
        W(f"\n{title}")
        hdr = ["system", "config (setting)", "n", "#L", "P", "R", "F1", "F1 95% CI", "dP", "dR", "dF1", "p(F1,Holm)",
               "ROUGE-L", "BERTSc", "R@9", "nDCG@9", "P@9", "MRR", "R@all", "ctx"]
        rows = []
        for c in cols:
            for n in lst:
                x = S[(S.config == n) & (S.column == c)]
                if x.empty:
                    continue
                d = x.iloc[0].to_dict()
                lab = baseline_label(D[n]["cfg"], g) if n == ref else d["setting"]
                ph = d.get("p_f1_holm", np.nan)
                rows.append([c, f"{n} ({lab})" + (" [ref]" if n == ref else ""), d["n_papers"], f3(d["avg_n_units"], 1),
                             f3(d["precision"]), f3(d["recall"]), f3(d["f1"]), f"[{f3(d['f1_lo'])},{f3(d['f1_hi'])}]",
                             delta(d, "precision"), delta(d, "recall"), delta(d, "f1"),
                             "ref" if n == ref else (f"{ph:.3g}" if not pd.isna(ph) else "-"),
                             f3(d["avg_rougeL"]), f3(d["avg_bertscore"])] + ir_cells(d, args.ir_metric))
        W(table(hdr, rows))
        if g == "seed":
            sd = S[S.config.isin(lst)].groupby("column").f1.agg(["mean", "std", "count"])
            for c, r in sd.iterrows():
                W(f"   seed noise {c:<45} F1 mean {r['mean']:.3f}  SD {f3(r['std'])} over {int(r['count'])} seeds "
                  f"(differences smaller than ~2 SD are within generation noise)")

    # ---------- NLP table
    W("\nC. NLP SIMILARITY OF THE MATCHED (Yes) PAIRS + unit counts")
    hdr = ["column", "config", "papers", "errors", "#GT", "#L", "yes pairs", "cosine", "jaccard", "ROUGE-L", "BERTScore"]
    rows = [[d["column"], d["config"], d["n_papers"], d["n_error"], f3(d["avg_n_gt"], 1), f3(d["avg_n_units"], 1),
             f3(d["n_yes_pairs"], 1), f3(d["avg_cosine_sim"]), f3(d["avg_jaccard_sim"]), f3(d["avg_rougeL"]),
             f3(d["avg_bertscore"])] for _, d in S.sort_values(["column", "config"], key=lambda s: s.map(
                 lambda v: col_order(v) if s.name == "column" else cfg_order(v))).iterrows()]
    W(table(hdr, rows))

    # ---------- retrieval
    if not IR.empty and "R@9" in S:
        W("\nD. RETRIEVAL QUALITY PER CONFIGURATION (mean [95% CI] over papers with >= 1 cited paper in the corpus)")
        ks = [int(x) for x in args.ks.split()]
        hdr = ["config", "papers"] + [f"R@{k}" for k in ks] + ["nDCG@9", "P@9", "Hit@9", "MRR", "R@all", "#retrieved"]
        rows = []
        irs = pd.read_csv(out_dir / f"ir_summary{args.tag}.csv")
        for n in names:
            x = irs[irs.config == n]
            if x.empty:
                continue
            d = x.iloc[0].to_dict()
            if pd.isna(d.get("R@9", np.nan)):
                rows.append([n, "-"] + ["-"] * (len(ks) + 5) + [f3(d.get("n_retrieved"), 1)])
                continue
            ci = lambda m: f"{f3(d.get(m))} [{f3(d.get(m + '_lo'))},{f3(d.get(m + '_hi'))}]"  # noqa: E731
            rows.append([n, int(d.get("papers_with_relevant_in_corpus", 0))] + [f3(d.get(f"R@{k}")) for k in ks] +
                        [ci("nDCG@9"), f3(d.get("P@9")), f3(d.get("Hit@9")), f3(d.get("MRR")), f3(d.get("R@all")),
                         f3(d.get("n_retrieved"), 1)])
        W(table(hdr, rows))

    # ---------- link
    if len(link):
        W(f"\nE. DOES BETTER RETRIEVAL GIVE BETTER LIMITATIONS? (Spearman rho, {args.ir_metric} vs F1)")
        rows = [[d["column"], d["level"], f3(d["rho"]), f3(d["p"]), d["n"]] for _, d in link.iterrows()]
        W(table(["column", "level", "rho", "p", "n"], rows))
        W("   within paper = each paper's own scores centred on its mean, so paper difficulty does not drive the correlation")

    # ---------- ranking
    if ref:
        W(f"\nF. CONFIGURATIONS RANKED BY dF1 vs {ref} (final columns)")
        for c in finals:
            x = S[(S.column == c) & (S.config != ref)].sort_values("d_f1", ascending=False)
            W(f"   {c}:  " + "  ".join(f"{r.config} {r.d_f1:+.3f}{'*' if r.get('p_f1_holm', 1) < 0.05 else ''}"
                                         for _, r in x.iterrows() if not pd.isna(r.d_f1)))
    W("=" * 150)
    txt = "\n".join(L) + "\n"
    (out_dir / f"RAG_EVAL_TABLES{args.tag}.txt").write_text(txt)
    print(txt)
    write_latex(out_dir, S, names, ref, finals, cols, D, args.tag)
    log(f"wrote {out_dir / f'RAG_EVAL_TABLES{args.tag}.txt'} (+ tables / summary / per_paper / ir_* with suffix '{args.tag}')")


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("@", "@")


def write_latex(out_dir, S, names, ref, finals, cols, D, tag=""):
    o = ["% generated by eval_rag_ablation.py  (\\usepackage{booktabs})", ""]
    o += [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{ll" + "c" * len(finals) + "ccc}", r"\toprule",
          "Config & Setting & " + " & ".join(f"F1 {tex(c)}" for c in finals) + r" & R@9 & nDCG@9 & MRR \\", r"\midrule"]
    for n in names:
        d0 = S[S.config == n]
        if d0.empty:
            continue
        r0 = d0.iloc[0]
        cells = [tex(n), tex(r0.setting if n != ref else "reference")]
        for c in finals:
            x = d0[d0.column == c]
            if x.empty:
                cells.append("")
                continue
            d = x.iloc[0]
            star = "$^{*}$" if d.get("p_f1_holm", 1) < 0.05 and n != ref else ""
            dd = "" if n == ref or pd.isna(d.get("d_f1", np.nan)) else f" ({d.d_f1:+.3f})"
            cells.append(f"{d.f1:.3f}{dd}{star}")
        cells += [f3(r0.get("R@9")), f3(r0.get("nDCG@9")), f3(r0.get("MRR"))]
        o.append(" & ".join(cells) + r" \\")
    o += [r"\bottomrule", r"\end{tabular}",
          r"\caption{RAG ablations: limitation F1 (difference vs.\ baseline; $^{*}$ Holm-corrected $p<0.05$) and retrieval quality.}",
          r"\end{table*}", ""]
    for g, title, _ in GROUPS:
        members = [n for n in names if group_of(n) == g and n != ref]
        if not members:
            continue
        lst = ([ref] if ref else []) + members
        o += [f"% {title}", r"\begin{table}[t]\centering\small", r"\begin{tabular}{llccccc}", r"\toprule",
              r"System & Setting & P & R & F1 & $\Delta$F1 & nDCG@9 \\", r"\midrule"]
        for c in finals:
            for n in lst:
                x = S[(S.config == n) & (S.column == c)]
                if x.empty:
                    continue
                d = x.iloc[0]
                lab = baseline_label(D[n]["cfg"], g) + " (ref)" if n == ref else d.setting
                star = "$^{*}$" if n != ref and d.get("p_f1_holm", 1) < 0.05 else ""
                dd = "--" if n == ref or pd.isna(d.get("d_f1", np.nan)) else f"{d.d_f1:+.3f}{star}"
                o.append(f"{tex(c)} & {tex(lab)} & {d.precision:.3f} & {d.recall:.3f} & {d.f1:.3f} & {dd} & {f3(d.get('nDCG@9'))} \\\\")
            o.append(r"\midrule")
        o[-1] = r"\bottomrule"
        o += [r"\end{tabular}", f"\\caption{{{tex(title.split('. ', 1)[-1].title())}}}", r"\end{table}", ""]
    (out_dir / f"tables{tag}.tex").write_text("\n".join(o))


def plots(out_dir, S, ref, metric, tag=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        warn(f"no plots ({e!r})")
        return
    pd_ = out_dir / "plots"
    pd_.mkdir(exist_ok=True)
    finals = [c for c in sorted(S.column.unique(), key=col_order) if "__" not in c]
    cfgs = sorted(S.config.unique(), key=cfg_order)
    fig, ax = plt.subplots(figsize=(max(8, len(cfgs) * 0.7), 4.5))
    w = 0.8 / max(1, len(finals))
    for j, c in enumerate(finals):
        x = S[S.column == c].set_index("config").reindex(cfgs)
        err = np.vstack([x.f1 - x.f1_lo, x.f1_hi - x.f1]).clip(0)
        ax.bar(np.arange(len(cfgs)) + j * w, x.f1, w, yerr=err, capsize=2, label=c)
    ax.set_xticks(np.arange(len(cfgs)) + w * (len(finals) - 1) / 2)
    ax.set_xticklabels(cfgs, rotation=45, ha="right")
    ax.set_ylabel("F1 (judge)")
    ax.set_title("Limitation F1 per RAG configuration (95% CI)")
    ax.legend(frameon=False, ncol=len(finals))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(pd_ / f"f1_by_config{tag}.png", dpi=150)
    plt.close(fig)
    if metric in S and S[metric].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c in finals:
            x = S[(S.column == c) & S[metric].notna()]
            ax.scatter(x[metric], x.f1, label=c)
            for _, r in x.iterrows():
                ax.annotate(r.config, (r[metric], r.f1), fontsize=6, alpha=.7)
        ax.set_xlabel(f"retrieval {metric}")
        ax.set_ylabel("limitation F1")
        ax.set_title("Retrieval quality vs. limitation F1 (one point = config)")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(pd_ / f"ir_vs_f1{tag}.png", dpi=150)
        plt.close(fig)
    log(f"plots -> {pd_}")


def dry_run(args, out_dir, names, jobs, units, gt_units, gt_text, sel):
    L = []
    W = L.append
    W("=" * 110)
    W(f" DRY RUN - unit split + cost estimate   ({time.strftime('%Y-%m-%d %H:%M')})")
    W("=" * 110)
    for rid in sel[:2]:
        W(f"\n# PAPER {rid}\nGROUND TRUTH -> {len(gt_units[rid])} units")
        for k, g in enumerate(gt_units[rid]):
            W(f"   GT{k}: {g}")
        for n, s, c in jobs:
            if n != (args.reference if args.reference in names else names[0]):
                continue
            uu = units.get((n, c, rid))
            W(f"\n--- {n} / {c}: " + ("GENERATION ERROR / missing" if uu is None else f"{len(uu[0])} units"))
            if uu:
                for k, x in enumerate(uu[0]):
                    W(f"     L{k}: {x}")
                for d in uu[1]:
                    W(f"     (dropped) {d[:300]}")
    W(f"\n{'=' * 110}\nUNITS PER (config, column) over the selected {len(sel)} papers")
    W(f"   {'config':<18} {'column':<55} {'mean #L':>8} {'errors':>7} {'pairs':>9}")
    uniq, total = set(), 0
    for n, s, c in jobs:
        nl, err, pc = [], 0, 0
        for rid in sel:
            uu = units.get((n, c, rid))
            if uu is None:
                err += 1
                continue
            nl.append(len(uu[0]))
            pc += len(uu[0]) * len(gt_units[rid])
            uniq.update((g, x) for g in gt_units[rid] for x in uu[0])
        total += pc
        W(f"   {n:<18} {c:<55} {np.mean(nl) if nl else 0:>8.1f} {err:>7} {pc:>9,}")
    cached = set()
    cp = out_dir / "judge_cache.jsonl"
    if cp.exists():
        for line in open(cp):
            try:
                cached.add(json.loads(line)["k"])
            except Exception:
                pass
    jk = lambda g, x: hashlib.sha1(f"{PROMPT_VERSION}|{args.judge_model}|{g}\x00{x}".encode()).hexdigest()  # noqa: E731
    new = sum(1 for g, x in uniq if jk(g, x) not in cached)
    tok = sum(len(judge_prompt(g, x)) for g, x in uniq) / 4.0 * (new / max(1, len(uniq)))
    gl = [len(v) for v in gt_units.values()]
    W(f"\n GT units per paper: mean {np.mean(gl):.1f}, papers without GT: {sum(1 for x in gl if not x)}")
    W(f" ESTIMATE for these {len(sel)} papers x {len(jobs)} jobs: {total:,} pairs, {len(uniq):,} unique, "
      f"{len(uniq) - new:,} cached -> {new:,} API calls")
    W(f"   ~{tok / 1e6:.1f}M input tokens -> roughly ${tok / 1e6 * 0.15 + new * 2 / 1e6 * 0.6:.2f} with gpt-4o-mini "
      f"(check current prices); at ~25 calls/s about {new / 25 / 3600:.1f} h")
    W("   (split examples above: first 2 papers of the reference configuration; full split check: configs_eval after run)")
    W("=" * 110)
    (out_dir / f"dry_run_split{args.tag}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    log(f"-> {out_dir / f'dry_run_split{args.tag}.txt'}")


if __name__ == "__main__":
    main()