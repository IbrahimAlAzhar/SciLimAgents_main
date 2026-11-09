"""
Unified evaluation (every eval.py / eval_fp.py / merge_and_eval.py / multi-seed eval.py).

For each evaluated column:
  1. PARSE    generated report -> limitation units (--parser auto|v1|asterisk|legacy)
  2. PAIR     every GT limitation x every generated limitation
  3. JUDGE    LLM Yes/No per pair (gpt-4o-mini default, or a local served model)
  4. METRICS  precision / recall / F1 (unique-id based) + cosine / Jaccard /
              ROUGE-L / BERTScore averaged over YES pairs
  5. (opt) FP unmatched LUs classified into 6 categories (valid-missing,
              too-broad, partial, hallucinated, off-target, redundant); a pair
              counts as valid if judged Yes OR its LU is valid (cat 1/2) ->
              precision_fp / recall_fp / f1_fp + similarity on valid pairs.
Saves every --save_every rows (default 10) and resumes; writes summary.csv.

    python -m scilim.evaluation.evaluate --input_csv outputs/qwen3_8b/generations/limagents.csv \
        --columns limagents --out_dir outputs/qwen3_8b/eval
    python -m scilim.evaluation.evaluate --gen_dir outputs/qwen3_8b/generations --out_dir ...   # all ablations
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..llm.client import ChatClient, is_openai_model, parse_json_obj
from ..prompts.evaluation import CATEGORY_LABELS, FP_SYSTEM_PROMPT, VALID_CATEGORIES, fp_user_prompt, pair_eval_prompt
from ..utils.io import atomic_to_csv
from ..utils.text import is_missing

# =============================================================================
# Parsers
# =============================================================================


def parse_auto(text_str):
    """merge_and_eval.py / eval_50_rows.py parser (most robust; default)."""
    if not isinstance(text_str, str) or not text_str.strip():
        return []
    lims, cat, cur, lid = [], "General", "", 0

    def commit():
        nonlocal cur, lid
        if cur:
            lims.append({"llm_id": lid, "category": cat, "limitation": cur.strip()})
            lid += 1
            cur = ""
    for raw in text_str.splitlines():
        line = raw.strip(" \t\n\r\xa0'\"")
        if not line:
            continue
        if "consolidated list" in line.lower() or "limitations identified" in line.lower():
            continue
        if re.match(r'^(?:-|\*|\d+\.)?\s*[\*\_]*Evidence[\*\_]*\s*:', line, re.IGNORECASE):
            continue
        m = re.match(r'^\*\*(.*?)\*\*:?$', line)
        if m:
            commit()
            cat = m.group(1).strip()
            continue
        m = re.match(r'^(?:####|###)\s+(.*)', line)
        if m:
            commit()
            cat = m.group(1).strip('*:- ')
            continue
        m = re.match(r'^(?:-|\*|\d+\.)\s*[\*\_]+(.*?)[*\_]+\s*:\s*(.*)$', line)
        if m:
            commit()
            cat, cur = m.group(1).strip('*:- '), m.group(2).strip()
            continue
        m = re.match(r'^[\*\_]{2}(.*?)[\*\_]{2}\s*:\s*(.*)$', line)
        if m:
            commit()
            cat, cur = m.group(1).strip('*:-[] '), m.group(2).strip()
            continue
        m = re.match(r'^(?:-|\*|\d+\.)\s*[\*\_]*Limitation[\*\_]*\s*:\s*(.*)$', line, re.IGNORECASE)
        if m:
            commit()
            cur = m.group(1).strip()
            continue
        m = re.match(r'^(?:-|\*|\d+\.)\s+(.*)$', line)
        if m:
            commit()
            cur = m.group(1).strip()
            continue
        cur = (cur + " " + line) if cur else line
    commit()
    return [{"llm_id": x["llm_id"], "llm_limitation": f"[{x['category']}] {x['limitation']}"} for x in lims]


def parse_v1(text_str, min_words=6):
    """limagents/ablation/eval/eval.py parser + filter (>= 6 words)."""
    if not isinstance(text_str, str) or not text_str:
        return []
    out, cat, lid = [], "General", 0
    ignore = {"here is the output format you requested:", "here is the consolidated list of limitations:",
              "**category:** **limitation statement**", "**category:**", "**limitation statement**"}
    starters = ("please let me know", "let me know if", "i will provide a", "these limitations are",
                "to address these issues,", "please note that", "here is the", "here are the")
    for line in text_str.splitlines():
        line = line.strip(" \t\n\r\xa0")
        if not line:
            continue
        low = line.lower()
        if low in ignore or low.startswith(starters):
            continue
        if line.startswith('#'):
            h = line.lstrip('#').strip()
            if "final list" not in h.lower() and h:
                cat = h
            continue
        m = re.match(r'^\*\*(.*?)\*\*:?$', line)
        if m:
            if m.group(1).strip().lower() not in ["category", "limitation statement"]:
                cat = m.group(1).strip()
            continue
        m = re.match(r'^(?:\d+[\.\)]|-|\*)\s+(.*)', line)
        if m:
            content = m.group(1).strip()
            title = body = ""
            b = re.match(r'^\*\*(.*?)\*\*:?\s*(.*)', content)
            if b:
                title, body = b.group(1).strip(), b.group(2).strip()
            elif ':' in content:
                title, body = [x.strip() for x in content.split(':', 1)]
            else:
                body = content
            if not body and title:
                body, title = title, ""
            out.append(f"{title}: {body} (- **{cat}**)" if title else f"{body} (- **{cat}**)")
        elif len(line) > 15:
            out.append(f"{line} (- **{cat}**)")
    out = [x for x in out if len(x.split()) >= min_words]
    return [{"llm_id": i, "llm_limitation": t} for i, t in enumerate(out)]


def parse_legacy(text_str):
    """limagents+novagents/eval/eval.py parser (merged reports)."""
    if not isinstance(text_str, str) or not text_str:
        return []
    out, cat = [], "General"
    for line in text_str.splitlines():
        line = line.strip(" \t\n\r\xa0")
        if not line:
            continue
        if line.startswith('#'):
            h = line.lstrip('#').strip()
            if "Final List" not in h:
                cat = h
            continue
        m = re.match(r'^\*\*(.*?)\*\*:?$', line)
        if m:
            cat = m.group(1).strip()
            continue
        if '**' in line and line[0].isdigit():
            parts = line.split('**')
            if len(parts) >= 3:
                out.append(f"{parts[1].strip()}: {'**'.join(parts[2:]).lstrip(': ').strip()} (- **{cat}**)")
                continue
        elif line.startswith('-') or line.startswith('*'):
            body = line[1:].strip()
            if '**' in body:
                parts = body.split('**')
                out.append(f"{parts[1].strip()}: {'**'.join(parts[2:]).lstrip(': ').strip()} (- **{cat}**)"
                           if len(parts) >= 3 else f"{body} (- **{cat}**)")
            elif ':' in body:
                t, d = body.split(':', 1)
                out.append(f"{t.strip()}: {d.strip()} (- **{cat}**)")
            else:
                out.append(f"{body} (- **{cat}**)")
    return [{"llm_id": i, "llm_limitation": t} for i, t in enumerate(out)]


def parse_asterisk(text):
    """dpo_limagents/inference/eval.py: split on '- **' blocks, merge Evidence/
    Support/Explanation/Impact continuations, drop items of <= 10 words."""
    if not isinstance(text, str) or not text.strip():
        return []
    res = []
    for chunk in re.split(r'\n(?=\s*-\s\*\*)', text):
        chunk = chunk.strip()
        if not chunk.startswith('- **'):
            continue
        keep = []
        for line in chunk.split('\n'):
            if line.startswith('- **') or line.startswith(' ') or line.startswith('\t') or not line.strip():
                keep.append(line)
            else:
                break
        res.append('\n'.join(keep).strip())
    # (original order) 1) drop items of <= 10 words, 2) merge Evidence/Support/... continuations
    res = [t for t in res if len(t.split()) > 10]
    merged = []
    for t in res:
        chk = t.lstrip('- ').strip()
        if chk.startswith(('**Evidence', '**Support', '**Explanation', '**Impact')):
            if merged:
                merged[-1] += f"\n    - {chk}"
        else:
            merged.append(t)
    return [{"llm_id": i, "llm_limitation": t} for i, t in enumerate(merged)]


PARSERS = {"auto": parse_auto, "v1": parse_v1, "legacy": parse_legacy, "asterisk": parse_asterisk}


def parse_gt(text_str):
    if not isinstance(text_str, str) or not text_str.strip():
        return []
    out = []
    for line in text_str.strip().split("\n"):
        if line.strip():
            out.append({"gt_id": len(out), "gt_limitation": line.strip()})
    return out


# =============================================================================
# Metrics
# =============================================================================
def parse_eval_list(v):
    if isinstance(v, list):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return []
    for _ in range(2):
        try:
            v = ast.literal_eval(v) if isinstance(v, str) else v
        except Exception:
            return []
    return v if isinstance(v, list) else []


def _field(item, prefix):
    for e in item:
        if isinstance(e, str) and e.startswith(prefix):
            return e.split(prefix, 1)[1].strip()
    return None


def pair_valid(item, use_fp):
    yes = "YES" in str(item[0]).upper()
    if use_fp:
        fv = _field(item, "fp_valid:")
        yes = yes or (fv is not None and fv.strip().upper() == "YES")
    return yes


def compute_prf(items, use_fp=False):
    items = parse_eval_list(items)
    ag, al, yg, yl = set(), set(), set(), set()
    for it in items:
        if not isinstance(it, (list, tuple)) or len(it) < 2:
            continue
        try:
            g, l = int(_field(it, "gt_id:")), int(_field(it, "llm_id:"))
        except (TypeError, ValueError):
            continue
        ag.add(g)
        al.add(l)
        if pair_valid(it, use_fp):
            yg.add(g)
            yl.add(l)
    r = len(yg) / len(ag) if ag else 0.0
    p = len(yl) / len(al) if al else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"n_unique_gt": len(ag), "n_unique_llm": len(al), "recall": r, "precision": p, "f1": f}


_rouge = None
_bert = None
_bert_lock = threading.Lock()


def _rougeL(g, l):
    global _rouge
    try:
        if _rouge is None:
            from rouge_score import rouge_scorer
            _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(_rouge.score(g, l)["rougeL"].fmeasure)
    except Exception:
        return 0.0


def _cos(a, b):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        v = TfidfVectorizer().fit([a, b])
        t = v.transform([a, b])
        return float(cosine_similarity(t[0], t[1])[0, 0])
    except Exception:
        return 0.0


def _jac(a, b):
    t1, t2 = set(a.lower().split()), set(b.lower().split())
    return float(len(t1 & t2) / len(t1 | t2)) if (t1 | t2) else 0.0


def compute_similarity(items, use_bertscore=True, use_fp=False):
    items = parse_eval_list(items)
    gts, lls = [], []
    for it in items:
        if not isinstance(it, (list, tuple)) or len(it) < 2 or not pair_valid(it, use_fp):
            continue
        g, l = _field(it, "gt_limitation:"), _field(it, "llm_limitation:")
        if g and l:
            gts.append(g)
            lls.append(l)
    res = {"avg_cosine_sim": 0.0, "avg_jaccard_sim": 0.0, "avg_rougeL": 0.0, "avg_bertscore": 0.0,
           "n_yes_pairs": len(gts)}
    if not gts:
        return res
    res["avg_cosine_sim"] = float(np.mean([_cos(g, l) for g, l in zip(gts, lls)]))
    res["avg_jaccard_sim"] = float(np.mean([_jac(g, l) for g, l in zip(gts, lls)]))
    res["avg_rougeL"] = float(np.mean([_rougeL(g, l) for g, l in zip(gts, lls)]))
    if use_bertscore:
        global _bert
        try:
            with _bert_lock:
                if _bert is None:
                    from bert_score import BERTScorer
                    _bert = BERTScorer(lang="en", rescale_with_baseline=False)
                _, _, F = _bert.score(lls, gts)
            res["avg_bertscore"] = float(F.mean())
        except Exception as e:  # noqa: BLE001
            print(f"  ! BERTScore unavailable: {e}")
    return res


# =============================================================================
# Evaluator
# =============================================================================
class Evaluator:
    def __init__(self, a):
        self.a = a
        self.judge = ChatClient(model=a.judge_model, base_url=None if is_openai_model(a.judge_model) else a.judge_base_url,
                                max_model_len=a.judge_max_model_len, tokenizer=a.judge_tokenizer,
                                timeout=120, max_retries=5)
        self.cache = Path(a.out_dir) / "pair_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fp_client = None
        if a.fp:
            m = a.fp_model or a.judge_model
            self.fp_client = ChatClient(model=m, base_url=None if is_openai_model(m) else (a.fp_base_url or a.judge_base_url),
                                        tokenizer=a.fp_tokenizer, max_model_len=a.fp_max_model_len, timeout=300)

    def judge_pair(self, gt, llm):
        key = hashlib.sha256(f"{self.a.judge_model}|{gt}|{llm}".encode()).hexdigest()[:32]
        p = self.cache / f"{key}.txt"
        if p.exists():
            return p.read_text()
        ans = self.judge.chat("", pair_eval_prompt(gt, llm), max_tokens=8, temperature=0.0, top_p=1.0)
        if not ans.startswith("ERROR"):
            p.write_text(ans)
        return ans

    def eval_row(self, gt_list, llm_list):
        pairs = [(g, l) for g in gt_list for l in llm_list]
        with ThreadPoolExecutor(max_workers=self.a.pair_workers) as ex:
            answers = list(ex.map(lambda pr: self.judge_pair(pr[0]["gt_limitation"], pr[1]["llm_limitation"]), pairs))
        return [[f"Pair {i+1}: {ans}", f"gt_id:{g['gt_id']}", f"gt_limitation:{g['gt_limitation']}",
                 f"llm_id:{l['llm_id']}", f"llm_limitation:{l['llm_limitation']}"]
                for i, ((g, l), ans) in enumerate(zip(pairs, answers))]

    def classify_unmatched(self, results, manuscript):
        info = {}
        for it in results:
            lid, lim = _field(it, "llm_id:"), _field(it, "llm_limitation:")
            verdict = str(it[0]).split(":", 1)[-1].strip().lower()
            e = info.setdefault(lid, {"lim": lim, "yes": False})
            e["yes"] |= "yes" in verdict
        out = {}
        tok = self.fp_client.tok
        budget = self.fp_client.max_model_len - 400 - 64
        for lid, e in info.items():
            if e["yes"]:
                continue
            base = tok.count(fp_user_prompt(e["lim"], "")) + tok.count(FP_SYSTEM_PROMPT)
            ms = self.fp_client.truncate(manuscript, max(0, budget - base), suffix="")
            key = hashlib.sha256(f"fp|{self.fp_client.model}|{e['lim']}|{ms[:2000]}".encode()).hexdigest()[:32]
            cp = self.cache / f"fp_{key}.json"
            if cp.exists():
                obj = json.loads(cp.read_text())
            else:
                obj = parse_json_obj(self.fp_client.chat(FP_SYSTEM_PROMPT, fp_user_prompt(e["lim"], ms),
                                                         max_tokens=400, temperature=0.1, top_p=0.9,
                                                         json_mode=self.fp_client.is_openai))
                cp.write_text(json.dumps(obj))
            try:
                cat = int(obj.get("category"))
            except (TypeError, ValueError):
                cat = None
            out[lid] = {"fp_category": "" if cat is None else str(cat),
                        "fp_category_label": CATEGORY_LABELS.get(cat, "ParseError" if not obj else "Unknown"),
                        "fp_valid": "Yes" if cat in VALID_CATEGORIES else "No",
                        "fp_evidence": obj.get("evidence", ""), "fp_reasoning": obj.get("reasoning", "")}
        return out

    def evaluate_column(self, df, col, out_csv):
        a = self.a
        parser = PARSERS[a.parser]
        work = df.copy()
        if os.path.exists(out_csv):
            old = pd.read_csv(out_csv)
            if "row_id" in old.columns and f"{col}__eval" in old.columns:
                m = old.set_index("row_id")[f"{col}__eval"].to_dict()
                work[f"{col}__eval"] = work["row_id"].map(m)
        if f"{col}__eval" not in work.columns:
            work[f"{col}__eval"] = None
        work[f"{col}__eval"] = work[f"{col}__eval"].astype(object)
        work[f"{col}__parsed"] = work[col].apply(lambda t: json.dumps(parser(t) if isinstance(t, str) else []))
        since = 0
        for i in range(len(work)):
            if not is_missing(work.at[i, f"{col}__eval"]):
                continue
            llm = json.loads(work.at[i, f"{col}__parsed"])
            gt = parse_gt(work.at[i, a.gt_col] if a.gt_col in work.columns else "")
            text = str(work.at[i, col])
            if text.startswith(("SKIPPED", "ERROR", "NO_OUTPUT", "NO_REPORT")):
                llm = []
            res = self.eval_row(gt, llm)
            if a.fp and res:
                fp = self.classify_unmatched(res, str(work.at[i, a.text_col]) if a.text_col in work.columns else "")
                empty = {k: "" for k in ("fp_category", "fp_category_label", "fp_valid", "fp_evidence", "fp_reasoning")}
                res = [it + [f"{k}:{v}" for k, v in fp.get(_field(it, "llm_id:"), empty).items()] for it in res]
            work.at[i, f"{col}__eval"] = repr(res)
            since += 1
            if since >= a.save_every:
                atomic_to_csv(work, out_csv)
                since = 0
                print(f"  [eval:{col}] saved at row {i+1}/{len(work)}", flush=True)
        for use_fp, sfx in ((False, ""), (True, "_fp")) if a.fp else ((False, ""),):
            prf = work[f"{col}__eval"].apply(lambda v: compute_prf(v, use_fp)).apply(pd.Series)
            sim = work[f"{col}__eval"].apply(lambda v: compute_similarity(v, a.bertscore, use_fp)).apply(pd.Series)
            for c in prf.columns:
                work[f"{col}__{c}{sfx}"] = prf[c]
            for c in sim.columns:
                work[f"{col}__{c}{sfx}"] = sim[c]
        atomic_to_csv(work, out_csv)
        return work

    def summarize(self, work, col, source):
        valid = (work[f"{col}__n_unique_gt"] > 0) & (work[f"{col}__n_unique_llm"] > 0)
        s = {"column": col, "source": source, "n_rows": len(work), "n_valid": int(valid.sum())}
        for m in ("precision", "recall", "f1", "avg_cosine_sim", "avg_jaccard_sim", "avg_rougeL", "avg_bertscore",
                  "n_yes_pairs", "n_unique_llm"):
            s[m] = float(work[f"{col}__{m}"].mean())
        for m in ("precision", "recall", "f1"):
            s[f"{m}_valid"] = float(work.loc[valid, f"{col}__{m}"].mean()) if valid.any() else 0.0
            if self.a.fp:
                s[f"{m}_fp"] = float(work[f"{col}__{m}_fp"].mean())
        return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_csv", nargs="*", default=[])
    p.add_argument("--gen_dir", default=None, help="evaluate every generations/<tag>.csv (column <tag>)")
    p.add_argument("--columns", nargs="*", default=None, help="columns to evaluate (default: file tag)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--parser", default="auto", choices=list(PARSERS))
    p.add_argument("--gt_col", default="ground_truth_lim_peer")
    p.add_argument("--text_col", default="input_text_cleaned")
    p.add_argument("--start_row", type=int, default=0)
    p.add_argument("--end_row", type=int, default=None)
    p.add_argument("--sample_n", type=int, default=None, help="random subset (eval.py used 30 of first 100)")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--judge_base_url", default=None)
    p.add_argument("--judge_tokenizer", default=None)
    p.add_argument("--judge_max_model_len", type=int, default=32768)
    p.add_argument("--pair_workers", type=int, default=8)
    p.add_argument("--fp", action="store_true", help="false-positive classification of unmatched LUs")
    p.add_argument("--fp_model", default=None, help="e.g. a served Qwen3-32B-AWQ; default = judge")
    p.add_argument("--fp_base_url", default=None)
    p.add_argument("--fp_tokenizer", default=None)
    p.add_argument("--fp_max_model_len", type=int, default=16384)
    p.add_argument("--bertscore", type=int, default=1)
    p.add_argument("--save_every", type=int, default=10)
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    ev = Evaluator(a)
    jobs = [(f, a.columns) for f in a.input_csv]
    if a.gen_dir:
        jobs += [(f, None) for f in sorted(glob.glob(os.path.join(a.gen_dir, "*.csv")))]
    summaries = []
    for f, cols in jobs:
        df = pd.read_csv(f)
        if "row_id" not in df.columns:
            df.insert(0, "row_id", range(len(df)))
        df = df.iloc[a.start_row:a.end_row].reset_index(drop=True)
        if a.sample_n:
            df = df.sample(n=min(a.sample_n, len(df)), random_state=a.sample_seed).reset_index(drop=True)
        tag = Path(f).stem
        cols = cols or ([tag] if tag in df.columns else [])
        for col in cols:
            if col not in df.columns:
                print(f"[skip] {col} not in {f}")
                continue
            print(f"\n=== evaluating {col} ({f}) ===")
            out_csv = os.path.join(a.out_dir, f"eval__{tag}__{col}.csv"[:200])
            work = ev.evaluate_column(df, col, out_csv)
            s = ev.summarize(work, col, f)
            summaries.append(s)
            print(json.dumps(s, indent=2))
    if summaries:
        sp = os.path.join(a.out_dir, "summary.csv")
        new = pd.DataFrame(summaries)
        if os.path.exists(sp):
            old = pd.read_csv(sp)
            new = pd.concat([old[~old.set_index(["column", "source"]).index.isin(
                new.set_index(["column", "source"]).index)], new], ignore_index=True)
        atomic_to_csv(new, sp)
        print(f"\nSummary -> {sp}")
        print(new[["column", "precision", "recall", "f1", "avg_rougeL", "avg_bertscore"]].to_string(index=False))


if __name__ == "__main__":
    main()
