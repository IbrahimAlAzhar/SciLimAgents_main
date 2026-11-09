"""
OUTCOME reward — hybrid LLM-as-judge reward model (reward_model_no_need.py,
worker_grpo.py). Scores the FINAL output of each agent:

  LAYER 1  LLM judge (gpt-4o-mini or local): bipartite pred<->GT matching
           (F-beta, beta=1.5, novelty bonus for valid+grounded unmatched items),
           per-item grounded/specific/valid/severity, master organisation/dedup,
           leader-feedback quality.
  LAYER 2  NLP: ROUGE-L vs GT, lexical cross-check of judge matches.
  LAYER 3  Rules: length penalty, redundancy penalty, boilerplate detector.

Composite weights per role are unchanged. Worker GT is restricted to the GT
items the judge assigns to the worker's specialty (fallback: all GT).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List

from ..prompts import judge as P
from ..utils.text import split_bullets
from . import common as K
from .stepwise import select_from_candidates

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:  # pragma: no cover
    HAS_ROUGE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

PAPER_TRUNC_CHARS = 16000
DOC_TRUNC_CHARS = 6000
NOVELTY_WEIGHT = 0.5
FBETA = 1.5
IDEAL_ITEM_COUNT_MIN, IDEAL_ITEM_COUNT_MAX = 3, 20
LENGTH_PENALTY_WEIGHT = REDUNDANCY_WEIGHT = ROUGE_WEIGHT = 0.10
WEIGHTS = {
    "worker": dict(f1=0.40, grounding=0.15, specificity=0.10, severity=0.05, rouge=ROUGE_WEIGHT,
                   length_pen=LENGTH_PENALTY_WEIGHT, redundancy_pen=REDUNDANCY_WEIGHT),
    "leader_feedback": dict(delta=0.40, feedback_quality=0.35, room_adjusted=0.25),
    "leader_handoff": dict(coverage=0.55, structure=0.45),
    "master": dict(f1=0.40, grounding=0.10, specificity=0.05, severity=0.05, dedup=0.10, organization=0.10,
                   rouge=ROUGE_WEIGHT, length_pen=LENGTH_PENALTY_WEIGHT, redundancy_pen=REDUNDANCY_WEIGHT),
}
GT_CATEGORIES = ["novelty_significance", "theoretical_methodological", "experimental_evaluation",
                 "generalization_robustness_efficiency", "clarity_interpretability_reproducibility",
                 "data_ethics", "other"]
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]\s|\Z)", re.DOTALL | re.MULTILINE)


def split_gt(ground_truth):
    if not ground_truth:
        return []
    return [l.strip() for l in ground_truth.splitlines() if l.strip() and len(l.strip()) > 10]


def _jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def compute_length_penalty(n):
    if IDEAL_ITEM_COUNT_MIN <= n <= IDEAL_ITEM_COUNT_MAX:
        return 0.0
    if n < IDEAL_ITEM_COUNT_MIN:
        return min(1.0, (IDEAL_ITEM_COUNT_MIN - n) / IDEAL_ITEM_COUNT_MIN)
    return min(1.0, math.log1p(n - IDEAL_ITEM_COUNT_MAX) / math.log1p(20))


def compute_redundancy_penalty(items):
    if len(items) < 2:
        return 0.0
    if HAS_SKLEARN and len(items) >= 3:
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(items)
            sim = sk_cosine(tfidf)
            for i in range(len(items)):
                sim[i, i] = 0.0
            return min(1.0, sum(1 for s in sim.max(axis=1) if s > 0.6) / len(items))
        except Exception:
            pass
    hi = tot = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            hi += _jaccard(items[i], items[j]) > 0.5
            tot += 1
    return min(1.0, hi / max(tot, 1) * 3)


_BP_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can", "need", "dare", "ought", "used",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "not", "no", "so", "than", "too", "very", "just", "about", "up", "down", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such", "only", "own", "same", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "their", "we", "our", "you", "your", "which",
    "what", "who", "whom", "how", "when", "where", "why", "if", "because", "while", "paper", "study", "work",
    "research", "method", "approach", "model", "results", "data", "analysis", "experiment", "also", "however",
    "limitation", "lack", "missing", "insufficient", "limited", "without",
}


def compute_boilerplate_ratio(items, paper_text):
    if not items or not paper_text:
        return 0.5
    low = paper_text.lower()
    bp = 0
    for item in items:
        words = [w for w in re.findall(r"[a-z]+", item.lower()) if w not in _BP_STOP and len(w) > 3]
        if not words or sum(1 for w in words if w in low) / len(words) < 0.15:
            bp += 1
    return bp / len(items)


def compute_rouge_scores(pred_text, gt_text):
    if not HAS_ROUGE or not pred_text.strip() or not gt_text.strip():
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(gt_text, pred_text)["rougeL"]
    return {"rouge_l_precision": s.precision, "rouge_l_recall": s.recall, "rouge_l_f1": s.fmeasure}


def compute_lexical_match_verification(pred, gt, matches):
    if not matches:
        return 1.0
    v = 0.0
    for m in matches:
        pi, gi = m.get("pred", -1), m.get("gt", -1)
        if 0 <= pi < len(pred) and 0 <= gi < len(gt):
            v += 1.0 if _jaccard(pred[pi], gt[gi]) > 0.03 else 0.5
        else:
            v += 0.5
    return v / len(matches)


def f_beta(p, r, beta=FBETA):
    if p + r < 1e-9:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r + 1e-9)


class OutcomeScorer:
    def __init__(self, judge: K.Judge, sft_min=0.15, dpo_min_gap=0.05, chosen_floor=0.3,
                 sft_top_k=None, leader_feedback_sft=False):
        self.j = judge
        self.sft_min, self.min_gap, self.floor = sft_min, dpo_min_gap, chosen_floor
        self.top_k = sft_top_k or {"worker": 1, "leader": 2, "master": 3}
        self.leader_feedback_sft = leader_feedback_sft

    # ---------------- judge layers ----------------
    def categorize_gt(self, pid, gt_items):
        if not gt_items:
            return {}
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(gt_items))
        out = self.j.json_call(f"gt_cat_p{pid}", P.JUDGE_JSON_SYSTEM,
                               P.GT_CAT_PROMPT.format(numbered=numbered) + "\n\n" + P.GT_CAT_SCHEMA)
        mapping = {}
        for a in out.get("assignments", []) or []:
            idx = a.get("index") if isinstance(a, dict) else None
            if isinstance(idx, int) and 1 <= idx <= len(gt_items):
                spec = str(a.get("specialty", "other"))
                mapping[idx - 1] = spec if spec in GT_CATEGORIES else "other"
        return mapping

    @staticmethod
    def gt_for_specialty(gt_items, cats, spec):
        sub = [gt_items[i] for i in range(len(gt_items)) if cats.get(i) == spec]
        return sub if sub else gt_items

    def bipartite_match(self, pred, gt, tag):
        if not pred or not gt:
            return {"matches": [], "unmatched_pred": list(range(len(pred))), "unmatched_gt": list(range(len(gt)))}
        out = self.j.json_call(tag, P.JUDGE_JSON_SYSTEM, P.MATCH_PROMPT.format(
            gt_block="\n".join(f"G{i+1}. {s}" for i, s in enumerate(gt)),
            pred_block="\n".join(f"P{i+1}. {s}" for i, s in enumerate(pred))) + "\n\n" + P.MATCH_SCHEMA)
        raw = []
        for m in out.get("matches", []) or []:
            if not isinstance(m, dict):
                continue
            p, g = m.get("pred"), m.get("gt")
            try:
                c = float(m.get("confidence", 0.5))
            except Exception:
                c = 0.5
            if isinstance(p, int) and isinstance(g, int) and 1 <= p <= len(pred) and 1 <= g <= len(gt) and c >= 0.55:
                raw.append({"pred": p - 1, "gt": g - 1, "confidence": c})
        raw.sort(key=lambda x: -x["confidence"])
        tp, tg, matches = set(), set(), []
        for m in raw:
            if m["pred"] in tp or m["gt"] in tg:
                continue
            matches.append(m)
            tp.add(m["pred"])
            tg.add(m["gt"])
        return {"matches": matches, "unmatched_pred": [i for i in range(len(pred)) if i not in tp],
                "unmatched_gt": [i for i in range(len(gt)) if i not in tg]}

    def judge_quality(self, items, paper, tag):
        if not items:
            return []
        out = self.j.json_call(tag, P.JUDGE_JSON_SYSTEM, P.QUALITY_PROMPT.format(
            paper=paper[:PAPER_TRUNC_CHARS], items="\n".join(f"L{i+1}. {s}" for i, s in enumerate(items)))
            + "\n\n" + P.QUALITY_SCHEMA, max_tokens=3000)
        by = {j["index"]: j for j in out.get("judgments", []) or [] if isinstance(j, dict) and isinstance(j.get("index"), int)}
        res = []
        for i in range(len(items)):
            j = by.get(i + 1, {})
            try:
                sev = min(2, max(0, int(j.get("severity", 0))))
            except Exception:
                sev = 0
            res.append({"grounded": int(bool(j.get("grounded", 0))), "specific": int(bool(j.get("specific", 0))),
                        "valid": int(bool(j.get("valid", 0))), "severity": sev})
        return res

    # ---------------- scoring ----------------
    def score_item_list(self, pred_text, gt_items, paper, tag):
        pred = split_bullets(pred_text)
        n_pred, n_gt = len(pred), len(gt_items)
        if not pred:
            return dict(f1=0.0, grounding=0.0, specificity=0.0, severity_avg=0.0, precision=0.0, recall=0.0,
                        n_pred=0, n_gt=n_gt, n_matched=0, n_novel_valid=0, rouge_l_f1=0.0, length_penalty=1.0,
                        redundancy_penalty=0.0, boilerplate_ratio=1.0, match_verification=1.0)
        mr = self.bipartite_match(pred, gt_items, f"{tag}__match") if gt_items else \
            {"matches": [], "unmatched_pred": list(range(n_pred)), "unmatched_gt": []}
        qual = self.judge_quality(pred, paper, f"{tag}__qual")
        n_matched = len(mr["matches"])
        n_novel = sum(1 for i in mr["unmatched_pred"] if i < len(qual) and qual[i]["valid"] and qual[i]["grounded"])
        precision = (n_matched + NOVELTY_WEIGHT * n_novel) / max(n_pred, 1)
        recall = (n_matched / max(n_gt, 1)) if n_gt else 0.0
        f1 = f_beta(precision, recall) if n_gt else precision
        grounding = sum(q["grounded"] for q in qual) / max(n_pred, 1)
        specificity = sum(q["specific"] for q in qual) / max(n_pred, 1)
        severity = sum(q["severity"] for q in qual) / (2.0 * max(n_pred, 1))
        rouge = compute_rouge_scores(" ".join(pred), " ".join(gt_items) if gt_items else "")["rouge_l_f1"]
        mv = compute_lexical_match_verification(pred, gt_items, mr["matches"])
        lp, rp = compute_length_penalty(n_pred), compute_redundancy_penalty(pred)
        bp = compute_boilerplate_ratio(pred, paper)
        return dict(f1=f1 * (0.5 + 0.5 * mv), grounding=grounding * (1.0 - 0.3 * bp), specificity=specificity,
                    severity_avg=severity, precision=precision, recall=recall, n_pred=n_pred, n_gt=n_gt,
                    n_matched=n_matched, n_novel_valid=n_novel, rouge_l_f1=rouge, length_penalty=lp,
                    redundancy_penalty=rp, boilerplate_ratio=bp, match_verification=mv, raw_f1=f1,
                    raw_grounding=grounding)

    @staticmethod
    def composite_worker(sc):
        w = WEIGHTS["worker"]
        raw = (w["f1"] * sc.get("f1", 0) + w["grounding"] * sc.get("grounding", 0)
               + w["specificity"] * sc.get("specificity", 0) + w["severity"] * sc.get("severity_avg", 0)
               + w["rouge"] * sc.get("rouge_l_f1", 0) - w["length_pen"] * sc.get("length_penalty", 0)
               - w["redundancy_pen"] * sc.get("redundancy_penalty", 0))
        return max(0.0, min(1.0, raw))

    def score_leader_feedback(self, init_sc, rev_sc, worker_init, feedback, specialty, tag):
        ci, cr = self.composite_worker(init_sc), self.composite_worker(rev_sc)
        delta = cr - ci
        room_adj = max(0.0, min(1.0, delta / max(1.0 - ci, 0.05)))
        delta_score = max(0.0, min(1.0, 0.5 + 2.0 * delta))
        out = self.j.json_call(tag, P.JUDGE_JSON_SYSTEM, P.FEEDBACK_QUALITY_PROMPT.format(
            specialty=specialty, worker_output=worker_init[:DOC_TRUNC_CHARS], feedback=feedback[:DOC_TRUNC_CHARS])
            + "\n\n" + P.FEEDBACK_SCHEMA)

        def f(k):
            try:
                return max(0.0, min(1.0, float(out.get(k, 0.5))))
            except Exception:
                return 0.5
        fq = {k: f(k) for k in ("actionable", "accurate", "targeted")}
        fqs = sum(fq.values()) / 3.0
        w = WEIGHTS["leader_feedback"]
        comp = max(0.0, min(1.0, w["delta"] * delta_score + w["feedback_quality"] * fqs + w["room_adjusted"] * room_adj))
        return dict(delta=delta, delta_score=delta_score, room_adjusted_delta=room_adj, feedback_quality=fqs,
                    feedback_detail=fq, composite=comp, initial_composite=ci, revised_composite=cr)

    @staticmethod
    def score_leader_handoff(text, worker_outputs):
        present = sum(1 for n in worker_outputs if n in text)
        coverage = present / max(len(worker_outputs), 1)
        n_b = len(_BULLET_RE.findall(text))
        has_sec = sum(1 for n in worker_outputs if n in text and "###" in text)
        s = 0.3 if "master" in text.lower() else 0.0
        s += 0.3 if n_b >= 6 else (0.15 if n_b >= 3 else 0.0)
        s += 0.4 if has_sec >= len(worker_outputs) * 0.5 else (0.2 if has_sec > 0 else 0.0)
        s = min(1.0, s)
        w = WEIGHTS["leader_handoff"]
        return dict(coverage=coverage, structure=s, composite=w["coverage"] * coverage + w["structure"] * s)

    def score_master(self, text, gt_items, paper, tag):
        sc = self.score_item_list(text, gt_items, paper, tag)
        org = self.j.json_call(f"{tag}__org", P.JUDGE_JSON_SYSTEM,
                               P.ORG_PROMPT.format(doc=text[:DOC_TRUNC_CHARS]) + "\n\n" + P.ORG_SCHEMA)

        def f(k):
            try:
                return max(0.0, min(1.0, float(org.get(k, 0.5))))
            except Exception:
                return 0.5
        o, d, coh = f("organization"), f("dedup"), f("coherence")
        w = WEIGHTS["master"]
        comp = (w["f1"] * sc["f1"] + w["grounding"] * sc["grounding"] + w["specificity"] * sc["specificity"]
                + w["severity"] * sc["severity_avg"] + w["dedup"] * d + w["organization"] * o
                + w["rouge"] * sc["rouge_l_f1"] - w["length_pen"] * sc["length_penalty"]
                - w["redundancy_pen"] * sc["redundancy_penalty"])
        return {**sc, "organization": o, "dedup": d, "coherence": coh, "composite": max(0.0, min(1.0, comp))}

    # ---------------- a single completion (used by GRPO) ----------------
    def score_worker_text(self, text, gt_items, paper, tag):
        sc = self.score_item_list(text, gt_items, paper, tag)
        sc["composite"] = self.composite_worker(sc)
        return sc

    # ---------------- per paper ----------------
    def candidates(self, paper):
        pid = paper.get("paper_idx")
        system = paper.get("system", "limagents")
        src = paper.get("paper_text", "") or ""
        gt_items = split_gt(paper.get("ground_truth", "") or "")
        cats = self.categorize_gt(pid, gt_items) if gt_items else {}
        cands = []
        for ro in paper.get("rollouts", []):
            if ro.get("error") or not ro.get("samples"):
                continue
            rid, mode = ro.get("rollout_id"), ro.get("mode")
            wf = K.extract_worker_final(ro)
            w_scores = {}
            for name, tr in wf.items():
                spec = K.specialty_of(system, name)
                gt_sub = self.gt_for_specialty(gt_items, cats, spec)
                sc = self.score_worker_text(tr["synthesize"], gt_sub, src, f"p{pid}_r{rid}_{name}_final")
                w_scores[name] = (sc, gt_sub, spec, tr)
                cands.append({"paper_idx": pid, "rollout_id": rid, "mode": mode, "agent_role": "worker",
                              "agent_name": name, "output": tr["synthesize"], "system": system,
                              "score": sc["composite"], "stepwise_score": None, "gt_similarity": sc.get("raw_f1"),
                              "score_detail": {"total": sc["composite"], "steps": sc,
                                               "n_units": sc.get("n_pred")}, "_tr": tr})
            # leader feedback (critique rounds): causal delta draft->revision + feedback quality
            for s in ro.get("samples", []):
                if s.get("agent_role") == "leader" and s.get("turn_type") == "feedback_to_worker":
                    tgt = s.get("target_worker", "")
                    if tgt not in w_scores:
                        continue
                    sc_rev, gt_sub, spec, tr = w_scores[tgt]
                    sc_init = self.score_worker_text(tr["draft"], gt_sub, src, f"p{pid}_r{rid}_{tgt}_draft")
                    fb = self.score_leader_feedback(sc_init, sc_rev, tr["draft"], s.get("output", ""), spec,
                                                    f"p{pid}_r{rid}_leader_fb_{tgt}_{s.get('critique_round', 1)}")
                    cands.append({"paper_idx": pid, "rollout_id": rid, "mode": mode,
                                  "agent_role": "leader_feedback", "agent_name": f"Leader_Agent->{tgt}",
                                  "output": s.get("output", ""), "input_messages": s.get("input_messages", []),
                                  "system": system, "score": fb["composite"],
                                  "score_detail": {"total": fb["composite"], "steps": fb}})
            ls = K.find_sample(ro, "leader", "handoff_to_master")
            if ls and (ls.get("output") or "").strip():
                sc = self.score_leader_handoff(ls["output"], {n: t["synthesize"] for n, t in wf.items()})
                cands.append({"paper_idx": pid, "rollout_id": rid, "mode": mode, "agent_role": "leader",
                              "agent_name": "Leader_Agent", "output": ls["output"], "system": system,
                              "input_messages": ls.get("input_messages", []), "score": sc["composite"],
                              "score_detail": {"total": sc["composite"], "steps": sc}})
            ms = K.find_sample(ro, "master", "synthesis")
            if ms and (ms.get("output") or "").strip():
                sc = self.score_master(ms["output"], gt_items, src, f"p{pid}_r{rid}_master")
                cands.append({"paper_idx": pid, "rollout_id": rid, "mode": mode, "agent_role": "master",
                              "agent_name": "Master_Agent", "output": ms["output"], "system": system,
                              "input_messages": ms.get("input_messages", []), "score": sc["composite"],
                              "gt_similarity": sc.get("raw_f1"),
                              "score_detail": {"total": sc["composite"], "steps": sc, "n_units": sc.get("n_pred")}})
        return cands

    def score_paper(self, paper):
        cands = self.candidates(paper)
        sel = [c for c in cands if c["agent_role"] != "leader_feedback"]
        clean, sft, dpo = select_from_candidates(paper, sel, self.sft_min, self.sft_min, self.min_gap,
                                                 chosen_floor=self.floor, sft_top_k_by_role=self.top_k)
        # leader/master: top-k SFT (reward_model.py SFT_TOP_K_BY_ROLE) instead of best-only
        sft = [s for s in sft if s["agent_role"] == "worker"]
        groups = defaultdict(list)
        for c in sel:
            if c["agent_role"] in ("leader", "master"):
                groups[(c["agent_role"], c["agent_name"])].append(c)
        if self.leader_feedback_sft:
            for c in cands:
                if c["agent_role"] == "leader_feedback":
                    groups[("leader", "Leader_Agent_feedback")].append(c)
        for (role, name), g in groups.items():
            g = sorted(g, key=lambda c: c["score"], reverse=True)[: self.top_k.get(role, 1)]
            for c in g:
                K.sft_best([c], role, name, paper.get("paper_idx"), sft, min_score=self.sft_min)
        fb = [{k: v for k, v in c.items() if k != "_tr"} for c in cands if c["agent_role"] == "leader_feedback"]
        return clean + fb, sft, dpo
