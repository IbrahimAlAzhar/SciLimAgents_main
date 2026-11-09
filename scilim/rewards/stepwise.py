"""
STEPWISE (process) reward  — stepwise_reward_select.py

    hop_search            -> SEARCH     (0.22)
    hop_query             -> RETRIEVE   (0.30)
    hop_draft / revised   -> SYNTHESIZE (0.48)
    worker_total = weighted sum;  master/leader/NovAgents workers = synthesize only

blended with ground-truth similarity:
    final = 0.70 * stepwise + 0.30 * gt_alignment
    worker gt_alignment = 0.70 LLM judge (gpt-4o-mini or local) + 0.15 ROUGE-L + 0.15 lexical
    master/leader gt    = 0.5 bullet-match F1 + 0.5 ROUGE-L

The stepwise part is API-free; only the GT-alignment judge calls an LLM.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Dict, List

from ..prompts.judge import ALIGN_SYSTEM, align_prompt
from ..utils.text import split_bullets, strip_think
from . import common as K

try:
    from rouge_score import rouge_scorer
    _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except Exception:  # pragma: no cover
    _ROUGE = None

STEP_WEIGHTS = {"search": 0.22, "retrieve": 0.30, "synthesize": 0.48}
SOURCE_TRUNC = 16000
GT_MATCH_JACCARD = 0.18
WORKER_GT_WEIGHTS = {"llm_align": 0.70, "rouge_l": 0.15, "lexical": 0.15}

SPECIALTY_KEYWORDS = {
    "novelty_significance": ["novel", "novelty", "significance", "contribution", "incremental",
                             "original", "prior work", "impact", "motivation"],
    "theoretical_methodological": ["theoretical", "theory", "assumption", "proof", "derivation",
                                   "method", "methodolog", "ablation", "component", "formulation"],
    "experimental_evaluation": ["experiment", "evaluation", "baseline", "metric", "statistical",
                                "significance test", "comparison", "benchmark", "result", "error bar"],
    "generalization_robustness_efficiency": ["generaliz", "robust", "efficien", "latency", "compute",
                                             "cost", "out-of-distribution", "ood", "scalab",
                                             "deployment", "resource"],
    "clarity_interpretability_reproducibility": ["clarity", "clear", "interpret", "explain", "reproduc",
                                                 "replicat", "code", "seed", "hyperparameter", "document"],
    "data_ethics": ["data", "dataset", "bias", "fairness", "ethic", "privacy", "label", "annotation",
                    "representative", "societal"],
}


def _clamp(v):
    return max(0.0, min(1.0, float(v)))


_STOP = {"the", "and", "for", "that", "with", "this", "from", "paper", "study",
         "method", "model", "results", "limitation", "limitations"}


def _tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 2 and t not in _STOP}


def overlap(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _contains_quote(quote, source_text, min_chars=18):
    quote = re.sub(r"\s+", " ", str(quote or "")).strip().lower()
    if len(quote) < min_chars:
        return False
    return quote[:180] in re.sub(r"\s+", " ", source_text).lower()


def build_units(text: str) -> List[Dict[str, Any]]:
    units = []
    for b in split_bullets(text):
        quotes = re.findall(r'["“‘\']([^"”’\']{12,200})["”’\']', b)
        units.append({"statement": b, "evidence_pointers": [{"quote": q} for q in quotes],
                      "impact": "", "suggested_fix": "", "severity": "", "title": ""})
    return units


# ------------------------------------------------------------- step scorers
def score_search(search_text):
    search = strip_think(search_text).strip()
    if not search:
        return 0.0
    lines = [l for l in search.splitlines() if l.strip()]
    has_sources = any(w in search.lower() for w in ("paper", "cited", "rag", "abstract", "introduction", "baseline"))
    specificity = min(1.0, len(_tokens(search)) / 35.0)
    structure = min(1.0, len(lines) / 3.0)
    return round(_clamp(0.35 + 0.35 * specificity + 0.20 * structure + 0.10 * float(has_sources)), 4)


def score_retrieve(retrieve_text, source_text):
    retrieve = strip_think(retrieve_text).strip()
    if not retrieve:
        return 0.0
    snippets = re.findall(r"(?:Evidence|Quote|Span)\s*:?\s*([^.\n]{18,220}(?:\.|$))", retrieve, flags=re.IGNORECASE)
    if not snippets:
        snippets = [l.strip(" -0123456789.") for l in retrieve.splitlines() if len(l.strip()) > 18]
    grounded = sum(1 for s in snippets[:6]
                   if _contains_quote(s, source_text) or overlap(s, source_text[:12000]) > 0.05)
    grounded_score = grounded / max(1, min(len(snippets), 6))
    density = min(1.0, len(snippets) / 3.0)
    return round(_clamp(0.25 + 0.55 * grounded_score + 0.20 * density), 4)


def _score_specificity(unit):
    text = " ".join(str(unit.get(k) or "") for k in ("title", "statement", "impact", "suggested_fix"))
    words = str(unit.get("statement") or "").split()
    score = 0.0
    if 8 <= len(words) <= 55:
        score += 0.30
    elif 5 <= len(words) <= 80:
        score += 0.16
    if any(t in text.lower() for t in ("baseline", "ablation", "dataset", "metric", "seed", "split", "compute",
                                       "code", "reproduc", "rag", "cited", "robust")):
        score += 0.35
    if re.search(r"\b\d+(?:\.\d+)?%?|\b[A-Z][A-Za-z0-9_-]{2,}\b", text):
        score += 0.15
    if str(unit.get("impact") or "").strip():
        score += 0.10
    if str(unit.get("suggested_fix") or "").strip():
        score += 0.10
    generic = len(re.findall(r"\b(some|various|several|many|things|issues|aspects)\b", text.lower()))
    score -= min(0.25, 0.05 * generic)
    return _clamp(score)


def _score_unit(unit, units, source_text, index):
    statement = str(unit.get("statement") or "")
    quotes = [str(p.get("quote") or "") for p in unit.get("evidence_pointers") or [] if isinstance(p, dict)]
    gs = [1.0 if _contains_quote(q, source_text) else min(1.0, 3.0 * overlap(q, source_text[:SOURCE_TRUNC]))
          for q in quotes]
    groundedness = max(gs, default=min(1.0, 2.2 * overlap(statement, source_text[:SOURCE_TRUNC])))
    evidence_quality = _clamp(0.35 * bool(quotes) + 0.65 * groundedness)
    specificity = _score_specificity(unit)
    actionability = _clamp(0.45 * bool(str(unit.get("impact") or "").strip())
                           + 0.45 * bool(str(unit.get("suggested_fix") or "").strip())
                           + 0.10 * bool(str(unit.get("severity") or "").strip()))
    redundancy = _clamp(max((overlap(statement, str(o.get("statement") or "")) for o in units
                             if str(o.get("statement") or "") != statement), default=0.0))
    score = _clamp(0.42 * groundedness + 0.24 * specificity + 0.20 * evidence_quality
                   + 0.10 * actionability + 0.04 * (1.0 - redundancy))
    return {"index": index, "statement": statement[:220], "score": round(score, 4),
            "groundedness": round(groundedness, 4), "specificity": round(specificity, 4),
            "evidence_quality": round(evidence_quality, 4), "actionability": round(actionability, 4),
            "redundancy_penalty": round(redundancy, 4)}


def score_synthesize(units, source_text):
    if not units:
        return 0.0, []
    us = [_score_unit(u, units, source_text, i) for i, u in enumerate(units)]
    mean_unit = sum(u["score"] for u in us) / len(us)
    n = len(units)
    count_factor = 1.0 if 2 <= n <= 7 else (0.80 if n == 1 or n <= 10 else 0.55)
    return round(_clamp(0.90 * mean_unit + 0.10 * count_factor), 4), us


def score_worker(search_text, retrieve_text, synth_text, source_text):
    s, r = score_search(search_text), score_retrieve(retrieve_text, source_text)
    syn, us = score_synthesize(build_units(synth_text), source_text)
    total = _clamp(STEP_WEIGHTS["search"] * s + STEP_WEIGHTS["retrieve"] * r + STEP_WEIGHTS["synthesize"] * syn)
    return {"total": round(total, 4), "steps": {"search": s, "retrieve": r, "synthesize": syn}, "n_units": len(us)}


def score_synth_only(synth_text, source_text):
    syn, us = score_synthesize(build_units(synth_text), source_text)
    return {"total": round(syn, 4), "steps": {"search": None, "retrieve": None, "synthesize": syn},
            "n_units": len(us)}


score_master = score_synth_only


# --------------------------------------------------------- GT similarity
def split_gt(gt_text):
    if not gt_text:
        return []
    items = [re.sub(r"^\s*[-*•\d.]+\s*", "", l).strip() for l in str(gt_text).splitlines()]
    return [i for i in items if len(i) > 10]


def rouge_l_f1(pred, gt):
    if not _ROUGE or not pred.strip() or not gt.strip():
        return 0.0
    return float(_ROUGE.score(gt, pred)["rougeL"].fmeasure)


def bullet_match_f1(pred_items, gt_items):
    if not pred_items or not gt_items:
        return 0.0
    pairs = sorted(((overlap(p, g), pi, gi) for pi, p in enumerate(pred_items)
                    for gi, g in enumerate(gt_items) if overlap(p, g) >= GT_MATCH_JACCARD), reverse=True)
    up, ug = set(), set()
    for _, pi, gi in pairs:
        if pi in up or gi in ug:
            continue
        up.add(pi)
        ug.add(gi)
    p, r = len(up) / max(len(pred_items), 1), len(ug) / max(len(gt_items), 1)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def gt_similarity(pred_text, gt_text):
    gt_items = split_gt(gt_text)
    if not gt_items:
        return None
    pred = split_bullets(pred_text)
    if not pred:
        return 0.0
    return _clamp(0.5 * bullet_match_f1(pred, gt_items) + 0.5 * rouge_l_f1(" ".join(pred), " ".join(gt_items)))


def route_gt_to_specialties(gt_items):
    routes = defaultdict(list)
    for item in gt_items:
        low = item.lower()
        best, hits = "other", 0
        for spec, kws in SPECIALTY_KEYWORDS.items():
            h = sum(1 for kw in kws if kw in low)
            if h > hits:
                hits, best = h, spec
        routes[best].append(item)
    return routes


def llm_gt_align(judge: K.Judge, worker_output, gt_slice_text, specialty):
    if not judge.available and not judge.cache.exists():
        return None
    obj = judge.json_call(f"align_{specialty}", ALIGN_SYSTEM,
                          align_prompt(specialty, gt_slice_text, worker_output), max_tokens=60)
    if "alignment" not in obj:
        return None
    try:
        return _clamp(float(obj["alignment"]))
    except Exception:
        return None


def worker_gt_alignment(judge, worker_output, gt_slice_items, specialty, use_llm=True):
    if not gt_slice_items:
        return None
    pred = split_bullets(worker_output)
    gt_text = "\n".join(gt_slice_items)
    rouge = rouge_l_f1(" ".join(pred), gt_text)
    lexical = bullet_match_f1(pred, gt_slice_items)
    llm = llm_gt_align(judge, worker_output, gt_text, specialty) if (use_llm and judge is not None) else None
    w = WORKER_GT_WEIGHTS
    if llm is None:
        return _clamp((w["rouge_l"] * rouge + w["lexical"] * lexical) / (w["rouge_l"] + w["lexical"]))
    return _clamp((w["llm_align"] * llm + w["rouge_l"] * rouge + w["lexical"] * lexical) / sum(w.values()))


def blend(stepwise_total, gt_sim, sw=0.70, gw=0.30):
    if gt_sim is None:
        return round(stepwise_total, 4)
    return round(_clamp((sw * stepwise_total + gw * gt_sim) / (sw + gw)), 4)


# --------------------------------------------------------- per-paper scoring
class StepwiseScorer:
    def __init__(self, judge, use_llm=True, stepwise_weight=0.70, gt_weight=0.30,
                 sft_min=0.30, master_sft_min=0.0, dpo_min_gap=0.05):
        self.judge, self.use_llm = judge, use_llm
        self.sw, self.gw = stepwise_weight, gt_weight
        self.sft_min, self.master_sft_min, self.min_gap = sft_min, master_sft_min, dpo_min_gap

    def candidates(self, paper):
        """All scored candidates of one paper (no selection)."""
        pid = paper.get("paper_idx")
        system = paper.get("system", "limagents")
        src = K.source_text_of(paper)
        gt_text = paper.get("ground_truth", "") or ""
        gt_all = split_gt(gt_text)
        routes = route_gt_to_specialties(gt_all) if gt_all else {}
        out = []
        for ro in paper.get("rollouts", []):
            if ro.get("error") or not ro.get("samples"):
                continue
            mode, rid = ro.get("mode"), ro.get("rollout_id")
            for name, tr in K.extract_worker_final(ro).items():
                sc = (score_worker(tr["search"], tr["retrieve"], tr["synthesize"], src) if tr["has_hops"]
                      else score_synth_only(tr["synthesize"], src))
                spec = K.specialty_of(system, name)
                gsim = worker_gt_alignment(self.judge, tr["synthesize"], routes.get(spec) or gt_all, spec, self.use_llm)
                out.append({"paper_idx": pid, "rollout_id": rid, "mode": mode, "agent_role": "worker",
                            "agent_name": name, "output": tr["synthesize"], "system": system,
                            "stepwise_score": sc["total"], "gt_similarity": gsim,
                            "score": blend(sc["total"], gsim, self.sw, self.gw), "score_detail": sc,
                            "_tr": tr})
            for role, turn, agent in (("master", "synthesis", "Master_Agent"),
                                      ("leader", "handoff_to_master", "Leader_Agent")):
                s = K.find_sample(ro, role, turn)
                if s and (s.get("output") or "").strip():
                    sc = score_synth_only(s["output"], src)
                    gsim = gt_similarity(s["output"], gt_text)
                    out.append({"paper_idx": pid, "rollout_id": rid, "mode": mode, "agent_role": role,
                                "agent_name": agent, "output": s["output"], "system": system,
                                "input_messages": s.get("input_messages", []),
                                "stepwise_score": sc["total"], "gt_similarity": gsim,
                                "score": blend(sc["total"], gsim, self.sw, self.gw), "score_detail": sc})
        return out

    def score_paper(self, paper):
        cands = self.candidates(paper)
        return select_from_candidates(paper, cands, self.sft_min, self.master_sft_min, self.min_gap)


def select_from_candidates(paper, cands, worker_sft_min, master_sft_min, min_gap, chosen_floor=0.0,
                           sft_top_k_by_role=None):
    pid = paper.get("paper_idx")
    system = paper.get("system", "limagents")
    sft, dpo = [], []
    groups = defaultdict(list)
    for c in cands:
        groups[(c["agent_role"], c["agent_name"])].append(c)
    k = sft_top_k_by_role or {}
    for (role, name), g in groups.items():
        if role == "worker":
            tr = g[0].get("_tr")
            prompt = K.canonical_worker_prompt(system, name, paper, tr)
            K.select(g, prompt, "worker", name, pid, sft, dpo, min_score=worker_sft_min, min_gap=min_gap,
                     chosen_floor=chosen_floor, sft_top_k=k.get("worker", 1))
        else:
            K.sft_best(g, role, name, pid, sft, min_score=master_sft_min if role == "master" else 0.0)
    clean = [{kk: vv for kk, vv in c.items() if kk != "_tr"} for c in cands]
    return clean, sft, dpo
