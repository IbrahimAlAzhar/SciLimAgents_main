"""
JSD reward — teacher as a DISTRIBUTIONAL judge (jsd_reward_select.py).

    reward = 1 - mean_t JSD( P_teacher(.|x_<t) || P_student(.|x_<t) )   (log2, in [0,1])

Teacher-forced forward passes over prompt + existing student completion; no
sampling, no gradients. Full-vocab logits need `transformers` (vLLM caps
logprobs at ~20). Teacher and student must share the id->token mapping; up to
5% of mismatching id slots (repurposed control tokens) are masked.

Teacher options
    --teacher <path>   a bigger model of the same family (e.g. Qwen3-32B-AWQ for
                       Qwen3-8B). TEACHER_4BIT/--teacher_4bit quantizes an
                       unquantized teacher to NF4 on load.
    --teacher self     the student BASE model is the teacher and the student is
                       base + --student_adapter (measures how far the trained
                       adapter drifts from the base distribution). Requires an
                       adapter.

CAVEAT: JSD measures agreement with the teacher's distribution, not factual
correctness. Use --blend_stepwise to mix in grounding.
"""
from __future__ import annotations

import gc
import math
import os
from collections import defaultdict
from typing import Optional

from . import common as K
from .stepwise import select_from_candidates


def jsd_from_logits(teacher_logits, student_logits, chunk: int = 64) -> float:
    import torch.nn.functional as F
    T = teacher_logits.shape[0]
    if T == 0:
        return float("nan")
    total, ln2 = 0.0, math.log(2.0)
    for i in range(0, T, chunk):
        tl, sl = teacher_logits[i:i + chunk].float(), student_logits[i:i + chunk].float()
        lp, lq = F.log_softmax(tl, dim=-1), F.log_softmax(sl, dim=-1)
        p, q = lp.exp(), lq.exp()
        lm = (0.5 * (p + q)).clamp_min(1e-12).log()
        jsd = 0.5 * ((p * (lp - lm)).sum(-1) + (q * (lq - lm)).sum(-1)) / ln2
        total += jsd.clamp(0.0, 1.0).sum().item()
    return total / T


def jsd_per_token(teacher_logits, student_logits):
    """Differentiable per-token JSD (bits) — used as a dense GRPO reward / loss."""
    import torch.nn.functional as F
    lp, lq = F.log_softmax(teacher_logits.float(), -1), F.log_softmax(student_logits.float(), -1)
    p, q = lp.exp(), lq.exp()
    lm = (0.5 * (p + q)).clamp_min(1e-12).log()
    return (0.5 * ((p * (lp - lm)).sum(-1) + (q * (lq - lm)).sum(-1)) / math.log(2.0)).clamp(0, 1)


def mean_nll(logits, targets):
    import torch.nn.functional as F
    if logits.shape[0] == 0:
        return float("nan")
    return F.cross_entropy(logits.float(), targets, reduction="mean").item()


def selftest():
    import torch
    V = 5000
    torch.manual_seed(0)
    a = torch.randn(8, V)
    print(f"identical  JSD = {jsd_from_logits(a, a):.6f} (expect 0)")
    p = torch.full((4, V), -50.0); p[:, 0] = 50.0
    q = torch.full((4, V), -50.0); q[:, 1] = 50.0
    print(f"disjoint   JSD = {jsd_from_logits(p, q):.6f} (expect 1)")
    b = torch.randn(8, V)
    assert abs(jsd_from_logits(a, b) - jsd_from_logits(b, a)) < 1e-6
    assert abs(jsd_from_logits(a, b) - jsd_from_logits(a, b, chunk=1000)) < 1e-5
    assert jsd_from_logits(a, a + 0.05 * torch.randn(8, V)) < jsd_from_logits(a, a + 5 * torch.randn(8, V))
    print("JSD SELFTEST PASSED")


def merge_system(messages):
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_txt, rest = messages[0]["content"], messages[1:]
    if rest and rest[0].get("role") == "user":
        return [{"role": "user", "content": f"{sys_txt}\n\n{rest[0]['content']}"}] + rest[1:]
    return [{"role": "user", "content": sys_txt}] + rest


def render_prompt(tok, messages, force_merge=False):
    for attempt in ((merge_system(messages) if force_merge else messages), merge_system(messages)):
        for kw in ({"enable_thinking": False}, {}):
            try:
                return tok.apply_chat_template(attempt, tokenize=False, add_generation_prompt=True, **kw)
            except Exception:
                continue
    raise RuntimeError("apply_chat_template failed for every fallback")


class JSDScorer:
    def __init__(self, student_base, teacher_path, student_adapter="", teacher_4bit=False,
                 max_prompt_tokens=3072, max_completion_tokens=512, chunk=64, device="auto"):
        try:
            from ..utils import awq_compat
            awq_compat.patch_activations(verbose=True)
        except Exception:
            pass
        import torch
        from transformers import AutoTokenizer
        from ..train.common import load_causal_lm
        self.torch = torch
        self.max_p, self.max_c, self.chunk = max_prompt_tokens, max_completion_tokens, chunk
        if teacher_path in ("self", "", None):
            if not student_adapter:
                raise SystemExit("--teacher self needs --student_adapter (otherwise JSD(base, base) == 0).")
            teacher_path = student_base
            print("[jsd] teacher = student BASE, student = base + adapter")
        self.tok = AutoTokenizer.from_pretrained(student_base, trust_remote_code=True)
        t_tok = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
        self.bad_ids = check_vocab(self.tok, t_tok)
        self.teacher = load_causal_lm(teacher_path, dtype=torch.float16, four_bit=teacher_4bit, device_map=device)
        self.teacher.eval()
        self.student = load_causal_lm(student_base, dtype=torch.bfloat16, device_map=device)
        if student_adapter:
            from peft import PeftModel
            self.student = PeftModel.from_pretrained(self.student, student_adapter).merge_and_unload()
        self.student.eval()
        self.force_merge = os.environ.get("MERGE_SYSTEM", "0") == "1"
        self._bad_idx = None

    def build_ids(self, messages, completion):
        ptxt = render_prompt(self.tok, messages, self.force_merge)
        p = self.tok(ptxt, add_special_tokens=False)["input_ids"]
        c = self.tok(completion.strip(), add_special_tokens=False)["input_ids"]
        if len(p) > self.max_p:
            p = p[:1] + p[-(self.max_p - 1):]
        return p, c[: self.max_c]

    def _logits(self, model, ids, n):
        with self.torch.no_grad():
            try:
                lg = model(ids, logits_to_keep=n + 1).logits[0][-(n + 1):-1, :]
            except TypeError:
                lg = model(ids).logits[0][-(n + 1):-1, :]
        return lg

    def score(self, messages, completion):
        torch = self.torch
        p, c = self.build_ids(messages, completion)
        if len(c) < 5:
            return {"jsd": None, "reward": None, "teacher_nll": None, "student_nll": None, "n_tokens": len(c)}
        dev = next(self.teacher.parameters()).device
        ids = torch.tensor([p + c], device=dev)
        n = len(c)
        tgt = torch.tensor(c, device=dev)
        t_lg = self._logits(self.teacher, ids, n)
        s_lg = self._logits(self.student, ids.to(next(self.student.parameters()).device), n).to(t_lg.device)
        V = min(t_lg.shape[-1], s_lg.shape[-1])      # padded embedding rows differ across sizes
        t_lg, s_lg = t_lg[:, :V], s_lg[:, :V]
        t_nll, s_nll = mean_nll(t_lg, tgt), mean_nll(s_lg, tgt)
        if self.bad_ids:
            if self._bad_idx is None:
                self._bad_idx = torch.tensor([i for i in self.bad_ids if i < V], device=t_lg.device)
            neg = torch.finfo(t_lg.dtype).min
            t_lg[:, self._bad_idx] = neg
            s_lg[:, self._bad_idx] = neg
        j = jsd_from_logits(t_lg, s_lg, self.chunk)
        del t_lg, s_lg, ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"jsd": round(j, 6), "reward": round(1 - j, 6), "teacher_nll": round(t_nll, 4),
                "student_nll": round(s_nll, 4), "n_tokens": n}


def check_vocab(s, t, max_frac=5.0):
    if s.vocab_size != t.vocab_size:
        raise SystemExit(f"Tokenizer mismatch: student vocab {s.vocab_size} != teacher {t.vocab_size}. "
                         "JSD needs a shared vocabulary — choose a teacher from the same family.")
    sv, tv = s.get_vocab(), t.get_vocab()
    if sv == tv:
        print(f"[jsd] shared vocab confirmed ({s.vocab_size})")
        return []
    inv_s, inv_t = {v: k for k, v in sv.items()}, {v: k for k, v in tv.items()}
    bad = [i for i in range(s.vocab_size) if inv_s.get(i) != inv_t.get(i)]
    frac = 100.0 * len(bad) / s.vocab_size
    if frac > max_frac:
        raise SystemExit(f"{frac:.1f}% of the vocabulary disagrees — not interchangeable; do not run JSD.")
    print(f"[jsd] masking {len(bad)} id slots ({frac:.2f}%) that map to different tokens")
    return bad


class JSDSelector:
    def __init__(self, scorer: JSDScorer, stepwise_lookup=None, blend_stepwise=0.0, sft_min=0.0, min_gap=0.02):
        self.s, self.lookup = scorer, stepwise_lookup or {}
        self.blend, self.sft_min, self.min_gap = blend_stepwise, sft_min, min_gap

    def _blend(self, r, sw):
        if self.blend <= 0 or sw is None:
            return r
        return round((1 - self.blend) * r + self.blend * sw, 6)

    def score_paper(self, paper):
        pid, system = paper.get("paper_idx"), paper.get("system", "limagents")
        cands = []
        for ro in paper.get("rollouts", []):
            if ro.get("error") or not ro.get("samples"):
                continue
            mode, rid = ro.get("mode"), ro.get("rollout_id")
            for name, tr in K.extract_worker_final(ro).items():
                text = (tr.get("synthesize") or "").strip()
                if not text:
                    continue
                m = self.s.score(K.canonical_worker_prompt(system, name, paper, tr), text)
                if m["reward"] is None:
                    continue
                sw = self.lookup.get((pid, "worker", name, rid))
                cands.append({"paper_idx": pid, "rollout_id": rid, "mode": mode, "agent_role": "worker",
                              "agent_name": name, "output": text, "system": system, "jsd": m["jsd"],
                              "jsd_reward": m["reward"], "teacher_nll": m["teacher_nll"],
                              "student_nll": m["student_nll"], "n_tokens": m["n_tokens"], "stepwise_score": sw,
                              "score": self._blend(m["reward"], sw),
                              "score_detail": {"total": m["reward"], "steps": {"jsd": m["jsd"],
                                               "teacher_nll": m["teacher_nll"], "student_nll": m["student_nll"]},
                                               "n_units": len(K.split_bullets(text))}, "_tr": tr})
            for role, turn, agent in (("master", "synthesis", "Master_Agent"),
                                      ("leader", "handoff_to_master", "Leader_Agent")):
                s = K.find_sample(ro, role, turn)
                if not s or not (s.get("output") or "").strip() or not s.get("input_messages"):
                    continue
                m = self.s.score(s["input_messages"], s["output"])
                if m["reward"] is None:
                    continue
                cands.append({"paper_idx": pid, "rollout_id": rid, "mode": mode, "agent_role": role,
                              "agent_name": agent, "output": s["output"], "input_messages": s["input_messages"],
                              "system": system, "jsd": m["jsd"], "jsd_reward": m["reward"],
                              "teacher_nll": m["teacher_nll"], "student_nll": m["student_nll"],
                              "n_tokens": m["n_tokens"], "stepwise_score": None, "score": m["reward"],
                              "score_detail": {"total": m["reward"], "steps": {"jsd": m["jsd"]},
                                               "n_units": len(K.split_bullets(s["output"]))}})
        return select_from_candidates(paper, cands, self.sft_min, 0.0, self.min_gap)


def free(obj):
    del obj
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
