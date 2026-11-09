"""
On-policy GRPO for the WORKER agent (worker_grpo.py), any reward model.

Per prompt (paper x worker):
  1. sample --group_size completions from the current policy (SFT-initialised
     LoRA), cycling the original 4 sampling configs (T/top_p/seed offsets);
  2. score each with --reward:
        outcome   LLM-judge composite (F-beta match, grounding, specificity,
                  severity, ROUGE, length/redundancy penalties) — original
        stepwise  synthesize-step grounding blended 0.7/0.3 with GT alignment
        jsd       1 - JSD(teacher || policy) on the completion (teacher-forced)
        combined  weighted mean of the above (--w_outcome/--w_stepwise/--w_jsd)
  3. advantage = (r - mean) / max(std, floor); train only on completions above
     the group mean (+ epsilon) with a PPO-clipped objective + k3 KL to the
     frozen SFT reference.

Prompts come from the rollout JSON (same papers / canonical worker prompts as
DPO), so GRPO and DPO are trained on identical inputs. The saved adapter
contains SFT + GRPO updates (no stacking needed at inference).

Memory: the reference is the SAME base with a frozen copy of the SFT adapter
(--ref_mode adapter, default) instead of a second full model (original).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ..prompts import canonical as C
from ..prompts import novagents as NOV
from ..rewards import common as K
from .common import load_causal_lm, load_tokenizer, patch_awq, render_prompt

ROLLOUT_CONFIGS = [
    {"temperature": 0.7, "top_p": 0.90, "seed_offset": 42},
    {"temperature": 0.9, "top_p": 0.95, "seed_offset": 123},
    {"temperature": 1.1, "top_p": 1.00, "seed_offset": 456},
    {"temperature": 0.5, "top_p": 0.85, "seed_offset": 789},
]


@dataclass
class Example:
    paper_idx: int
    worker_name: str
    specialty_key: str
    paper_text: str
    gt_all: List[str]
    gt_items: List[str]
    messages: list


def load_examples(args, judge) -> List[Example]:
    from ..rewards import stepwise as S
    papers = json.load(open(args.rollout_json))
    if args.limit_papers:
        papers = papers[: args.limit_papers]
    out = []
    outcome = None
    if args.reward in ("outcome", "combined") and judge is not None:
        from ..rewards.outcome import OutcomeScorer
        outcome = OutcomeScorer(judge)
    for p in papers:
        system = p.get("system", "limagents")
        if system != args.system:
            continue
        pid = p.get("paper_idx")
        gt_all = S.split_gt(p.get("ground_truth", "") or "")
        if len((p.get("paper_text") or "")) < 100 or not gt_all:
            continue
        cats = outcome.categorize_gt(pid, gt_all) if outcome else {}
        routes = S.route_gt_to_specialties(gt_all)
        names = ([w[0] for w in C.WORKER_CONFIGS] if system == "limagents"
                 else [n for n, _ in NOV.SPECIALIST_AGENTS])
        for name in names:
            if args.worker_name != "all" and name != args.worker_name:
                continue
            spec = K.specialty_of(system, name)
            gt_sub = (outcome.gt_for_specialty(gt_all, cats, spec) if outcome
                      else (routes.get(spec) or gt_all))
            tr = None
            if system == "novagents":
                ro = next((r for r in p.get("rollouts", []) if r.get("samples")), None)
                tr = K.extract_worker_final(ro).get(name) if ro else None
            msgs = K.canonical_worker_prompt(system, name, p, tr, user_style=args.user_style)
            out.append(Example(pid, name, spec, K.source_text_of(p), gt_all, gt_sub, msgs))
    return out


class RewardFn:
    def __init__(self, args, judge):
        self.a, self.judge = args, judge
        self.outcome = self.jsd = None
        if args.reward in ("outcome", "combined"):
            from ..rewards.outcome import OutcomeScorer
            self.outcome = OutcomeScorer(judge)
        if args.reward in ("jsd", "combined") and args.jsd_teacher:
            from .common import load_causal_lm as L
            import torch
            self.teacher = L(args.jsd_teacher, dtype=torch.float16, four_bit=args.teacher_4bit)
            self.teacher.eval()
            self.jsd = True

    def __call__(self, ex: Example, text: str, tag: str, policy_logits=None, teacher_ids=None) -> dict:
        from ..rewards import stepwise as S
        parts = {}
        if self.outcome is not None:
            parts["outcome"] = self.outcome.score_worker_text(text, ex.gt_items, ex.paper_text, tag)["composite"]
        if self.a.reward in ("stepwise", "combined"):
            sc = S.score_synth_only(text, ex.paper_text)["total"]
            g = S.worker_gt_alignment(self.judge, text, ex.gt_items, ex.specialty_key,
                                      use_llm=self.judge is not None and self.judge.available)
            parts["stepwise"] = S.blend(sc, g)
        if self.jsd and policy_logits is not None and teacher_ids is not None:
            parts["jsd"] = self._jsd_reward(teacher_ids, policy_logits)
        w = {"outcome": self.a.w_outcome, "stepwise": self.a.w_stepwise, "jsd": self.a.w_jsd}
        if self.a.reward != "combined":
            w = {self.a.reward: 1.0}
        num = sum(w.get(k, 0) * v for k, v in parts.items())
        den = sum(w.get(k, 0) for k in parts)
        return {"reward": (num / den) if den else 0.0, **parts}

    def _jsd_reward(self, ids, policy_logits):
        import torch
        from ..rewards.jsd import jsd_from_logits
        n = policy_logits.shape[0]
        with torch.no_grad():
            tid = ids.to(next(self.teacher.parameters()).device)
            try:
                t = self.teacher(tid, logits_to_keep=n + 1).logits[0][:-1, :]
            except TypeError:
                t = self.teacher(tid).logits[0][-(n + 1):-1, :]
        V = min(t.shape[-1], policy_logits.shape[-1])
        return 1.0 - jsd_from_logits(t[:, :V], policy_logits[:, :V].to(t.device))


def completion_logprobs(model, input_ids, attention_mask, loss_mask, prompt_len):
    """Log-probs of completion tokens only. Logits are computed for the
    completion positions alone (logits_to_keep) — full-vocab log-softmax over a
    6k-token prompt x group would need tens of GB."""
    n_keep = input_ids.shape[1] - prompt_len + 1
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=n_keep)
    except TypeError:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, -n_keep:, :][:, :-1, :]
    labels = input_ids[:, prompt_len:]
    m = loss_mask[:, prompt_len:]
    lp = torch_log_softmax_gather(logits, labels)
    return lp * m, m


def torch_log_softmax_gather(logits, labels, chunk=256):
    import torch
    outs = []
    for i in range(0, logits.shape[1], chunk):
        lg = logits[:, i:i + chunk, :].float()
        outs.append(lg.gather(-1, labels[:, i:i + chunk].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1))
    return torch.cat(outs, dim=1)


def masked_mean(v, m):
    return v.sum() / m.sum().clamp_min(1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default="limagents", choices=["limagents", "novagents"])
    p.add_argument("--base_model", required=True)
    p.add_argument("--sft_adapter", required=True, help="worker SFT LoRA (or a full SFT model dir)")
    p.add_argument("--rollout_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--reward", default="outcome", choices=["outcome", "stepwise", "jsd", "combined"])
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--judge_base_url", default=None)
    p.add_argument("--no_llm_judge", action="store_true")
    p.add_argument("--jsd_teacher", default=None)
    p.add_argument("--teacher_4bit", action="store_true")
    p.add_argument("--w_outcome", type=float, default=0.5)
    p.add_argument("--w_stepwise", type=float, default=0.25)
    p.add_argument("--w_jsd", type=float, default=0.25)
    p.add_argument("--worker_name", default="all")
    p.add_argument("--user_style", default="canonical", choices=["canonical", "grpo"])
    p.add_argument("--limit_papers", type=int, default=None)
    p.add_argument("--four_bit", action="store_true")
    p.add_argument("--ref_mode", default="adapter", choices=["adapter", "copy"])
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_length", type=int, default=6144)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--advantage_std_floor", type=float, default=0.05)
    p.add_argument("--above_average_epsilon", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=25)
    a = p.parse_args(argv)

    patch_awq()
    import torch
    from peft import PeftModel
    from torch.nn.utils.rnn import pad_sequence
    from transformers import set_seed

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    random.seed(a.seed)
    judge = None
    if a.reward in ("outcome", "stepwise", "combined"):
        a.cache_dir = str(out / "reward_cache")
        judge = K.make_judge(a)
    tok = load_tokenizer(a.base_model)
    tok.padding_side = "right"
    examples = load_examples(a, judge)
    if not examples:
        raise SystemExit("No GRPO prompts (check --rollout_json / --system).")
    random.Random(a.seed).shuffle(examples)
    print(f"[GRPO] {len(examples)} prompts | reward={a.reward} | group={a.group_size} | kl_beta={a.kl_beta}")

    base = load_causal_lm(a.base_model, dtype=torch.bfloat16, four_bit=a.four_bit,
                          attn_implementation=a.attn_implementation)
    is_peft = (Path(a.sft_adapter) / "adapter_config.json").exists()
    if not is_peft:
        raise SystemExit("--sft_adapter must be a LoRA adapter directory")
    if a.four_bit:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    policy = PeftModel.from_pretrained(base, a.sft_adapter, adapter_name="default", is_trainable=True)
    reference = None
    if a.ref_mode == "adapter":
        policy.load_adapter(a.sft_adapter, adapter_name="ref", is_trainable=False)
        policy.set_adapter("default")
    else:
        reference = PeftModel.from_pretrained(load_causal_lm(a.base_model, dtype=torch.bfloat16,
                                                             four_bit=a.four_bit), a.sft_adapter)
        reference.eval()
        for prm in reference.parameters():
            prm.requires_grad_(False)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.enable_input_require_grads()
    trainable = [prm for n, prm in policy.named_parameters() if prm.requires_grad and ".ref." not in n]
    print(f"Trainable params: {sum(x.numel() for x in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=a.learning_rate, weight_decay=a.weight_decay)
    rf = RewardFn(a, judge)
    if rf.jsd:
        from ..rewards.jsd import check_vocab
        check_vocab(tok, load_tokenizer(a.jsd_teacher))
    dev = next(policy.parameters()).device

    def ref_logps(ids, am, lm, P):
        with torch.no_grad():
            if reference is not None:
                return completion_logprobs(reference, ids, am, lm, P)[0]
            policy.set_adapter("ref")
            try:
                return completion_logprobs(policy, ids, am, lm, P)[0]
            finally:
                policy.set_adapter("default")
                for n_, prm in policy.named_parameters():   # keep only "default" trainable
                    if ".ref." in n_:
                        prm.requires_grad_(False)

    log_path, roll_path = out / "grpo_train_log.jsonl", out / "grpo_rollouts.jsonl"
    step = 0
    for epoch in range(a.num_epochs):
        for ex_i, ex in enumerate(examples):
            if a.max_steps is not None and step >= a.max_steps:
                break
            ptxt = render_prompt(tok, ex.messages)
            prompt_ids = tok(ptxt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if len(prompt_ids) > a.max_prompt_length:
                prompt_ids = prompt_ids[-a.max_prompt_length:]
            # ---- sample group (KV cache on for generation) ----
            policy.eval()
            policy.config.use_cache = True
            comps = []
            inp = prompt_ids.unsqueeze(0).to(dev)
            with torch.no_grad():
                for g in range(a.group_size):
                    cfg = ROLLOUT_CONFIGS[g % len(ROLLOUT_CONFIGS)]
                    torch.manual_seed(a.seed + step * 1009 + cfg["seed_offset"] + g)
                    gen = policy.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True,
                                          temperature=cfg["temperature"], top_p=cfg["top_p"],
                                          max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id,
                                          eos_token_id=tok.eos_token_id, use_cache=True)
                    comps.append(tok.decode(gen[0, inp.shape[1]:], skip_special_tokens=True).strip())
            policy.config.use_cache = False
            # ---- batch tensors ----
            seqs = []
            for c in comps:
                cid = tok(c, add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
                seqs.append(torch.tensor(prompt_ids.tolist() + cid, dtype=torch.long))
            ids = pad_sequence(seqs, batch_first=True, padding_value=tok.pad_token_id).to(dev)
            am = torch.zeros_like(ids)
            lm = torch.zeros_like(ids, dtype=torch.bool)
            for i, s in enumerate(seqs):
                am[i, :len(s)] = 1
                lm[i, len(prompt_ids):len(s)] = True
            # ---- rewards ----
            scores = []
            pol_logits_for_jsd = None
            for gi, c in enumerate(comps):
                extra = {}
                if rf.jsd:
                    n = int(lm[gi].sum())
                    with torch.no_grad():
                        try:
                            pl = policy(ids[gi:gi + 1, :len(seqs[gi])], logits_to_keep=n + 1).logits[0][:-1, :]
                        except TypeError:
                            pl = policy(ids[gi:gi + 1, :len(seqs[gi])]).logits[0][-(n + 1):-1, :]
                    extra = {"policy_logits": pl, "teacher_ids": ids[gi:gi + 1, :len(seqs[gi])]}
                scores.append(rf(ex, c, f"p{ex.paper_idx}_{ex.worker_name}_step{step}_g{gi}", **extra))
            rewards = [s["reward"] for s in scores]
            avg = sum(rewards) / len(rewards)
            std = math.sqrt(sum((r - avg) ** 2 for r in rewards) / max(len(rewards), 1))
            sel = [i for i, r in enumerate(rewards) if r > avg + a.above_average_epsilon]
            rec = {"step": step, "epoch": epoch, "paper_idx": ex.paper_idx, "worker_name": ex.worker_name,
                   "rewards": rewards, "reward_parts": scores, "avg_reward": avg, "std_reward": std,
                   "selected_indices": sel, "completions": comps}
            if not sel:
                rec["skipped"] = True
                with roll_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                step += 1
                continue
            adv = torch.tensor([(rewards[i] - avg) / max(std, a.advantage_std_floor) for i in sel],
                               dtype=torch.float32, device=dev).view(-1, 1)
            sids, sam, slm = ids[sel], am[sel], lm[sel]
            policy.train()
            with torch.no_grad():
                old_lp, _ = completion_logprobs(policy, sids, sam, slm, len(prompt_ids))
            r_lp = ref_logps(sids, sam, slm, len(prompt_ids))
            cur_lp, tmask = completion_logprobs(policy, sids, sam, slm, len(prompt_ids))
            ratio = torch.exp(cur_lp - old_lp.detach()).clamp(0.0, 10.0)
            surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - a.clip_range, 1 + a.clip_range) * adv)
            pol_loss = -masked_mean(surr, tmask)
            lr_ = r_lp - cur_lp
            kl = masked_mean(torch.exp(lr_) - lr_ - 1.0, tmask)   # k3 estimator (as original)
            loss = pol_loss + a.kl_beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            stats = {"policy_loss": float(pol_loss), "kl": float(kl), "loss": float(loss)}
            with roll_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "epoch": epoch, "avg_reward": avg, "max_reward": max(rewards),
                                    "min_reward": min(rewards), "selected": len(sel), **stats}) + "\n")
            if step % a.logging_steps == 0:
                print(f"step={step} avg_r={avg:.4f} max={max(rewards):.4f} sel={len(sel)} "
                      f"loss={stats['loss']:.4f} kl={stats['kl']:.6f}", flush=True)
            if step > 0 and step % a.save_steps == 0:
                policy.save_pretrained(out / f"checkpoint-{step}", selected_adapters=["default"])
            step += 1
    final = out / "final"
    policy.save_pretrained(final, selected_adapters=["default"])
    tok.save_pretrained(final)
    print(f"GRPO worker adapter (SFT+GRPO) -> {final}")


if __name__ == "__main__":
    main()
