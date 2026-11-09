"""
Role-parameterized SFT (LoRA / QLoRA) — train_sft.py (+ train_{worker,leader,master}_sft.py).

    python -m scilim.train.sft --role worker --base_model M --sft_data select/sft_dataset_worker.json \
        --out_dir adapters/limagents/stepwise/worker_sft

Prompts are pre-rendered with the model's chat template (Qwen3 thinking OFF),
so training formatting == vLLM inference formatting; completion-only loss.
Leader/master (≈1 sample per paper) default to 5 epochs; when there are too
few steps for eval, everything is used for training and checkpoints are per epoch.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .common import load_causal_lm, load_tokenizer, lora_targets, make_config, patch_awq, render_prompt


def load_sft(path, tok, prerender, force_merge, max_len, role):
    from datasets import Dataset
    recs = json.load(open(path))
    print(f"Loaded {len(recs)} SFT records from {path}")
    rows, skipped = [], 0
    eos = tok.eos_token or ""
    for r in recs:
        msgs = r.get("messages") or []
        if not msgs or not any(m["role"] == "assistant" for m in msgs) or not str(msgs[-1].get("content", "")).strip():
            skipped += 1
            continue
        if prerender:
            ai = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
            rows.append({"prompt": render_prompt(tok, msgs[:ai], force_merge),
                         "completion": msgs[ai]["content"].strip() + eos})
        else:
            rows.append({"messages": msgs})
    print(f"After filtering: {len(rows)} valid ({skipped} skipped)")
    if not rows:
        raise SystemExit(f"No usable SFT samples in {path}")
    if len(rows) < 50:
        print(f"[WARN] only {len(rows)} {role} samples — consider more epochs")
    if prerender:
        lens = [len(tok(x["prompt"] + x["completion"], add_special_tokens=False)["input_ids"]) for x in rows[:64]]
        print(f"Token length (first {len(lens)}): mean={sum(lens)//len(lens)} max={max(lens)} >{max_len}: "
              f"{sum(l > max_len for l in lens)}")
    return Dataset.from_list(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", default="worker", choices=["worker", "leader", "master"])
    p.add_argument("--base_model", required=True)
    p.add_argument("--sft_data", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_split", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--four_bit", action="store_true", help="QLoRA (NF4) — use for 24B/27B models")
    p.add_argument("--no_prerender", action="store_true")
    p.add_argument("--merge_system", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_samples", type=int, default=5)
    a = p.parse_args(argv)
    epochs = a.epochs or (5 if a.role in ("leader", "master") else 3)

    patch_awq()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    set_seed(a.seed)
    final = os.path.join(a.out_dir, "final")
    print("=" * 60 + f"\n{a.role.upper()} SFT — {a.base_model}\n  data {a.sft_data}\n  out  {a.out_dir}\n"
          f"  epochs={epochs} bs={a.batch_size} accum={a.grad_accum} lr={a.lr} max_len={a.max_seq_len} "
          f"r={a.lora_r} 4bit={a.four_bit}\n" + "=" * 60)
    os.makedirs(a.out_dir, exist_ok=True)
    tok = load_tokenizer(a.base_model)
    ds = load_sft(a.sft_data, tok, not a.no_prerender, a.merge_system, a.max_seq_len, a.role)
    if len(ds) < a.min_samples:
        print(f"[skip] only {len(ds)} samples (< {a.min_samples})")
        return
    model = load_causal_lm(a.base_model, dtype=torch.bfloat16, four_bit=a.four_bit)
    model.config.use_cache = False
    if a.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                                             target_modules=lora_targets(model), task_type=TaskType.CAUSAL_LM,
                                             bias="none"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    eff = max(1, a.batch_size * a.grad_accum)
    total_steps = max(1, -(-len(ds) // eff)) * epochs
    if a.eval_split > 0 and len(ds) > 20 and total_steps >= 2 * a.save_steps:
        sp = ds.train_test_split(test_size=a.eval_split, seed=42)
        tr, ev, strat = sp["train"], sp["test"], "steps"
    else:
        tr, ev, strat = ds, None, "epoch"
        print(f"Train on all {len(tr)} samples ({total_steps} steps); checkpoint per epoch")
    cfg = make_config(
        SFTConfig, output_dir=a.out_dir, num_train_epochs=epochs, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=a.grad_accum, eval_accumulation_steps=4,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=a.warmup_ratio, weight_decay=0.01,
        bf16=True, bf16_full_eval=True, logging_steps=10, save_strategy=strat,
        save_steps=a.save_steps if strat == "steps" else 500, save_total_limit=2,
        eval_strategy=strat if ev is not None else "no", eval_steps=a.save_steps if ev is not None else None,
        load_best_model_at_end=ev is not None, metric_for_best_model="eval_loss" if ev is not None else None,
        report_to="none", gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, dataloader_num_workers=2, remove_unused_columns=False, max_length=a.max_seq_len,
        completion_only_loss=True, seed=a.seed)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=tr, eval_dataset=ev, processing_class=tok)
    trainer.train()
    trainer.save_model(final)
    tok.save_pretrained(final)
    print(f"{a.role} SFT adapter -> {final}")


if __name__ == "__main__":
    main()
