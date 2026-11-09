"""
Worker DPO on top of the worker SFT adapter — train_dpo.py / train_worker_dpo.py /
train_dpo_3seeds.py.

    python -m scilim.train.dpo --base_model M --sft_adapter .../worker_sft/final \
        --dpo_data select/dpo_dataset_worker.json --out_dir .../worker_dpo [--seed 0] [--four_bit]

* SFT LoRA is merged into the base (in memory for bf16, or once to disk and
  then loaded in NF4 for --four_bit), a fresh DPO LoRA is applied with
  get_peft_model (peft_config NOT passed to DPOTrainer -> no double wrap), and
  the reference policy is the adapter-disabled model = base+SFT (no 2nd copy).
* LR 5e-6 (LoRA-appropriate; 5e-7 barely moves a LoRA policy).
* At inference the SFT and DPO LoRAs are stacked into one adapter by
  `python -m scilim.serve.prepare` so vLLM serves base + SFT + DPO.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .common import (load_causal_lm, load_tokenizer, lora_targets, make_config, merge_adapter_to_disk,
                     patch_awq, render_prompt)


def load_pairs(path, tok, force_merge, max_prompt):
    from datasets import Dataset
    recs = json.load(open(path))
    eos = tok.eos_token or ""
    rows = []
    for r in recs:
        pm = r.get("prompt") or r.get("input_messages") or []
        ch, rj = (r.get("chosen") or "").strip(), (r.get("rejected") or "").strip()
        if not pm or not ch or not rj or ch == rj:
            continue
        if isinstance(pm, str):
            pm = [{"role": "user", "content": pm}]
        # TRL's DPO tokenisation appends EOS itself -> strip any existing one (no double EOS)
        ch, rj = (ch.removesuffix(eos).rstrip(), rj.removesuffix(eos).rstrip()) if eos else (ch, rj)
        if not ch or not rj or ch == rj:
            continue
        rows.append({"prompt": render_prompt(tok, pm, force_merge), "chosen": ch, "rejected": rj})
    print(f"[DPO] {len(rows)} valid pairs from {path}")
    if rows:
        lens = [len(tok(x["prompt"], add_special_tokens=False)["input_ids"]) for x in rows[:64]]
        print(f"[DPO] prompt tokens mean={sum(lens)//len(lens)} max={max(lens)} >{max_prompt}: "
              f"{sum(l > max_prompt for l in lens)}")
    return Dataset.from_list(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True)
    p.add_argument("--sft_adapter", required=True)
    p.add_argument("--dpo_data", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--loss_type", default="sigmoid")
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--max_prompt_len", type=int, default=6656)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--eval_split", type=float, default=0.0, help="DPO eval casts logits to fp32 (OOM risk)")
    p.add_argument("--four_bit", action="store_true")
    p.add_argument("--merged_base_dir", default=None, help="where the SFT-merged base is cached (--four_bit)")
    p.add_argument("--merge_system", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_pairs", type=int, default=10)
    a = p.parse_args(argv)

    patch_awq()
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    from trl import DPOConfig, DPOTrainer
    set_seed(a.seed)
    final = os.path.join(a.out_dir, "final")
    if not os.path.exists(a.sft_adapter):
        raise SystemExit(f"SFT adapter missing: {a.sft_adapter} — run SFT first")
    os.makedirs(a.out_dir, exist_ok=True)
    tok = load_tokenizer(a.base_model)
    ds = load_pairs(a.dpo_data, tok, a.merge_system, a.max_prompt_len)
    if len(ds) < a.min_pairs:
        print(f"[skip] only {len(ds)} pairs (< {a.min_pairs})")
        return
    if a.four_bit:
        merged = merge_adapter_to_disk(a.base_model, a.sft_adapter,
                                       a.merged_base_dir or os.path.join(os.path.dirname(a.sft_adapter.rstrip("/")),
                                                                         "merged_base"))
        model = load_causal_lm(merged, dtype=torch.bfloat16, four_bit=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = load_causal_lm(a.base_model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, a.sft_adapter).merge_and_unload()
    model.config.use_cache = False
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                                             target_modules=lora_targets(model), task_type=TaskType.CAUSAL_LM,
                                             bias="none"))
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    if a.eval_split > 0 and len(ds) > 20:
        sp = ds.train_test_split(test_size=a.eval_split, seed=42)
        tr, ev = sp["train"], sp["test"]
    else:
        tr, ev = ds, None
    cfg = make_config(
        DPOConfig, output_dir=a.out_dir, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.1, weight_decay=0.01, beta=a.beta, loss_type=a.loss_type, max_length=a.max_seq_len,
        max_prompt_length=a.max_prompt_len, truncation_mode="keep_end", bf16=True, bf16_full_eval=True,
        logging_steps=5, save_steps=50, save_total_limit=2, eval_strategy="steps" if ev is not None else "no",
        eval_steps=50 if ev is not None else None, load_best_model_at_end=ev is not None, report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, max_grad_norm=1.0,
        remove_unused_columns=False, seed=a.seed, data_seed=a.seed,
        optim="paged_adamw_8bit" if a.four_bit else "adamw_torch")
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=tr, eval_dataset=ev,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(final)
    tok.save_pretrained(final)
    json.dump({"sft_adapter": os.path.abspath(a.sft_adapter), "base_model": a.base_model, "seed": a.seed,
               "four_bit": a.four_bit, "dpo_data": os.path.abspath(a.dpo_data)},
              open(os.path.join(final, "scilim_dpo_manifest.json"), "w"), indent=2)
    print(f"DPO adapter -> {final}")


if __name__ == "__main__":
    main()
