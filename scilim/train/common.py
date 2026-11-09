"""Shared training helpers (model loading, LoRA targets, chat rendering)."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

LORA_TARGET_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def patch_awq():
    try:
        from ..utils import awq_compat
        awq_compat.patch_activations(verbose=True)
    except Exception:
        pass


def _dtype_kw(dtype):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def load_causal_lm(path, dtype=None, four_bit=False, device_map="auto", attn_implementation=None):
    """Load a causal LM. Falls back to the image-text-to-text class for
    multimodal checkpoints (Gemma-3 27B, Mistral-Small-3.1) — only the language
    model is used/trained."""
    import torch
    from transformers import AutoModelForCausalLM
    patch_awq()
    dtype = dtype or torch.bfloat16
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True, device_map=device_map, **_dtype_kw(dtype))
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kw)
    except (ValueError, KeyError) as e:
        print(f"[load] AutoModelForCausalLM failed ({str(e)[:120]}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(path, **kw)


def lora_targets(model):
    """Restrict LoRA to the language model for multimodal checkpoints."""
    names = [n for n, _ in model.named_modules()]
    if any(".language_model." in f".{n}." or n.endswith("language_model") for n in names) and \
            any(("vision" in n or "visual" in n) for n in names):
        return r".*language_model.*\.(" + "|".join(LORA_TARGET_NAMES) + r")$"
    return list(LORA_TARGET_NAMES)


def make_config(cls, **kw):
    """Build a TRL/HF config, dropping/renaming keys this TRL version lacks."""
    fields = {f.name for f in dataclasses.fields(cls)}
    if "max_length" in kw and "max_length" not in fields and "max_seq_length" in fields:
        kw["max_seq_length"] = kw.pop("max_length")
    if "eval_strategy" in kw and "eval_strategy" not in fields and "evaluation_strategy" in fields:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in fields]
    if dropped:
        print(f"[config] {cls.__name__}: ignoring unsupported {dropped}")
    return cls(**{k: v for k, v in kw.items() if k in fields})


def merge_system(messages):
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_txt, rest = messages[0]["content"], messages[1:]
    if rest and rest[0].get("role") == "user":
        return [{"role": "user", "content": f"{sys_txt}\n\n{rest[0]['content']}"}] + rest[1:]
    return [{"role": "user", "content": sys_txt}] + rest


def render_prompt(tokenizer, messages, force_merge=False):
    """Chat-template a prompt (thinking off for Qwen3); tolerate templates
    without a system role or without the enable_thinking kwarg."""
    for attempt in ((merge_system(messages) if force_merge else messages), merge_system(messages)):
        for kw in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(attempt, tokenize=False, add_generation_prompt=True, **kw)
            except Exception:
                continue
    raise RuntimeError("apply_chat_template failed for every fallback")


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def merge_adapter_to_disk(base, adapter, out_dir):
    """CPU-merge a LoRA into the base once (needed to QLoRA-train DPO on top of SFT)."""
    out = Path(out_dir)
    marker = out / "SFT_MERGE_COMPLETE.json"
    ident = {"base": str(Path(base).resolve()), "adapter": str(Path(adapter).resolve()),
             "adapter_mtime": os.path.getmtime(Path(adapter) / "adapter_config.json")}
    if marker.exists() and json.loads(marker.read_text()) == ident:
        print(f"[merge] reusing {out}")
        return str(out)
    import torch
    from peft import PeftModel
    print(f"[merge] merging {adapter} into {base} on CPU -> {out}")
    m = load_causal_lm(base, dtype=torch.bfloat16, device_map={"": "cpu"})
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload(safe_merge=True)
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    load_tokenizer(base).save_pretrained(out)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
    except Exception:
        pass
    marker.write_text(json.dumps(ident))
    del m
    return str(out)
