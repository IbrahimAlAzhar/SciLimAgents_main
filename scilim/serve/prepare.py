"""
Prepare trained adapters for vLLM serving and print `--lora-modules` args.

DPO adapters are trained on top of the SFT-merged base, so at inference the
worker needs base + SFT + DPO. vLLM serves ONE LoRA per request, so the two
LoRAs are composed exactly (prepare_vllm_adapters.py):

    B = [B_sft, B_dpo],  A = [s_sft * A_sft ; s_dpo * A_dpo]   =>  B@A = dSFT + dDPO
    rank = r_sft + r_dpo, alpha = rank (scale 1)

CPU-only, no model weights loaded. Layout produced:

    outputs/<model>/serve_adapters/<system>-<reward>-<role>-<algo>/   (one dir per served name)

    python -m scilim.serve.prepare --output_root outputs/qwen3_8b            # prepare all
    python -m scilim.serve.prepare --output_root outputs/qwen3_8b --print_args
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .. import config as CFG

UNSUPPORTED = ("rank_pattern", "alpha_pattern", "modules_to_save", "use_dora", "use_rslora", "fan_in_fan_out",
               "lora_bias")


def _adapter_ok(d: Path) -> bool:
    return (d / "adapter_config.json").exists() and (
        (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists())


def _load(d: Path):
    if (d / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        return load_file(str(d / "adapter_model.safetensors"), device="cpu")
    import torch
    return torch.load(str(d / "adapter_model.bin"), map_location="cpu")


def stack_lora(sft_dir: Path, dpo_dir: Path, out_dir: Path, base_model: str, max_rank: int = 64):
    import torch
    from safetensors.torch import save_file
    sc = json.loads((sft_dir / "adapter_config.json").read_text())
    dc = json.loads((dpo_dir / "adapter_config.json").read_text())
    for c in (sc, dc):
        for f in UNSUPPORTED:
            if c.get(f):
                raise ValueError(f"unsupported LoRA option {f}")
        if c.get("bias", "none") != "none" or c.get("peft_type") != "LORA":
            raise ValueError("expected bias-free LoRA")
    sft, dpo = _load(sft_dir), _load(dpo_dir)
    if set(sft) != set(dpo):
        raise ValueError("SFT and DPO adapters must target the same modules")
    rank = int(sc["r"]) + int(dc["r"])
    if rank > max_rank:
        raise ValueError(f"combined rank {rank} > max_lora_rank {max_rank}")
    merged = {}
    for k in sft:
        if k.endswith(".lora_A.weight"):
            dim, ss, sd = 0, sc["lora_alpha"] / sc["r"], dc["lora_alpha"] / dc["r"]
        elif k.endswith(".lora_B.weight"):
            dim, ss, sd = 1, 1.0, 1.0
        else:
            raise ValueError(f"unsupported tensor {k}")
        l, r = sft[k], dpo[k]
        if l.shape[1 - dim] != r.shape[1 - dim]:
            raise ValueError(f"incompatible shapes for {k}")
        merged[k] = torch.cat([l.float() * ss, r.float() * sd], dim=dim).to(torch.bfloat16).contiguous()
    cfg = dict(sc)
    cfg.update(r=rank, lora_alpha=rank, lora_dropout=0.0, inference_mode=True, base_model_name_or_path=base_model)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file(merged, str(tmp / "adapter_model.safetensors"), metadata={"format": "pt"})
    (tmp / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    (tmp / "scilim_stack.json").write_text(json.dumps({"sft": str(sft_dir), "dpo": str(dpo_dir),
                                                      "sft_mtime": os.path.getmtime(sft_dir / "adapter_config.json"),
                                                      "dpo_mtime": os.path.getmtime(dpo_dir / "adapter_config.json")}))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)


def copy_adapter(src: Path, dst: Path, base_model: str):
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for f in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    cfg = json.loads((dst / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = base_model
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))


def prepare_all(output_root: str, base_model: str, max_rank: int = 64, systems=None, rewards=None):
    root = Path(output_root)
    serve = root / "serve_adapters"
    serve.mkdir(parents=True, exist_ok=True)
    made = []
    for system in systems or CFG.SYSTEMS:
        for reward in rewards or CFG.REWARDS:
            ad = lambda role, algo: Path(CFG.adapter_dir(str(root), system, reward, role, algo))
            sft = ad("worker", "sft")
            plan = {CFG.lora_name(system, reward, "worker", "sft"): ("copy", sft),
                    CFG.lora_name(system, reward, "leader", "sft"): ("copy", ad("leader", "sft")),
                    CFG.lora_name(system, reward, "master", "sft"): ("copy", ad("master", "sft")),
                    CFG.lora_name(system, reward, "worker", "grpo"): ("copy", ad("worker", "grpo")),
                    CFG.lora_name(system, reward, "worker", "dpo"): ("stack", ad("worker", "dpo")),
                    CFG.lora_name(system, reward, "worker", "dpo_aug"): ("stack", ad("worker", "dpo_aug"))}
            for name, (kind, src) in plan.items():
                if not _adapter_ok(src):
                    continue
                dst = serve / name
                stamp = dst / ".source_mtime"
                mt = str(os.path.getmtime(src / "adapter_config.json"))
                if dst.exists() and stamp.exists() and stamp.read_text() == mt:
                    made.append(name)
                    continue
                try:
                    if kind == "stack":
                        if not _adapter_ok(sft):
                            print(f"[prepare] {name}: worker SFT missing, cannot stack — skipped")
                            continue
                        stack_lora(sft, src, dst, base_model, max_rank)
                    else:
                        copy_adapter(src, dst, base_model)
                    stamp.write_text(mt)
                    made.append(name)
                    print(f"[prepare] {name} <- {kind} {src}")
                except Exception as e:  # noqa: BLE001
                    print(f"[prepare] {name} FAILED: {e}")
    return made


def lora_module_args(output_root: str):
    serve = Path(output_root) / "serve_adapters"
    if not serve.exists():
        return [], 16
    mods, max_r = [], 16
    for d in sorted(serve.iterdir()):
        if d.is_dir() and _adapter_ok(d) and not d.name.endswith(".tmp"):
            mods.append(f"{d.name}={d}")
            try:
                max_r = max(max_r, int(json.loads((d / "adapter_config.json").read_text())["r"]))
            except Exception:
                pass
    r = 8
    while r < max_r:
        r *= 2
    return mods, r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_root", required=True)
    p.add_argument("--base_model", default="")
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--print_args", action="store_true",
                   help="print '--enable-lora --max-lora-rank R --lora-modules ...' for vLLM and exit")
    a = p.parse_args(argv)
    if a.print_args:
        mods, r = lora_module_args(a.output_root)
        if mods:
            print(f"--enable-lora --max-loras {min(8, len(mods))} --max-lora-rank {max(r, 16)} "
                  f"--lora-modules {' '.join(mods)}")
        return
    made = prepare_all(a.output_root, a.base_model, a.max_lora_rank)
    print(f"[prepare] {len(made)} servable adapters: {made}")


if __name__ == "__main__":
    main()
