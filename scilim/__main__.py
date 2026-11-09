"""
scilim — unified LimAgents / NovAgents / RL pipeline.

    python -m scilim generate  ...   limitation generation for any ablation
    python -m scilim rollout   ...   rollouts for SFT/DPO/GRPO (teacher = self | gpt | other)
    python -m scilim reward    ...   outcome | stepwise | jsd | combined | synthetic
    python -m scilim sft       ...   LoRA SFT (worker / leader / master)
    python -m scilim dpo       ...   worker DPO on top of SFT
    python -m scilim grpo      ...   worker GRPO (any reward)
    python -m scilim prepare   ...   stack SFT+DPO LoRA for vLLM, print --lora-modules
    python -m scilim eval      ...   P/R/F1 + similarity (+ false-positive analysis)
    python -m scilim collect   ...   merge all ablation columns into one CSV
    python -m scilim check_vocab --student A --teacher B
Run any sub-command with -h for its options.
"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "generate":
        from .generate import main as m
    elif cmd == "rollout":
        from .rollout.generate import main as m
    elif cmd == "reward":
        from .rewards.__main__ import main as m
    elif cmd == "sft":
        from .train.sft import main as m
    elif cmd == "dpo":
        from .train.dpo import main as m
    elif cmd == "grpo":
        from .train.grpo import main as m
    elif cmd == "prepare":
        from .serve.prepare import main as m
    elif cmd == "eval":
        from .evaluation.evaluate import main as m
    elif cmd == "collect":
        from .collect import main as m
    elif cmd == "check_vocab":
        from .serve.check_vocab import main as m
    else:
        raise SystemExit(f"unknown command {cmd}\n{__doc__}")
    m(argv)


if __name__ == "__main__":
    main()
