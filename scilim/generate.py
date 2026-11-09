"""
Generate limitations for every requested ablation.

Each ablation writes its final limitations to a NEW column named after the
ablation tag (plus `<tag>__<agent>` columns for intermediate agent outputs),
in outputs/<model>/generations/<tag>.csv, saving every --save_every rows
(default 10) and resuming automatically.

ABLATION NAMES  (combine several with spaces/commas in --ablations)
-----------------------------------------------------------------------------
Baselines        zero_shot | rag | rag_only
LimAgents        limagents            original v1 (7 specialists + Master)
                 limagents_seq        7 workers -> Leader -> Master
                 limagents_seq_hops   ... with 3-hop (search/query/draft) workers
                 limagents_canonical  6 canonical workers -> Leader B -> Master+ToT
                 limagents_loo        v1 + leave-one-agent-out Masters (7 cols)
                 limagents_agent:<Agent>   one v1 specialist alone (no Master)
                 limagents_each       expands to all 7 single-agent ablations
NovAgents        novagents | novagents_agent:<Agent> | novagents_each
Merge            merge                     limagents(v1) + novagents
                 merge:<lim_variant>+<nov_variant>   e.g. merge:limagents_dpo+novagents_grpo
Trained          limagents_{sft,dpo,dpo_aug,grpo}   canonical style + LoRA adapters
                 novagents_{sft,dpo,dpo_aug,grpo}
                 (adapters chosen by --reward outcome|stepwise|jsd|combined)
Modifiers        append @mem (shared-memory RAG), @hops (3-hop workers),
                 @critique (2 Leader critique rounds), @workeronly (only the
                 worker is trained; Leader/Master use the base model), e.g.
                 limagents_dpo@mem  limagents_canonical@hops@mem
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import config as CFG
from .agents import LimAgents, LimAgentsConfig, NovAgents, NovAgentsConfig, run_merge, run_single
from .llm.client import ChatClient, is_openai_model
from .prompts import limagents_v1 as V1
from .prompts import novagents as N
from .utils.io import IncrementalFrame, load_rows
from .utils.text import is_missing

TRAINED = ("sft", "dpo", "dpo_aug", "grpo")


# ---------------------------------------------------------------------------
# Ablation parsing
# ---------------------------------------------------------------------------
def expand_ablations(spec: List[str]) -> List[str]:
    out = []
    for s in spec:
        for a in re.split(r"[,\s]+", s.strip()):
            if not a:
                continue
            base, *mods = a.split("@")
            suffix = "".join("@" + m for m in mods)
            if base == "limagents_each":
                out += [f"limagents_agent:{n}{suffix}" for n, _ in V1.SPECIALIST_COLUMNS]
            elif base == "novagents_each":
                out += [f"novagents_agent:{n}{suffix}" for n, _ in N.SPECIALIST_AGENTS]
            else:
                out.append(a)
    return list(dict.fromkeys(out))


def ablation_tag(ablation: str, reward: str) -> str:
    needs_reward = any(re.search(rf"_{t}(\b|@|\+|$)", ablation) for t in TRAINED)
    tag = ablation.replace(":", "__").replace("+", "_plus_").replace("@", "_at_")
    tag = re.sub(r"[^A-Za-z0-9_\-]", "_", tag)
    return f"{tag}__{reward}" if needs_reward else tag


def parse_variant(name: str):
    """'limagents_dpo@mem' -> ('limagents', 'dpo', {'mem'}, agent)"""
    base, *mods = name.split("@")
    agent = None
    if ":" in base:
        base, agent = base.split(":", 1)
    system = "novagents" if base.startswith("novagents") else "limagents"
    rest = base[len(system):].lstrip("_")
    if rest == "agent":
        rest = ""
    return system, rest, set(mods), agent


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, args):
        self.args = args
        self.llm = make_client(args.llm_model, args.llm_base_url, args.tokenizer, args.max_model_len,
                               args.family, args.repetition_penalty)
        self.merger = None
        self.out_dir = os.path.join(args.output_root, "generations")
        os.makedirs(self.out_dir, exist_ok=True)

    # -- model/adapters -----------------------------------------------------
    def roles_for(self, system: str, algo: str, mods: set) -> Dict[str, Optional[str]]:
        a = self.args
        if not algo:
            return {}
        if algo not in TRAINED:
            raise ValueError(f"unknown training variant '{algo}'")
        if self.llm.is_openai:
            raise SystemExit(f"Trained ablations ({system}_{algo}) need an open-weights model served by "
                             "vLLM; GPT cannot be fine-tuned here. Use GPT as --teacher/--judge instead.")
        worker_algo = algo
        roles = {"worker": CFG.lora_name(system, a.reward, "worker", worker_algo)}
        if "workeronly" not in mods:
            roles["leader"] = CFG.lora_name(system, a.reward, "leader", "sft") if system == "limagents" else None
            roles["master"] = CFG.lora_name(system, a.reward, "master", "sft")
        return roles

    def make_pipeline(self, variant: str):
        a = self.args
        system, rest, mods, agent = parse_variant(variant)
        algo = rest if rest in TRAINED else ""
        roles = self.roles_for(system, algo, mods)
        if system == "limagents":
            style = ("v1" if rest in ("", "loo") and not algo else
                     "seq" if rest.startswith("seq") else "canonical")
            if agent is not None and not algo and rest == "":
                style = "v1"
            cfg = LimAgentsConfig(style=style, roles=roles, text_col=a.text_col,
                                  agents=[agent] if agent else None, with_master=agent is None,
                                  hops=("hops" in mods) or rest.endswith("hops"),
                                  critique_rounds=2 if "critique" in mods else 0,
                                  shared_memory="mem" in mods, worker_threads=a.worker_threads,
                                  leader_ascii_prompt=a.leader_ascii_prompt, seed=a.seed,
                                  retrieval_cols=tuple(a.retrieval_cols.split(",")))
            if a.temperature is not None:
                cfg.temperature = cfg.v1_temperature = a.temperature
            return system, LimAgents(self.llm, cfg), rest
        cfg = NovAgentsConfig(roles=roles, text_col=a.text_col, agents=[agent] if agent else None,
                              with_master=agent is None, worker_threads=a.worker_threads, seed=a.seed)
        if a.temperature is not None:
            cfg.temperature = a.temperature
        return system, NovAgents(self.llm, cfg), rest

    def get_merger(self) -> Tuple[ChatClient, Optional[str]]:
        a = self.args
        if a.merger in ("self", "", None):
            return self.llm, None
        if self.merger is None:
            if is_openai_model(a.merger):
                self.merger = make_client(a.merger, None, None, 128000, "openai", None)
            else:
                self.merger = make_client(a.merger, a.merger_base_url or a.llm_base_url, a.tokenizer,
                                          a.max_model_len, a.family, None)
        return self.merger, None

    # -- component reuse for merge --------------------------------------------
    def component_value(self, variant: str, row) -> Optional[str]:
        tag = ablation_tag(variant, self.args.reward)
        path = os.path.join(self.out_dir, f"{tag}.csv")
        if not self.args.reuse_components or not os.path.exists(path):
            return None
        cache = getattr(self, "_cache", {})
        self._cache = cache
        if path not in cache:
            d = pd.read_csv(path)
            cache[path] = d.set_index("row_id")[tag].to_dict() if tag in d.columns else {}
        v = cache[path].get(int(row["row_id"]))
        return None if is_missing(v) or str(v).startswith("ERROR") else str(v)

    def run_variant_row(self, variant: str, row) -> Dict[str, str]:
        _, pipe, rest = self.make_pipeline(variant)
        if rest == "loo":
            res = pipe.run_v1(row)
            res.update(pipe.run_v1_leave_one_out(row, res))
            return res
        return pipe.run(row)

    # -- per-ablation processing ------------------------------------------------
    def process_row(self, ablation: str, row) -> Dict[str, str]:
        a = self.args
        if ablation in ("zero_shot", "rag", "rag_only"):
            return run_single(self.llm, row, ablation, text_col=a.text_col,
                              retrieval_cols=tuple(a.retrieval_cols.split(",")),
                              temperature=a.temperature if a.temperature is not None else 0.2,
                              seed=a.seed)
        if ablation == "merge" or ablation.startswith("merge:"):
            lim_v, nov_v = ("limagents", "novagents") if ablation == "merge" else ablation[6:].split("+", 1)
            out = {}
            texts = {}
            for key, v in (("lim", lim_v), ("nov", nov_v)):
                txt = self.component_value(v, row)
                if txt is None:
                    res = self.run_variant_row(v, row)
                    txt = res.get("final", "")
                    out.update({f"{key}__{k}": val for k, val in res.items() if k != "final"})
                texts[key] = txt
                out[f"{key}_report"] = txt
            merger, model = self.get_merger()
            out["final"] = run_merge(merger, texts["lim"], texts["nov"], style=a.merge_style, model=model)
            return out
        return self.run_variant_row(ablation, row)

    def required_models(self, ablation: str) -> List[str]:
        names = []
        parts = ablation[6:].split("+", 1) if ablation.startswith("merge:") else [ablation]
        for v in parts:
            if v in ("zero_shot", "rag", "rag_only", "merge"):
                continue
            system, rest, mods, _ = parse_variant(v)
            if rest in TRAINED:
                names += [n for n in self.roles_for(system, rest, mods).values() if n]
        return names

    def check_served(self, ablation: str) -> bool:
        if self.llm.mock or self.llm.is_openai:
            return True
        need = self.required_models(ablation)
        if not need:
            return True
        if not hasattr(self, "_served"):
            self._served = set(self.llm.list_models())
        missing = [n for n in need if n not in self._served]
        if missing:
            print(f"[skip] {ablation}: adapters not served by vLLM: {missing}\n"
                  f"       train them (stage train) and run stage prepare first.", flush=True)
            return False
        return True

    def run_ablation(self, ablation: str, df: pd.DataFrame):
        a = self.args
        tag = ablation_tag(ablation, a.reward)
        if not self.check_served(ablation):
            return None
        out_csv = os.path.join(self.out_dir, f"{tag}.csv")
        print(f"\n{'=' * 70}\nABLATION {ablation}  ->  column '{tag}'\n  {out_csv}\n{'=' * 70}", flush=True)
        frame = IncrementalFrame(df.copy(), out_csv, new_columns=[tag], done_column=tag,
                                 save_every=a.save_every)
        todo = frame.pending_indices(retry_errors=a.retry_errors)
        print(f"  {len(todo)} / {len(frame.df)} rows to generate", flush=True)
        t0 = time.time()

        def work(i):
            row = frame.df.iloc[i]
            try:
                res = self.process_row(ablation, row)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                res = {"final": f"ERROR: {type(e).__name__}: {e}"}
            failed = [k for k, v in res.items() if k != "final" and isinstance(v, str) and v.startswith("ERROR")]
            if failed and not str(res.get("final", "")).startswith("ERROR"):
                res["final"] = f"ERROR: agent call failed ({', '.join(failed[:3])})"
            vals = {tag: res.pop("final", "")}
            vals.update({f"{tag}__{k}": v for k, v in res.items()})
            return i, vals

        done = 0
        # rows are processed in chunks of save_every; each chunk runs concurrently
        for start in range(0, len(todo), a.save_every):
            chunk = todo[start:start + a.save_every]
            with ThreadPoolExecutor(max_workers=max(1, a.row_workers)) as ex:
                for fut in as_completed([ex.submit(work, i) for i in chunk]):
                    i, vals = fut.result()
                    frame.set(i, vals)
                    done += 1
            frame.save()
            el = time.time() - t0
            print(f"  [{ablation}] {done}/{len(todo)} rows | {el / max(done, 1):.1f}s/row | "
                  f"ETA {(len(todo) - done) * el / max(done, 1) / 3600:.2f}h", flush=True)
        frame.save()
        return out_csv


def make_client(model, base_url, tokenizer, max_model_len, family, rep):
    return ChatClient(model=model, base_url=base_url or None, tokenizer=tokenizer,
                      max_model_len=max_model_len, family=family or "auto",
                      default_repetition_penalty=rep)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablations", nargs="+", required=True)
    p.add_argument("--input_csv", default=CFG.DEFAULT_INPUT_CSV)
    p.add_argument("--start_row", type=int, default=0)
    p.add_argument("--end_row", type=int, default=None)
    p.add_argument("--n_rows", type=int, default=None, help="rows to process (if --end_row unset)")
    p.add_argument("--output_root", required=True, help="outputs/<model_tag>")
    p.add_argument("--llm_model", required=True, help="served name (vLLM) | gpt-4o-mini | mock")
    p.add_argument("--llm_base_url", default=None, help="http://127.0.0.1:8000/v1 (empty for OpenAI)")
    p.add_argument("--tokenizer", default=None, help="HF model path for token budgeting")
    p.add_argument("--family", default="auto")
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None, help="override pipeline defaults")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reward", default="stepwise", choices=list(CFG.REWARDS),
                   help="which trained adapters to use for *_sft/_dpo/_grpo ablations")
    p.add_argument("--merger", default="gpt-4o-mini", help="merge model: gpt-4o-mini | self | served name")
    p.add_argument("--merger_base_url", default=None)
    p.add_argument("--merge_style", default="anchor", choices=["anchor", "anchor_v0", "simple"])
    p.add_argument("--reuse_components", type=int, default=1,
                   help="merge: reuse already generated component columns if present")
    p.add_argument("--text_col", default=CFG.COLS["text"])
    p.add_argument("--retrieval_cols", default=CFG.COLS["retrieval_cols"])
    p.add_argument("--leader_ascii_prompt", action="store_true")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--row_workers", type=int, default=4, help="rows generated concurrently")
    p.add_argument("--worker_threads", type=int, default=7, help="agents per row concurrently")
    p.add_argument("--retry_errors", type=int, default=1)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    abls = expand_ablations(args.ablations)
    df = load_rows(args.input_csv, args.start_row, args.end_row, args.n_rows)
    print(f"Loaded {len(df)} rows [{args.start_row}:{args.end_row or (args.start_row + (args.n_rows or len(df)))}] "
          f"from {args.input_csv}")
    print(f"Ablations: {abls}")
    r = Runner(args)
    if not r.llm.mock and not r.llm.is_openai:
        r.llm.wait_ready()
    for ab in abls:
        r.run_ablation(ab, df)
    print("\nAll ablations done.")


if __name__ == "__main__":
    main()
