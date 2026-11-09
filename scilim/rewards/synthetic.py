"""
Augmented DPO set ("TYPE B", build_synthetic_dpo.py).

Pool = natural worker DPO pairs, ranked by score gap. Top NATURAL_FRAC keep the
natural pair; the bottom (1-NATURAL_FRAC) keep the SAME chosen sample but get a
generated plausible-but-ungrounded rejection (generic claims, fabricated
datasets/numbers, invented evidence IDs). Same size / groups / chosen samples
as the natural set, so a DPO difference is attributable to the rejections.

Quality gate per generation: >= MIN_BULLETS bullets, token-Jaccard with chosen
< MAX_COPY_JACCARD, grounding score >= MIN_SYNTH_MARGIN below chosen.
Failed generations fall back to the natural pair ("natural_fallback").
The generator is gpt-4o-mini by default or any served model (--gen_model/--gen_base_url).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from ..llm.client import ChatClient, is_openai_model
from ..prompts.judge import SYNTH_WEAK_SYSTEM, synth_weak_user
from ..utils.text import split_bullets
from . import common as K
from . import stepwise as S


def _fallback_bad_sample(chosen, specialty):
    fake = ["the CIFAR-Bench suite", "the OpenEval-2M corpus", "the NeuroBase-7 baseline", "Table 9",
            "Appendix D.4", "a 12.7% drop", "the DistilCompare baseline"]
    out = []
    for i, b in enumerate(split_bullets(chosen)):
        b = re.sub(r"\[E\d+\]", f"[E{random.randint(400, 999)}]", b)
        b = re.sub(r'["“][^"”]{10,}["”]', "", b)
        out.append(f"- The work does not adequately address {specialty}; results on {random.choice(fake)} "
                   f"are omitted and {random.choice(fake)} is not discussed, which several related efforts "
                   f"consider standard. {b[:120]}")
        if i >= 6:
            break
    return "\n".join(out)


def generate_bad(client, cache_dir: Path, gen_model, paper_excerpt, specialty, chosen, source_text,
                 chosen_score, allow_fallback, temp0=0.95, max_tokens=900, retries=3, min_bullets=3,
                 max_copy=0.80, margin=0.05):
    n_b = max(min_bullets, min(8, len(split_bullets(chosen)) or 5))
    uses_ids = bool(re.search(r"\[E\d+\]", chosen))
    meta = {"attempts": 0, "gen_method": None, "reject_reasons": []}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries if client else 1):
        meta["attempts"] = attempt + 1
        temp = min(1.15, temp0 + 0.1 * attempt)
        if client:
            key = hashlib.sha256((gen_model + specialty + chosen[:3000] + paper_excerpt[:1500] + str(attempt)
                                  ).encode()).hexdigest()[:24]
            cp = cache_dir / f"bad_{key}.json"
            cand = json.loads(cp.read_text())["text"] if cp.exists() else None
            if cand is None:
                cand = client.chat(SYNTH_WEAK_SYSTEM, synth_weak_user(paper_excerpt, specialty, chosen, n_b, uses_ids),
                                   max_tokens=max_tokens, temperature=temp, top_p=1.0)
                if cand.startswith("ERROR"):
                    meta["reject_reasons"].append("api_error")
                    continue
                cp.write_text(json.dumps({"text": cand}))
            method = f"{gen_model}@T{temp:.2f}"
        elif allow_fallback:
            cand, method = _fallback_bad_sample(chosen, specialty), "fallback_template"
        else:
            meta["reject_reasons"].append("no_api_and_no_fallback")
            return None, meta
        cand = re.sub(r"<think>.*?</think>", " ", cand, flags=re.DOTALL | re.IGNORECASE).strip()
        bl = split_bullets(cand)
        if len(bl) < min_bullets:
            meta["reject_reasons"].append(f"too_few_bullets:{len(bl)}")
            continue
        cj = S.overlap(cand, chosen)
        if cj >= max_copy:
            meta["reject_reasons"].append(f"near_copy:{cj:.2f}")
            continue
        ss, _ = S.score_synthesize(S.build_units(cand), source_text)
        if ss > chosen_score - margin:
            meta["reject_reasons"].append(f"not_worse:{ss:.3f}vs{chosen_score:.3f}")
            continue
        meta.update({"gen_method": method, "rejected_synth_score": ss, "chosen_synth_score": chosen_score,
                     "copy_jaccard": round(cj, 4), "n_bullets": len(bl)})
        return cand, meta
    return None, meta


def build_augmented(natural_dpo: str, rollout_json: str, out: str, natural_frac=0.70, min_gap=None,
                    gen_model="gpt-4o-mini", gen_base_url=None, no_api=False, allow_fallback=False,
                    seed=0, excerpt_chars=6000, cache_dir=None):
    random.seed(seed)
    client = None
    if not no_api:
        try:
            client = ChatClient(model=gen_model, base_url=None if is_openai_model(gen_model) else gen_base_url)
        except Exception as e:  # noqa: BLE001
            print(f"[synthetic] generator unavailable ({e})")
    if client is None and not allow_fallback:
        raise SystemExit("No generator: set OPENAI_API_KEY / --gen_base_url, or --allow_fallback (tests only).")
    pool = [r for r in json.loads(Path(natural_dpo).read_text())
            if r.get("agent_role", "worker") == "worker" and (r.get("chosen") or "").strip()
            and (r.get("rejected") or "").strip()]
    if not pool:
        raise SystemExit(f"No usable worker pairs in {natural_dpo}")
    if min_gap is not None:
        pool = [r for r in pool if r.get("score_gap", 0) >= min_gap]
    papers = json.loads(Path(rollout_json).read_text())
    src = {p.get("paper_idx"): K.source_text_of(p) for p in papers}
    sysmap = {p.get("paper_idx"): p.get("system", "limagents") for p in papers}
    pool.sort(key=lambda r: r.get("score_gap", 0.0), reverse=True)
    n_nat = max(0, min(len(pool), int(round(natural_frac * len(pool)))))
    top, bottom = pool[:n_nat], pool[n_nat:]
    cache = Path(cache_dir or (Path(out).parent.parent / "synth_cache"))
    rows, n_syn, n_fb, fails = [], 0, 0, defaultdict(int)
    for rank, r in enumerate(top, 1):
        rows.append({**r, "pair_type": "natural", "gap_rank": rank, "chosen_score_basis": "reward_model",
                     "rejected_score_basis": "reward_model"})
    for rank, r in enumerate(bottom, len(top) + 1):
        pid, name, chosen = r.get("paper_idx"), r.get("agent_name"), r["chosen"]
        s = src.get(pid, "")
        spec = K.specialty_label(sysmap.get(pid, "limagents"), name)
        rej, meta = None, {"reject_reasons": ["no_source_text"], "attempts": 0}
        if s:
            cs, _ = S.score_synthesize(S.build_units(chosen), s)
            rej, meta = generate_bad(client, cache, gen_model, s[:excerpt_chars], spec, chosen, s, cs,
                                     allow_fallback)
        row = {**r, "gap_rank": rank, "chosen_score_basis": "reward_model"}
        if rej is None:
            row.update(pair_type="natural_fallback", rejected_score_basis="reward_model",
                       synth_failed_reasons=meta.get("reject_reasons", []))
            n_fb += 1
            for x in meta.get("reject_reasons", []):
                fails[x.split(":")[0]] += 1
        else:
            row.update(pair_type="synthetic", rejected_score_basis="synthesize_grounding",
                       natural_rejected=r["rejected"], natural_rejected_mode=r.get("rejected_mode"),
                       natural_score_gap=r.get("score_gap"), rejected=rej, rejected_mode="synthetic_hallucinated",
                       rejected_score=meta.get("rejected_synth_score"), rejected_stepwise=meta.get("rejected_synth_score"),
                       rejected_gt_similarity=None, rejected_rollout_id=None,
                       chosen_synth_score=meta.get("chosen_synth_score"),
                       rejected_synth_score=meta.get("rejected_synth_score"),
                       synth_score_gap=round((meta.get("chosen_synth_score") or 0) - (meta.get("rejected_synth_score") or 0), 4),
                       copy_jaccard=meta.get("copy_jaccard"), gen_method=meta.get("gen_method"),
                       gen_attempts=meta.get("attempts"))
            n_syn += 1
        rows.append(row)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2, default=str))
    with open(Path(out).with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["paper_idx", "agent_name", "pair_type", "gap_rank", "chosen_mode", "rejected_mode", "chosen_score",
                "rejected_score", "score_gap", "synth_score_gap", "copy_jaccard", "gen_method", "gen_attempts"]
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"[synthetic] {len(rows)} pairs (natural {len(top)}, synthetic {n_syn}, fallback {n_fb}); "
          f"gate rejections {dict(fails)}")
    if n_syn and rows[-1].get("gen_method") == "fallback_template":
        print("*** WARNING: offline template generator used — plumbing test only, do NOT train on this.")
    return out
