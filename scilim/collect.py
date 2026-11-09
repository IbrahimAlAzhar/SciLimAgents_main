"""Merge every generations/<tag>.csv into ONE dataframe (input columns + one
column per ablation), joined on row_id.

    python -m scilim collect --output_root outputs/qwen3_8b [--with_intermediate]
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from .utils.io import atomic_to_csv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", required=True)
    p.add_argument("--with_intermediate", action="store_true", help="also keep <tag>__<agent> columns")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.output_root, "generations", "*.csv")))
    if not files:
        raise SystemExit("no generation files found")
    base = None
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        d = pd.read_csv(f)
        keep = [c for c in d.columns if c == tag or (a.with_intermediate and c.startswith(tag + "__"))]
        if base is None:
            gen_cols = {c for c in d.columns if c == tag or c.startswith(tag + "__")}
            base = d[[c for c in d.columns if c not in gen_cols]].copy()
        base = base.merge(d[["row_id"] + keep], on="row_id", how="outer")
        print(f"  + {tag} ({len(keep)} cols)")
    out = a.out or os.path.join(a.output_root, "all_ablations.csv")
    atomic_to_csv(base, out)
    print(f"-> {out}  ({len(base)} rows, {len(base.columns)} cols)")


if __name__ == "__main__":
    main()
