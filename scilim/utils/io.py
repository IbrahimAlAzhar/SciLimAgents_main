"""Incremental, resumable CSV output.

Every generation is written to NEW columns of a copy of the input dataframe,
and the dataframe is saved atomically every `save_every` (default 10) rows.
Re-running the same command resumes: rows whose target column is already
filled are skipped.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
from typing import Iterable, List, Optional

import pandas as pd

from .text import is_missing

try:
    from tqdm import tqdm  # noqa: F401
except ImportError:  # pragma: no cover
    def tqdm(it=None, **kw):  # type: ignore
        return it


def atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(obj, path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_rows(input_csv: str, start: int = 0, end: Optional[int] = None,
              n_rows: Optional[int] = None) -> pd.DataFrame:
    """Read a slice [start:end) (or [start:start+n_rows)) and keep the original
    row index in a `row_id` column so outputs can be joined back."""
    df = pd.read_csv(input_csv)
    if n_rows is not None and end is None:
        end = start + n_rows
    end = len(df) if end is None else min(end, len(df))
    df = df.iloc[start:end].copy()
    if "row_id" not in df.columns:
        df.insert(0, "row_id", df.index.astype(int))
    return df.reset_index(drop=True)


class IncrementalFrame:
    """Holds the working dataframe, resumes from an existing output CSV, and
    saves every `save_every` completed rows (plus on SIGTERM / SIGINT)."""

    def __init__(self, df: pd.DataFrame, output_csv: str, new_columns: Iterable[str],
                 done_column: str, save_every: int = 10):
        new_columns = list(new_columns)
        self.output_csv = output_csv
        self.save_every = max(1, int(save_every))
        self.done_column = done_column
        self.df = df
        self._since_save = 0
        if os.path.exists(output_csv):
            old = pd.read_csv(output_csv)
            if "row_id" in old.columns and "row_id" in df.columns:
                old = old.set_index("row_id")
                for col in old.columns:
                    if col in df.columns and col not in new_columns and col != done_column:
                        continue
                    vals = df["row_id"].map(old[col]) if col in old.columns else None
                    if vals is not None:
                        if col not in self.df.columns:
                            self.df[col] = vals
                        else:
                            mask = vals.notna()
                            self.df[col] = self.df[col].astype(object)
                            self.df.loc[mask, col] = vals[mask]
                print(f"[resume] loaded previous output {output_csv}")
        for c in list(new_columns) + [done_column]:
            if c not in self.df.columns:
                self.df[c] = ""
            self.df[c] = self.df[c].astype(object)
        self._install_signal_handlers()

    def pending_indices(self, retry_errors: bool = True) -> List[int]:
        out = []
        for i in range(len(self.df)):
            v = self.df.at[i, self.done_column]
            if is_missing(v) or (retry_errors and str(v).startswith(("ERROR", "[ERROR]"))):
                out.append(i)
        return out

    def set(self, i: int, values: dict) -> None:
        for k, v in values.items():
            if k not in self.df.columns:
                self.df[k] = ""
                self.df[k] = self.df[k].astype(object)
            self.df.at[i, k] = v

    def row_done(self) -> None:
        self._since_save += 1
        if self._since_save >= self.save_every:
            self.save()

    def save(self) -> None:
        atomic_to_csv(self.df, self.output_csv)
        self._since_save = 0
        print(f"  [save] {self.output_csv} ({(~self.df[self.done_column].map(is_missing)).sum()}"
              f"/{len(self.df)} rows done)", flush=True)

    def _install_signal_handlers(self):
        def handler(signum, frame):
            print(f"\n[signal {signum}] saving progress to {self.output_csv}", flush=True)
            try:
                self.save()
            finally:
                sys.exit(130)
        for s in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(s, handler)
            except Exception:
                pass
