"""Raw CSV loading and subsampling."""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def load_raw_folder(raw_dir: str, encoding: str = "latin1",
                    rows_per_file: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Load and concatenate every CSV under ``raw_dir`` (searched recursively).

    ``raw_dir`` may also point at a single CSV file. Column names are stripped of
    surrounding whitespace, since CIC CSVs often contain names like ' Flow Duration'.
    If ``rows_per_file`` is set, each file is randomly reduced to at most that many rows
    right after reading, which keeps memory bounded for very large datasets.
    """
    if os.path.isfile(raw_dir):
        paths = [raw_dir]
    else:
        paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under '{raw_dir}'.")

    print(f"Found {len(paths)} CSV file(s) under {raw_dir}")
    frames: List[pd.DataFrame] = []
    for i, p in enumerate(paths):
        try:
            df = pd.read_csv(p, low_memory=False, encoding=encoding)
        except Exception as e:  # unreadable file: report loudly, keep going
            print(f"  [skip] failed to read {p}: {e}")
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if rows_per_file is not None and len(df) > rows_per_file:
            df = df.sample(n=rows_per_file, random_state=seed + i)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"None of the CSV files under '{raw_dir}' could be read.")

    col_sets = {tuple(sorted(f.columns)) for f in frames}
    if len(col_sets) > 1:
        print(f"  [warn] the CSV files do not all share the same columns ({len(col_sets)} distinct "
              f"column sets). Rows with missing values will be dropped later.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Combined raw shape: {df.shape}")
    return df


def find_label_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the actual column name of the label, trying candidates in order (case insensitive)."""
    lower = {}
    for c in df.columns:
        lower.setdefault(str(c).strip().lower(), c)
    for cand in candidates:
        if cand.strip().lower() in lower:
            return lower[cand.strip().lower()]
    raise KeyError(f"None of the label column candidates {candidates} were found. "
                   f"Available columns: {list(df.columns)[:15]}...")


def subsample(df: pd.DataFrame, is_attack: pd.Series, target_total: int,
              mode: str, seed: int = 42) -> pd.DataFrame:
    """Reduce ``df`` to at most ``target_total`` rows before expensive processing.

    mode='stratified': keep the dataset's natural attack:benign ratio.
    mode='balanced':   aim for 50/50, capped by whichever class has fewer rows.
    ``is_attack`` must be a boolean Series aligned with ``df``.
    """
    if mode not in ("stratified", "balanced"):
        raise ValueError(f"Unknown sample mode '{mode}'.")
    is_attack = is_attack.reindex(df.index).astype(bool)
    n_attack, n_benign = int(is_attack.sum()), int((~is_attack).sum())
    print(f"Raw class counts: attack {n_attack}, benign {n_benign}")
    if n_attack == 0 or n_benign == 0:
        raise ValueError("The data contains only one class. Check `benign_labels` in the dataset config "
                         "(labels are matched case insensitively).")

    atk_df, ben_df = df[is_attack], df[~is_attack]
    if mode == "balanced":
        half = target_total // 2
        n_atk_take = min(half, n_attack)
        n_ben_take = min(target_total - n_atk_take, n_benign)
        n_atk_take = min(target_total - n_ben_take, n_attack)
    else:
        if len(df) <= target_total:
            n_atk_take, n_ben_take = n_attack, n_benign
        else:
            frac = target_total / len(df)
            n_atk_take = max(1, min(n_attack, int(round(n_attack * frac))))
            n_ben_take = max(1, min(n_benign, int(round(n_benign * frac))))
            # trim rounding overshoot from the larger class
            while n_atk_take + n_ben_take > target_total:
                if n_atk_take >= n_ben_take:
                    n_atk_take -= 1
                else:
                    n_ben_take -= 1

    out = pd.concat([atk_df.sample(n=n_atk_take, random_state=seed),
                     ben_df.sample(n=n_ben_take, random_state=seed)])
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"Subsampled to {len(out)} rows: attack {n_atk_take}, benign {n_ben_take} (mode={mode})")
    return out
