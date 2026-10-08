"""Dataset agnostic preprocessing driven entirely by ``DataConfig``.

Leakage control: every statistic that depends on the data (constant columns, the
percentile caps, scaler ranges, one hot categories) is fitted on the TRAIN split only
and then applied to validation and test.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ..constants import ATTACK_LABEL, BENIGN_LABEL
from ..config import DataConfig
from .loading import find_label_column, load_raw_folder, subsample

_LABEL = "__label__"
_Y = "__y__"


@dataclass
class PreparedData:
    X_train: torch.Tensor
    X_val: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor   # long, ATTACK_LABEL=0 / BENIGN_LABEL=1
    y_val: torch.Tensor
    y_test: torch.Tensor
    feature_names: List[str]
    feature_indices: List[int]      # columns of X the attacker may perturb
    fingerprint: str
    n_attack: int
    n_benign: int

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


def fingerprint(df: pd.DataFrame, dataset_name: str) -> str:
    """Short hash of the sampled data, to detect accidentally identical or stale runs."""
    h = hashlib.md5()
    h.update(dataset_name.encode())
    h.update(str(df.shape).encode())
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        h.update(str(round(float(numeric.sum().sum()), 2)).encode())
    return h.hexdigest()[:12]


def prepare_data(cfg: DataConfig, seed: int) -> PreparedData:
    if not cfg.raw_dir:
        raise ValueError("data.raw_dir is not set. Pass --raw-dir or set it in the config.")

    print("Loading raw CSV(s)...")
    df = load_raw_folder(cfg.raw_dir, cfg.csv_encoding, cfg.rows_per_file, seed)

    df = df.drop(columns=[c for c in cfg.drop_columns if c in df.columns])
    label_col = find_label_column(df, cfg.label_columns)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col].str.lower() != "nan"]
    benign_set = {b.strip().lower() for b in cfg.benign_labels}

    def attack_mask(d: pd.DataFrame) -> pd.Series:
        return ~d[label_col].str.lower().isin(benign_set)

    print("Subsampling...")
    df = subsample(df, attack_mask(df), cfg.sample_size, cfg.sample_mode, seed)
    fp = fingerprint(df, cfg.name)
    print(f"Data fingerprint: {fp}")
    print(df[label_col].value_counts().to_string())

    # Explicit label encoding: never rely on alphabetical order.
    is_attack = attack_mask(df)
    df[_Y] = np.where(is_attack, ATTACK_LABEL, BENIGN_LABEL).astype(np.int64)
    df = df.rename(columns={label_col: _LABEL})

    feature_cols = [c for c in df.columns if c not in (_LABEL, _Y)]
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print("Cleaning...")
    for col in cfg.nonnegative_columns:
        if col in df.columns:
            df = df[df[col] >= 0]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    counts = df[_Y].value_counts()
    if counts.get(ATTACK_LABEL, 0) < 3 or counts.get(BENIGN_LABEL, 0) < 3:
        raise ValueError(f"After cleaning, too few rows remain in one class ({counts.to_dict()}). "
                         "Check the dataset config (benign_labels, nonnegative_columns, drop_columns).")
    n_attack, n_benign = int(counts[ATTACK_LABEL]), int(counts[BENIGN_LABEL])

    # Split first, then fit every data dependent step on train only.
    idx = np.arange(len(df))
    tr_idx, ho_idx = train_test_split(idx, test_size=cfg.holdout_fraction,
                                      random_state=seed, stratify=df[_Y].values)
    te_idx, va_idx = train_test_split(ho_idx, test_size=cfg.val_share_of_holdout,
                                      random_state=seed, stratify=df[_Y].values[ho_idx])

    X_all = df[feature_cols]
    y_all = df[_Y]
    X_train, X_val, X_test = X_all.iloc[tr_idx].copy(), X_all.iloc[va_idx].copy(), X_all.iloc[te_idx].copy()
    y_train, y_val, y_test = y_all.iloc[tr_idx], y_all.iloc[va_idx], y_all.iloc[te_idx]

    constant = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    if constant:
        print(f"Dropping {len(constant)} column(s) constant on train: {constant}")
        X_train = X_train.drop(columns=constant)
        X_val = X_val.drop(columns=constant)
        X_test = X_test.drop(columns=constant)

    for col in cfg.cap_columns:
        if col in X_train.columns:
            cap = X_train[col].quantile(cfg.cap_quantile)
            for part in (X_train, X_val, X_test):
                part[col] = part[col].clip(upper=cap)

    cat_present = [c for c in cfg.categorical_columns if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_present]
    if not num_cols and not cat_present:
        raise ValueError("No feature columns remain after cleaning.")

    if cfg.oversample_train:
        ros = RandomOverSampler(sampling_strategy="auto", random_state=seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    transformers = []
    if num_cols:
        transformers.append(("num", MinMaxScaler(), num_cols))
    if cat_present:
        transformers.append(("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_present))
    pre = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)
    out_names = [str(n) for n in pre.get_feature_names_out()]

    def to_x(a: np.ndarray) -> torch.Tensor:
        return torch.clamp(torch.as_tensor(np.array(a, dtype=np.float32)), 0.0, 1.0)

    def to_y(s) -> torch.Tensor:
        return torch.as_tensor(np.array(s, dtype=np.int64))

    # Resolve perturbable features by NAME against the final column order.
    name_to_idx = {n: i for i, n in enumerate(out_names)}
    wanted = list(dict.fromkeys(cfg.perturbable_features))
    found = [f for f in wanted if f in name_to_idx and f in num_cols]
    missing = [f for f in wanted if f not in found]
    feature_indices = sorted(name_to_idx[f] for f in found)
    print(f"Perturbable features: {len(found)} found, {len(missing)} not present in this dataset")
    if missing:
        print(f"  not present: {missing}")
    if not feature_indices:
        raise ValueError(
            "None of data.perturbable_features exist in the processed data, so attacks would be no ops "
            "and robustness results would be meaningless. Fix the names in the dataset config. "
            f"Example available columns: {out_names[:12]}")

    data = PreparedData(
        X_train=to_x(Xtr), X_val=to_x(Xva), X_test=to_x(Xte),
        y_train=to_y(y_train), y_val=to_y(y_val), y_test=to_y(y_test),
        feature_names=out_names, feature_indices=feature_indices,
        fingerprint=fp, n_attack=n_attack, n_benign=n_benign,
    )
    print(f"Features: {data.n_features} | Train: {len(data.X_train)} | "
          f"Val: {len(data.X_val)} | Test: {len(data.X_test)}")
    print(f"Label mapping: attack={ATTACK_LABEL}, benign={BENIGN_LABEL}")
    return data
