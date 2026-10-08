"""Typed configuration.

All dataset specific knowledge (label column, benign label, columns to drop,
columns to cap, which features an attacker may perturb) lives in YAML files
under ``configs/``. Nothing dataset specific is hard coded in the Python code.

Loading order: ``configs/default.yaml`` is deep merged with the dataset YAML,
then command line overrides are applied on top.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

import yaml

KNOWN_ARCHS = ("cnn", "lstm", "clstm")
SAMPLE_MODES = ("stratified", "balanced")
THREAT_MODELS = ("masked", "masked_mape")


@dataclass
class DataConfig:
    name: str = "dataset"
    raw_dir: Optional[str] = None
    csv_encoding: str = "latin1"
    # Candidate label column names, tried in order, case insensitive.
    label_columns: List[str] = field(default_factory=lambda: ["Label", "label", "Attack Type"])
    # Labels (case insensitive) that count as benign. Everything else is an attack.
    benign_labels: List[str] = field(default_factory=lambda: ["benign", "benigntraffic", "normal traffic"])
    drop_columns: List[str] = field(default_factory=list)
    # Columns that are one hot encoded. Everything else is min max scaled.
    categorical_columns: List[str] = field(default_factory=list)
    # Columns whose upper tail is clipped at `cap_quantile`, fitted on TRAIN only.
    cap_columns: List[str] = field(default_factory=list)
    cap_quantile: float = 0.95
    # Rows with a negative value in any of these columns are dropped.
    nonnegative_columns: List[str] = field(default_factory=list)
    # The only columns an attacker may modify (physically realisable features).
    perturbable_features: List[str] = field(default_factory=list)
    sample_size: int = 100000
    sample_mode: str = "stratified"
    # Optional: randomly keep at most this many rows per CSV file while loading.
    # Useful for very large datasets (CICIoT2023) so the full data never sits in RAM.
    rows_per_file: Optional[int] = None
    oversample_train: bool = True
    holdout_fraction: float = 0.2      # val + test share of the data
    val_share_of_holdout: float = 0.5  # half of holdout is validation, half test

    def __post_init__(self):
        self.cap_quantile = float(self.cap_quantile)
        self.holdout_fraction = float(self.holdout_fraction)
        self.val_share_of_holdout = float(self.val_share_of_holdout)


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 4
    lr: float = 0.001
    l2_lambda: float = 0.00001   # Adam weight decay
    l1_lambda: float = 0.000001  # explicit L1 penalty on all parameters
    temperature: float = 4.0
    alpha: float = 0.4           # weight of the logit distillation loss
    beta: float = 0.4            # weight of the feature distillation loss
    # Teachers draw epsilon uniformly from this range every batch.
    teacher_epsilon_range: List[float] = field(default_factory=lambda: [0.01, 0.1])
    # The student draws epsilon uniformly from this list every batch.
    student_epsilons: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.3])

    def __post_init__(self):
        self.lr = float(self.lr)
        self.l2_lambda = float(self.l2_lambda)
        self.l1_lambda = float(self.l1_lambda)
        self.temperature = float(self.temperature)
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.teacher_epsilon_range = [float(e) for e in self.teacher_epsilon_range]
        self.student_epsilons = [float(e) for e in self.student_epsilons]


@dataclass
class AttackConfig:
    # A perturbed malicious row is rolled back to its clean value if any non near zero
    # feature changed by more than this many percent. None disables the check.
    mape_threshold: Optional[float] = 20.0
    ifgsm_abs_stepsize: float = 0.01
    ifgsm_steps: int = 20
    pgd_rel_stepsize: float = 0.02
    pgd_steps: int = 40
    pgd_random_start: bool = True


@dataclass
class EvalConfig:
    epsilons: List[float] = field(default_factory=lambda: [0.05, 0.07, 0.1])
    # masked:       perturb only `perturbable_features`, no realism clamp (strongest attacker).
    # masked_mape:  as above, plus the MAPE realism rollback used during training.
    threat_models: List[str] = field(default_factory=lambda: ["masked", "masked_mape"])
    # A model whose CLEAN validation performance falls below these has collapsed to a
    # majority class predictor.
    collapse_min_macro_f1: float = 0.6
    collapse_min_attack_recall: float = 0.05

    def __post_init__(self):
        self.epsilons = [float(e) for e in self.epsilons]
        self.collapse_min_macro_f1 = float(self.collapse_min_macro_f1)
        self.collapse_min_attack_recall = float(self.collapse_min_attack_recall)


@dataclass
class ExperimentConfig:
    seed: int = 42
    out_dir: str = "outputs"
    archs: List[str] = field(default_factory=lambda: list(KNOWN_ARCHS))
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> "ExperimentConfig":
        if not self.archs:
            raise ValueError("No architectures selected.")
        for a in self.archs:
            if a not in KNOWN_ARCHS:
                raise ValueError(f"Unknown arch '{a}'. Choose from {KNOWN_ARCHS}.")
        if self.data.sample_mode not in SAMPLE_MODES:
            raise ValueError(f"data.sample_mode must be one of {SAMPLE_MODES}.")
        if not self.eval.threat_models:
            raise ValueError("eval.threat_models must not be empty.")
        for t in self.eval.threat_models:
            if t not in THREAT_MODELS:
                raise ValueError(f"eval.threat_models entries must be in {THREAT_MODELS}, got '{t}'.")
        if not 0.0 <= self.train.alpha + self.train.beta < 1.0:
            raise ValueError("train.alpha + train.beta must be in [0, 1) so the CE term keeps positive weight.")
        if len(self.train.teacher_epsilon_range) != 2 or \
                self.train.teacher_epsilon_range[0] > self.train.teacher_epsilon_range[1]:
            raise ValueError("train.teacher_epsilon_range must be [low, high] with low <= high.")
        if not self.train.student_epsilons:
            raise ValueError("train.student_epsilons must not be empty.")
        if not self.data.perturbable_features:
            raise ValueError("data.perturbable_features is empty: the attacker could not change anything, "
                             "so every robustness number would be meaningless.")
        if not 0.0 < self.data.cap_quantile <= 1.0:
            raise ValueError("data.cap_quantile must be in (0, 1].")
        if not 0.0 < self.data.holdout_fraction < 1.0:
            raise ValueError("data.holdout_fraction must be in (0, 1).")
        if not 0.0 < self.data.val_share_of_holdout < 1.0:
            raise ValueError("data.val_share_of_holdout must be in (0, 1).")
        if self.train.epochs < 1:
            raise ValueError("train.epochs must be >= 1.")
        for key in ("label_columns", "benign_labels", "drop_columns", "categorical_columns",
                    "cap_columns", "nonnegative_columns", "perturbable_features"):
            bad = [v for v in getattr(self.data, key) if not isinstance(v, str)]
            if bad:
                raise ValueError(f"data.{key} must contain only strings, got {bad}. "
                                 f"Quote entries that contain ': ' in the YAML.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _from_dict(cls, d: Dict[str, Any]):
    hints = get_type_hints(cls)
    unknown = set(d) - set(hints)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
                         f"Valid keys: {sorted(hints)}")
    kwargs = {}
    for k, v in d.items():
        t = hints[k]
        if is_dataclass(t):
            if v is None:
                v = {}
            if not isinstance(v, dict):
                raise ValueError(f"Config section '{k}' must be a mapping.")
            kwargs[k] = _from_dict(t, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return data


def load_config(dataset_config: str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load ``default.yaml`` (next to the dataset YAML), merge the dataset YAML, then overrides.

    ``overrides`` is a nested dict, e.g. ``{"train": {"epochs": 2}, "data": {"raw_dir": "x"}}``.
    """
    path = Path(dataset_config)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    default_path = path.parent / "default.yaml"
    merged: Dict[str, Any] = {}
    if default_path.exists() and default_path.resolve() != path.resolve():
        merged = _read_yaml(default_path)
    merged = _deep_merge(merged, _read_yaml(path))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return _from_dict(ExperimentConfig, merged).validate()
