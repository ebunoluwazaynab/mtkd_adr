"""Synthetic CIC style data so tests run without the real datasets."""
import os

import numpy as np
import pandas as pd
import pytest

from mtkd_adr.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "configs")


def make_cic2017_frame(n=600, seed=0, attack_frac=0.4):
    rng = np.random.default_rng(seed)
    is_attack = rng.random(n) < attack_frac
    shift = np.where(is_attack, 1.0, 0.0)[:, None]
    timing = ["Flow Duration", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Flow IAT Std",
              "Fwd IAT Tot", "Fwd IAT Mean", "Bwd IAT Tot", "Active Mean", "Idle Mean", "Idle Max"]
    other = ["Tot Fwd Pkts", "Tot Bwd Pkts", "Pkt Len Mean", "Pkt Len Std"]
    df = pd.DataFrame(rng.exponential(1.0, size=(n, len(timing) + len(other))) + shift,
                      columns=timing + other)
    df["Protocol"] = rng.choice([6, 17, 1], size=n)
    df["SYN Flag Cnt"] = rng.integers(0, 2, size=n)
    df["Constant Col"] = 5
    df["Src IP"] = "10.0.0.1"
    df["Timestamp"] = "2017-07-07"
    df["Label"] = np.where(is_attack, "DoS", "BENIGN")
    df.columns = [" " + c if c == "Flow Duration" else c for c in df.columns]  # leading space quirk
    return df


def make_ciciot_frame(n=600, seed=1, attack_frac=0.6):
    rng = np.random.default_rng(seed)
    is_attack = rng.random(n) < attack_frac
    shift = np.where(is_attack, 1.0, 0.0)[:, None]
    cols = ["flow_duration", "Header_Length", "Rate", "Srate", "Drate", "IAT", "Tot sum", "Min", "Max", "AVG"]
    df = pd.DataFrame(rng.exponential(1.0, size=(n, len(cols))) + shift, columns=cols)
    df["syn_flag_number"] = rng.integers(0, 2, size=n)
    df["label"] = np.where(is_attack, "DDoS-SYN_Flood", "BenignTraffic")
    return df


@pytest.fixture
def cic2017_dir(tmp_path):
    d = tmp_path / "cic2017"
    d.mkdir()
    make_cic2017_frame(400, 0).to_csv(d / "a.csv", index=False)
    make_cic2017_frame(400, 1).to_csv(d / "b.csv", index=False)
    return str(d)


@pytest.fixture
def ciciot_dir(tmp_path):
    d = tmp_path / "ciciot"
    d.mkdir()
    make_ciciot_frame(500, 0).to_csv(d / "p1.csv", index=False)
    make_ciciot_frame(500, 1).to_csv(d / "p2.csv", index=False)
    return str(d)


def cfg_for(name, raw_dir, **train):
    ov = {"data": {"raw_dir": raw_dir, "sample_size": 5000, "rows_per_file": None},
          "train": {"epochs": 1, "batch_size": 64, **train},
          "attack": {"pgd_steps": 3, "ifgsm_steps": 3},
          "eval": {"epsilons": [0.1], "threat_models": ["masked", "masked_mape"]}}
    return load_config(os.path.join(CONFIGS, f"{name}.yaml"), ov)
