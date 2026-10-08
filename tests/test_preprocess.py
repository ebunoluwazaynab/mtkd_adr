import numpy as np
import pandas as pd
import pytest
import torch

from conftest import cfg_for
from mtkd_adr.constants import ATTACK_LABEL, BENIGN_LABEL
from mtkd_adr.data.preprocess import prepare_data


def test_cic2017_pipeline(cic2017_dir):
    cfg = cfg_for("cicids2017", cic2017_dir)
    d = prepare_data(cfg.data, cfg.seed)
    assert d.X_train.shape[1] == d.X_val.shape[1] == d.X_test.shape[1] == len(d.feature_names)
    for X in (d.X_train, d.X_val, d.X_test):
        assert float(X.min()) >= 0.0 and float(X.max()) <= 1.0
    assert set(d.y_train.tolist()) == {ATTACK_LABEL, BENIGN_LABEL}
    assert d.y_train.dtype == torch.long
    # train is oversampled to balance
    assert int((d.y_train == 0).sum()) == int((d.y_train == 1).sum())
    # constant / identifier columns removed
    assert "Constant Col" not in d.feature_names and "Src IP" not in d.feature_names
    # leading whitespace in ' Flow Duration' was stripped and the feature is perturbable
    assert "Flow Duration" in d.feature_names
    names = [d.feature_names[i] for i in d.feature_indices]
    assert "Flow Duration" in names and "Idle Max" in names
    # categoricals are one hot, never perturbable
    assert any(n.startswith("Protocol_") for n in d.feature_names)
    assert not any(n.startswith("Protocol_") for n in names)
    assert "Pkt Len Mean" not in names


def test_label_encoding_is_explicit(cic2017_dir):
    """Attack must map to 0 regardless of how the label strings sort alphabetically."""
    cfg = cfg_for("cicids2017", cic2017_dir)
    d = prepare_data(cfg.data, cfg.seed)
    raw = pd.concat([pd.read_csv(f"{cic2017_dir}/a.csv"), pd.read_csv(f"{cic2017_dir}/b.csv")])
    frac_attack_raw = (raw["Label"] != "BENIGN").mean()
    frac_attack_test = float((d.y_test == ATTACK_LABEL).float().mean())
    assert abs(frac_attack_raw - frac_attack_test) < 0.1


def test_ciciot_pipeline(ciciot_dir):
    cfg = cfg_for("ciciot2023", ciciot_dir)
    d = prepare_data(cfg.data, cfg.seed)
    names = [d.feature_names[i] for i in d.feature_indices]
    assert set(names) == {"flow_duration", "Rate", "Srate", "Drate", "IAT"}
    assert int((d.y_test == ATTACK_LABEL).sum()) > 0 and int((d.y_test == BENIGN_LABEL).sum()) > 0


def test_caps_are_fitted_on_train_only(cic2017_dir):
    """An extreme outlier placed only in the test split must not influence the train cap."""
    cfg = cfg_for("cicids2017", cic2017_dir)
    d = prepare_data(cfg.data, cfg.seed)
    i = d.feature_names.index("Flow Duration")
    # val/test may exceed the train range before clamping but are clamped to <= 1
    assert float(d.X_train[:, i].max()) <= 1.0
    assert float(d.X_test[:, i].max()) <= 1.0


def test_missing_perturbable_features_raises(ciciot_dir):
    cfg = cfg_for("ciciot2023", ciciot_dir)
    cfg.data.perturbable_features = ["Flow IAT Mean"]  # CICIDS name on a CICIoT dataset
    with pytest.raises(ValueError, match="no ops"):
        prepare_data(cfg.data, cfg.seed)


def test_wrong_benign_label_raises(ciciot_dir):
    cfg = cfg_for("ciciot2023", ciciot_dir)
    cfg.data.benign_labels = ["nothing"]
    with pytest.raises(ValueError, match="only one class"):
        prepare_data(cfg.data, cfg.seed)


def test_missing_raw_dir_raises():
    from mtkd_adr.config import load_config
    from conftest import CONFIGS
    import os
    cfg = load_config(os.path.join(CONFIGS, "ciciot2023.yaml"))
    with pytest.raises(ValueError, match="raw_dir"):
        prepare_data(cfg.data, cfg.seed)
