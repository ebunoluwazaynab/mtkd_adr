import os

import pytest

from mtkd_adr.config import load_config
from conftest import CONFIGS


@pytest.mark.parametrize("name", ["cicids2017", "ciciot2023"])
def test_shipped_configs_load(name):
    cfg = load_config(os.path.join(CONFIGS, f"{name}.yaml"))
    assert cfg.data.name == name
    assert cfg.data.perturbable_features
    assert cfg.train.lr == 0.001 and isinstance(cfg.train.lr, float)


def test_overrides_and_unknown_keys(tmp_path):
    cfg = load_config(os.path.join(CONFIGS, "cicids2017.yaml"), {"train": {"epochs": 7}})
    assert cfg.train.epochs == 7
    with pytest.raises(ValueError):
        load_config(os.path.join(CONFIGS, "cicids2017.yaml"), {"train": {"epochz": 7}})


def test_validation_rejects_bad_values():
    p = os.path.join(CONFIGS, "cicids2017.yaml")
    with pytest.raises(ValueError):
        load_config(p, {"archs": ["transformer"]})
    with pytest.raises(ValueError):
        load_config(p, {"train": {"alpha": 0.7, "beta": 0.5}})
    with pytest.raises(ValueError):
        load_config(p, {"data": {"perturbable_features": []}})


def test_non_string_column_entries_rejected():
    with pytest.raises(ValueError, match="only strings"):
        load_config(os.path.join(CONFIGS, "cicids2017.yaml"), {"data": {"drop_columns": [{"Unnamed": 0}]}})
