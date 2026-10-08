"""Tiny end to end runs: train every stage, evaluate, write JSON and tables."""
import json
import os

import pytest

from conftest import CONFIGS
from mtkd_adr.run import main as run_main
from mtkd_adr.tables import flatten, make_tables


@pytest.mark.slow
@pytest.mark.parametrize("name,fixture", [("cicids2017", "cic2017_dir"), ("ciciot2023", "ciciot_dir")])
def test_full_run(name, fixture, request, tmp_path):
    raw = request.getfixturevalue(fixture)
    out = str(tmp_path / "out")
    args = ["--config", os.path.join(CONFIGS, f"{name}.yaml"), "--raw-dir", raw, "--out-dir", out,
            "--arch", "cnn,lstm,clstm", "--epochs", "1", "--sample-size", "3000"]
    # shrink attacks for speed through a temp override config
    import yaml
    cfgp = tmp_path / "cfg"
    cfgp.mkdir()
    with open(os.path.join(CONFIGS, "default.yaml")) as f:
        d = yaml.safe_load(f)
    d["attack"].update({"pgd_steps": 3, "ifgsm_steps": 3})
    # one epoch on tiny data legitimately collapses; this test checks plumbing, not accuracy
    d["eval"].update({"epsilons": [0.1], "collapse_min_macro_f1": 0.0, "collapse_min_attack_recall": 0.0})
    d["train"].update({"batch_size": 128})
    with open(cfgp / "default.yaml", "w") as f:
        yaml.safe_dump(d, f)
    with open(os.path.join(CONFIGS, f"{name}.yaml")) as f:
        ds = yaml.safe_load(f)
    ds["data"]["rows_per_file"] = None
    with open(cfgp / f"{name}.yaml", "w") as f:
        yaml.safe_dump(ds, f)
    args[1] = str(cfgp / f"{name}.yaml")

    res_path = run_main(args)
    with open(res_path) as f:
        res = json.load(f)
    assert res["_meta"]["perturbation_constrained_to"]
    for arch in ("cnn", "lstm", "clstm"):
        assert set(res[arch]["models"]) == {"baseline", "mtkd_adr"}
        assert set(res[arch]["teachers"]) == {"teacher_fgsm", "teacher_ifgsm", "teacher_pgd"}
        cell = res[arch]["models"]["baseline"]["attacks"]["masked"]["pgd"]["0.1"]
        assert cell["perturbed_fraction"] > 0.0, "attack must actually perturb samples"

    # second invocation reuses checkpoints and eval only works
    run_main(args + ["--eval-only"])

    df = flatten(res)
    assert {"clean", "fgsm", "ifgsm", "pgd"} <= set(df["attack"])
    make_tables([res_path], str(tmp_path / "tables"))
    assert (tmp_path / "tables" / "results_long.csv").exists()
    assert (tmp_path / "tables" / "tables_attack_recall.tex").read_text().count(r"\begin{table}") >= 3
