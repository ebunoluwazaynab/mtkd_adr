# MTKD ADR

Multi teacher knowledge distillation for adversarially robust network intrusion detection.

A student detector is distilled from several teachers, each hardened against a different white box evasion attack (FGSM, I-FGSM, PGD). The attacker is restricted to realistic changes: only flow timing features may be perturbed, and perturbations that move a feature by more than a set percentage can be rejected as unrealistic. The code runs on any CIC style flow dataset through a YAML file, with CIC-IDS2017 and CICIoT2023 included.

## Method in brief

1. **Baseline.** An undefended CNN, LSTM or CNN-LSTM trained for binary classification (attack vs benign).
2. **Teachers.** For each attack, the baseline is fine tuned on a mix of clean batches and batches whose malicious rows are adversarially perturbed.
3. **Student.** A fresh model trained on clean data plus one adversarial stream per teacher, with a loss that mixes cross entropy, temperature scaled logit distillation and feature distillation. Teacher weights adapt to how closely each teacher agrees with the student.
4. **Evaluation.** Baseline, teachers and student are attacked on the held out test set under two threat models (below).

Teachers that collapse to a majority class predictor on clean validation data are detected, reported loudly, and excluded from distillation.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # runs on synthetic data, no datasets needed
```

For GPU training install a CUDA build of PyTorch first, following pytorch.org.

## Data

Download the raw CSVs yourself and point `--raw-dir` at the folder. Subfolders are searched recursively.

| Dataset | Config | Label column | Benign label |
|---|---|---|---|
| CIC-IDS2017 | `configs/cicids2017.yaml` | `Label` | `BENIGN` |
| CICIoT2023 | `configs/ciciot2023.yaml` | `label` | `BenignTraffic` |

Sources: https://www.unb.ca/cic/datasets/ids-2017.html and https://www.unb.ca/cic/datasets/iotdataset-2023.html

CICIoT2023 is very large, so its config keeps at most 5000 rows per CSV while loading (`rows_per_file`). Set it to `null` to load everything.

## Run

```bash
python scripts/run_experiment.py --config configs/cicids2017.yaml --raw-dir data/cicids2017
python scripts/run_experiment.py --config configs/ciciot2023.yaml --raw-dir data/ciciot2023
```

Useful flags: `--arch cnn,lstm,clstm`, `--epochs 4`, `--sample-size 100000`, `--sample-mode stratified|balanced`, `--out-dir outputs`, `--seed 42`, `--force` (retrain even if checkpoints exist), `--eval-only` (load checkpoints and evaluate).

Outputs go to `outputs/`:

```
outputs/<dataset>/<arch>/{baseline,teacher_fgsm,teacher_ifgsm,teacher_pgd,mtkd_student}.pth
outputs/<dataset>_results.json      # all metrics plus full config, versions, seed, device
```

Checkpoints are tied to the data fingerprint and feature count. Reusing a folder with different data raises an error instead of silently mixing runs.

### Tables for the paper

```bash
python scripts/make_tables.py outputs/cicids2017_results.json outputs/ciciot2023_results.json \
    --metric attack_recall --out-dir outputs/tables
```

This writes `results_long.csv` (every metric for every cell), one CSV per table, and `tables_<metric>.tex`.

## Adding a dataset

Copy one of the files in `configs/`, change the name and set: the label column candidates, the benign labels, columns to drop, categorical columns, columns to cap, non negative columns, and `perturbable_features`. No Python changes are needed. The run fails with a clear message if none of the perturbable features exist, because the attacks would then do nothing and every robustness number would look perfect for the wrong reason. YAML entries that contain a colon and a space, such as `"Unnamed: 0"`, must be quoted.

## Threat model

Only malicious rows are perturbed, because the attacker wants to hide attacks. Only the columns listed in `perturbable_features` may change, which are timing and rate features an attacker can influence by delaying or spacing packets. Everything else, including one hot encoded protocol and flag columns, is restored to its clean value. Features live in [0, 1] after min max scaling and epsilon is an L infinity bound in that scaled space.

Two variants are evaluated and reported separately:

* `masked`: the feature restriction only. This is the strongest attacker.
* `masked_mape`: the feature restriction plus a realism rollback. Any perturbed row where a non near zero feature moved by more than `attack.mape_threshold` percent (default 20) is reverted to clean. This attacker is weaker, so these numbers are optimistic and should be read next to the `masked` ones.

Training uses the `masked_mape` variant. Attacks are always crafted with the model in eval mode so BatchNorm statistics are neither used in training mode nor corrupted. Each result cell also records `perturbed_fraction`, the share of malicious test rows the attacker actually changed. A value near zero means the attack was a no op for that cell.

## Reproducibility

The seed, config, library versions and device are stored in the results JSON. Splits and oversampling use the seed. GPU kernels can still introduce small run to run differences.

Data dependent preprocessing (constant column removal, percentile caps, scaling, one hot categories) is fitted on the training split only. Duplicate rows are removed before splitting so identical rows cannot appear in both train and test. Labels are encoded explicitly, attack as 0 and benign as 1.

## Layout

```
configs/            default.yaml plus one YAML per dataset
src/mtkd_adr/
  config.py         typed config, merging and validation
  data/             loading, subsampling, preprocessing
  models.py         CNN, LSTM, CNN-LSTM
  attacks.py        attack suite and the single craft_adv routine
  training/         baseline, teacher hardening, MTKD student
  evaluate.py       metrics, collapse check, evaluation under attack
  pipeline.py       per architecture driver with checkpoint handling
  run.py            command line entry point
  tables.py         JSON to CSV and LaTeX
scripts/            thin wrappers around run.py and tables.py
tests/              unit tests and tiny end to end runs on synthetic data
```

## License

MIT
