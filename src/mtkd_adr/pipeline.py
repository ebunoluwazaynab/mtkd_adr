"""Per architecture driver: baseline, teachers, MTKD student, evaluation."""
from __future__ import annotations

import json
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from .attacks import ThreatModel, make_attacks
from .config import ExperimentConfig
from .data.preprocess import PreparedData
from .evaluate import evaluate_all
from .models import RECURRENT, build_model
from .training import harden_teacher, train_baseline, train_mtkd_student

TEACHER_NAMES = ("fgsm", "ifgsm", "pgd")


def load_ckpt(arch: str, num_features: int, path: str, device) -> torch.nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run without --eval-only first.")
    m = build_model(arch, num_features).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


def _check_meta(folder: str, arch: str, data: PreparedData, force: bool) -> None:
    """Refuse to reuse checkpoints that were trained on different data."""
    meta_path = os.path.join(folder, "meta.json")
    current = {"arch": arch, "n_features": data.n_features, "fingerprint": data.fingerprint}
    if os.path.exists(meta_path) and not force:
        with open(meta_path) as f:
            old = json.load(f)
        if old != current:
            raise RuntimeError(
                f"Existing checkpoints in {folder} were trained with {old} but the current data is "
                f"{current}. Use --force to retrain, or choose a different --out-dir.")
    with open(meta_path, "w") as f:
        json.dump(current, f, indent=2)


def run_architecture(arch: str, data: PreparedData, cfg: ExperimentConfig, device,
                     eval_only: bool = False, force: bool = False) -> Dict:
    recurrent = arch in RECURRENT
    n_features = data.n_features
    threat = ThreatModel(feature_indices=tuple(data.feature_indices), n_features=n_features,
                         mape_threshold=cfg.attack.mape_threshold, disable_cudnn=recurrent)

    folder = os.path.join(cfg.out_dir, cfg.data.name, arch)
    os.makedirs(folder, exist_ok=True)
    paths = {k: os.path.join(folder, f"{k}.pth") for k in
             ["baseline", "teacher_fgsm", "teacher_ifgsm", "teacher_pgd", "mtkd_student"]}

    print("\n" + "#" * 72)
    print(f"#  DATASET: {cfg.data.name.upper()}   ARCHITECTURE: {arch.upper()}   (recurrent={recurrent})")
    print("#" * 72)

    bs = cfg.train.batch_size
    val_loader = DataLoader(TensorDataset(data.X_val, data.y_val), batch_size=bs, shuffle=False)
    test_loader = DataLoader(TensorDataset(data.X_test, data.y_test), batch_size=bs, shuffle=False)

    if eval_only:
        baseline = load_ckpt(arch, n_features, paths["baseline"], device)
        student = load_ckpt(arch, n_features, paths["mtkd_student"], device)
        teachers = {n: load_ckpt(arch, n_features, paths[f"teacher_{n}"], device) for n in TEACHER_NAMES}
    else:
        _check_meta(folder, arch, data, force)
        train_loader = DataLoader(TensorDataset(data.X_train, data.y_train), batch_size=bs, shuffle=True)

        def reuse(key: str) -> bool:
            if os.path.exists(paths[key]) and not force:
                print(f"[skip] {key} exists: {paths[key]}")
                return True
            return False

        if reuse("baseline"):
            baseline = load_ckpt(arch, n_features, paths["baseline"], device)
        else:
            baseline = train_baseline(arch, n_features, train_loader, val_loader, cfg, device)
            torch.save(baseline.state_dict(), paths["baseline"])
        baseline_state = {k: v.clone() for k, v in baseline.state_dict().items()}

        perm = torch.randperm(len(data.X_train))
        split = len(perm) // 2
        a, b = perm[:split], perm[split:]
        clean_loader = DataLoader(TensorDataset(data.X_train[a], data.y_train[a]), batch_size=bs, shuffle=True)
        adv_loader = DataLoader(TensorDataset(data.X_train[b], data.y_train[b]), batch_size=bs, shuffle=True)

        teachers = {}
        for name, atk in make_attacks(cfg.attack).items():
            key = f"teacher_{name}"
            if reuse(key):
                teachers[name] = load_ckpt(arch, n_features, paths[key], device)
            else:
                t = harden_teacher(arch, n_features, baseline_state, atk, name, clean_loader, adv_loader,
                                   val_loader, threat, cfg, device)
                torch.save(t.state_dict(), paths[key])
                teachers[name] = t

        if reuse("mtkd_student"):
            student = load_ckpt(arch, n_features, paths["mtkd_student"], device)
        else:
            student = train_mtkd_student(arch, n_features, teachers, train_loader, val_loader,
                                         threat, cfg, device)
            torch.save(student.state_dict(), paths["mtkd_student"])

    models = {"baseline": baseline, "mtkd_adr": student}
    results = evaluate_all(models, test_loader, threat, cfg, device)
    teacher_results = evaluate_all({f"teacher_{k}": v for k, v in teachers.items()},
                                   test_loader, threat, cfg, device)
    return {"models": results, "teachers": teacher_results}
