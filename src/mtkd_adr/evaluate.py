"""Metrics, collapse detection and evaluation under attack."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

from .attacks import ThreatModel, craft_adv, make_attacks, make_fmodel
from .config import ExperimentConfig
from .constants import ATTACK_LABEL, BENIGN_LABEL


def compute_metrics(y_true, y_pred) -> Dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "attack_recall": recall_score(y_true, y_pred, pos_label=ATTACK_LABEL, zero_division=0),
        "benign_recall": recall_score(y_true, y_pred, pos_label=BENIGN_LABEL, zero_division=0),
        "attack_precision": precision_score(y_true, y_pred, pos_label=ATTACK_LABEL, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, labels=[ATTACK_LABEL, BENIGN_LABEL],
                                        target_names=["Attack", "Benign"],
                                        output_dict=True, zero_division=0),
    }


@torch.no_grad()
def evaluate_clean(model, loader, device) -> Dict:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = (torch.sigmoid(model(x).squeeze(-1)) > 0.5).long().cpu().numpy()
        preds.extend(p.tolist())
        labels.extend(y.numpy().tolist())
    return compute_metrics(labels, preds)


def evaluate_under_attack(model, loader, attack, epsilon: float, threat: ThreatModel, device,
                          desc: str = "") -> Dict:
    model.eval()
    fmodel = make_fmodel(model, device)
    preds, labels = [], []
    n_mal, n_changed = 0, 0
    for x, y in tqdm(loader, desc=desc or f"  {attack.__class__.__name__} eps={epsilon}", leave=False):
        x, y = x.to(device), y.to(device)
        adv = craft_adv(attack, fmodel, x, y, epsilon, threat)
        mal = y == ATTACK_LABEL
        n_mal += int(mal.sum())
        n_changed += int((adv[mal] != x[mal]).any(dim=1).sum())
        with torch.no_grad():
            p = (torch.sigmoid(model(adv).squeeze(-1)) > 0.5).long().cpu().numpy()
        preds.extend(p.tolist())
        labels.extend(y.cpu().numpy().tolist())
    m = compute_metrics(labels, preds)
    m["n_malicious"] = n_mal
    # Fraction of malicious rows the attacker actually managed to change. If this is 0
    # the attack was a no op and the robustness number is meaningless.
    m["perturbed_fraction"] = (n_changed / n_mal) if n_mal else 0.0
    return m


def check_collapse(model, val_loader, name: str, cfg: ExperimentConfig, device) -> Tuple[bool, Dict]:
    """A model that predicts only the majority class shows near zero clean attack recall.

    This is not an attack effect, so it is checked on CLEAN validation data.
    """
    m = evaluate_clean(model, val_loader, device)
    collapsed = (m["macro_f1"] < cfg.eval.collapse_min_macro_f1 or
                 m["attack_recall"] < cfg.eval.collapse_min_attack_recall)
    print(f"  [collapse-check] {name}: clean Macro-F1={m['macro_f1']:.4f} "
          f"Attack-Recall={m['attack_recall']:.4f} -> {'COLLAPSED' if collapsed else 'ok'}")
    if collapsed:
        print(f"  *** WARNING: {name} looks like a degenerate majority class predictor on CLEAN data. "
              f"Do not trust results or distil from it. Try a lower lr, more epochs or a smaller "
              f"epsilon range. ***")
    return collapsed, m


def _threat_for(name: str, base: ThreatModel, mape_threshold) -> ThreatModel:
    if name == "masked":
        return base.without_mape()
    if name == "masked_mape":
        return base.with_mape(mape_threshold)
    raise ValueError(f"Unknown threat model '{name}'.")


def evaluate_all(models: Dict[str, torch.nn.Module], test_loader, base_threat: ThreatModel,
                 cfg: ExperimentConfig, device) -> Dict:
    """Returns {model: {'clean': metrics, 'attacks': {threat: {attack: {eps: metrics}}}}}."""
    print("\n" + "=" * 72)
    print("  EVALUATION: clean + white box attacks")
    print("=" * 72)
    results = {}
    for name, model in models.items():
        print(f"\n>>> Model: {name}")
        clean = evaluate_clean(model, test_loader, device)
        results[name] = {"clean": clean, "attacks": {}}
        print(f"    clean | Acc {clean['accuracy']*100:.2f}% | Attack-Recall {clean['attack_recall']:.4f} "
              f"| Benign-Recall {clean['benign_recall']:.4f}")
        for threat_name in cfg.eval.threat_models:
            threat = _threat_for(threat_name, base_threat, cfg.attack.mape_threshold)
            results[name]["attacks"][threat_name] = {}
            for atk_name, atk in make_attacks(cfg.attack).items():
                results[name]["attacks"][threat_name][atk_name] = {}
                for eps in cfg.eval.epsilons:
                    m = evaluate_under_attack(model, test_loader, atk, eps, threat, device,
                                              desc=f"  {threat_name}/{atk_name} eps={eps}")
                    results[name]["attacks"][threat_name][atk_name][str(eps)] = m
                    print(f"    {threat_name:<11} {atk_name:<5} eps={eps:<4} | Acc {m['accuracy']*100:5.2f}% "
                          f"| Attack-Recall {m['attack_recall']:.4f} | Benign-Recall {m['benign_recall']:.4f} "
                          f"| perturbed {m['perturbed_fraction']*100:5.1f}%")
                    if m["n_malicious"] > 0 and m["perturbed_fraction"] == 0.0:
                        print("      [warn] attack changed no malicious sample: check perturbable_features "
                              "and mape_threshold, results for this cell are not meaningful.")
    return results
