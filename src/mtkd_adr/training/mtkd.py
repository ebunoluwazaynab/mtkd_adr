"""Stage 3: multi teacher knowledge distillation with adversarial data (MTKD ADR)."""
from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..attacks import ThreatModel, craft_adv, make_attacks, make_fmodel
from ..config import ExperimentConfig
from ..evaluate import check_collapse
from ..models import build_model
from .common import report_val


def compute_adaptive_weights(student_logits: torch.Tensor, teacher_logits_list: List[torch.Tensor]):
    """Teachers whose logits agree more with the student get a larger weight."""
    sims = [F.cosine_similarity(student_logits, t, dim=0).mean() for t in teacher_logits_list]
    weights = [1.0 + s for s in sims]
    total = sum(weights)
    return [w / total for w in weights]


def distillation_loss(student_logits, teacher_logits_list, weights, T: float):
    s_soft = torch.sigmoid(student_logits / T)
    s_probs = torch.cat([1 - s_soft, s_soft], dim=1)
    s_log = torch.log(s_probs.clamp(min=1e-8))
    weighted_t = torch.zeros_like(s_soft)
    for t_logits, w in zip(teacher_logits_list, weights):
        weighted_t = weighted_t + w * torch.sigmoid(t_logits / T)
    t_probs = torch.cat([1 - weighted_t, weighted_t], dim=1)
    return F.kl_div(s_log, t_probs.clamp(min=1e-8), reduction="batchmean") * (T ** 2)


def feature_distillation_loss(student_features, teacher_features_list, weights):
    loss = torch.zeros((), device=student_features.device)
    for t_feat, w in zip(teacher_features_list, weights):
        loss = loss + w * F.mse_loss(student_features, t_feat.detach())
    return loss


def train_mtkd_student(arch, num_features, teachers: Dict[str, nn.Module], train_loader, val_loader,
                       threat: ThreatModel, cfg: ExperimentConfig, device, skip_collapsed: bool = True):
    print("\n[Stage 3] Training MTKD ADR student...")
    t = cfg.train

    # Never silently distil from a collapsed teacher: it would teach "always predict benign".
    usable: Dict[str, nn.Module] = {}
    for name, teacher in teachers.items():
        collapsed, _ = check_collapse(teacher, val_loader, f"teacher_{name} (pre MTKD check)", cfg, device)
        if collapsed and skip_collapsed:
            print(f"  [MTKD] excluding collapsed teacher_{name} from the ensemble")
            continue
        usable[name] = teacher
    if not usable:
        raise RuntimeError("All teachers collapsed, cannot train the MTKD student. Fix teacher hardening "
                           "(lr, epochs, epsilon range) before retrying.")

    student = build_model(arch, num_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(student.parameters(), lr=t.lr, weight_decay=t.l2_lambda)

    names = list(usable.keys())
    teacher_list = list(usable.values())
    for teacher in teacher_list:
        teacher.eval()
    fmodels = [make_fmodel(teacher, device) for teacher in teacher_list]
    all_attacks = make_attacks(cfg.attack)
    attacks = [all_attacks[n] for n in names]   # attack i is crafted against teacher i
    train_threat = threat.with_mape(cfg.attack.mape_threshold)
    n_streams = 1 + len(teacher_list)           # clean + one adversarial stream per teacher

    for epoch in range(t.epochs):
        student.train()
        pbar = tqdm(train_loader, desc=f"[MTKD] Epoch {epoch+1}/{t.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            epsilon = random.choice(t.student_epsilons)

            adv_batches = [craft_adv(atk, fm, inputs, labels, epsilon, train_threat)
                           for atk, fm in zip(attacks, fmodels)]
            all_x = torch.cat([inputs] + adv_batches, dim=0)
            all_y = torch.cat([labels] * n_streams, dim=0)

            optimizer.zero_grad()
            s_logits, s_feats = student(all_x, return_features=True)
            with torch.no_grad():
                t_logits_list, t_feats_list = [], []
                for teacher in teacher_list:
                    tl, tf = teacher(all_x, return_features=True)
                    t_logits_list.append(tl)
                    t_feats_list.append(tf)

            weights = compute_adaptive_weights(s_logits.detach(), t_logits_list)
            ce = criterion(s_logits.squeeze(-1), all_y.float())
            kd = distillation_loss(s_logits, t_logits_list, weights, t.temperature)
            feat = feature_distillation_loss(s_feats, t_feats_list, weights)
            loss = (1 - t.alpha - t.beta) * ce + t.alpha * kd + t.beta * feat
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ce": f"{ce.item():.3f}",
                              "kd": f"{kd.item():.3f}", "feat": f"{feat.item():.3f}"})
        report_val(student, val_loader, epoch, device)
    check_collapse(student, val_loader, "mtkd_adr", cfg, device)
    return student
