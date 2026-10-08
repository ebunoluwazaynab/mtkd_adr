"""Stage 2: one adversarially hardened teacher per attack."""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..attacks import ThreatModel, craft_adv, make_fmodel
from ..config import ExperimentConfig
from ..evaluate import check_collapse
from ..models import build_model
from .common import l1_penalty, report_val


def harden_teacher(arch, num_features, baseline_state, attack, attack_name, clean_loader, adv_loader,
                   val_loader, threat: ThreatModel, cfg: ExperimentConfig, device):
    print(f"\n[Stage 2] Hardening {attack_name.upper()} teacher...")
    t = cfg.train
    model = build_model(arch, num_features).to(device)
    model.load_state_dict(baseline_state)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=t.lr, weight_decay=t.l2_lambda)
    fmodel = make_fmodel(model, device)   # shares weights with `model`
    lo, hi = t.teacher_epsilon_range
    train_threat = threat.with_mape(cfg.attack.mape_threshold)

    for epoch in range(t.epochs):
        model.train()
        pbar = tqdm(zip(clean_loader, adv_loader),
                    desc=f"[{attack_name.upper()} Teacher] Epoch {epoch+1}/{t.epochs}",
                    total=min(len(clean_loader), len(adv_loader)))
        for (clean_x, clean_y), (adv_x, adv_y) in pbar:
            clean_x, clean_y = clean_x.to(device), clean_y.to(device)
            adv_x, adv_y = adv_x.to(device), adv_y.to(device)

            epsilon = random.uniform(lo, hi)
            # craft_adv runs the attack in eval mode and restores train mode afterwards.
            adv_pert = craft_adv(attack, fmodel, adv_x, adv_y, epsilon, train_threat)

            mixed_x = torch.cat([clean_x, adv_pert], dim=0)
            mixed_y = torch.cat([clean_y, adv_y], dim=0)

            optimizer.zero_grad()
            out = model(mixed_x).squeeze(-1)
            loss = criterion(out, mixed_y.float()) + t.l1_lambda * l1_penalty(model)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "eps": f"{epsilon:.3f}"})
        report_val(model, val_loader, epoch, device)
    check_collapse(model, val_loader, f"teacher_{attack_name}", cfg, device)
    return model
