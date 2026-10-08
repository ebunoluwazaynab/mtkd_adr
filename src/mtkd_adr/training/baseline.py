"""Stage 1: undefended baseline."""
from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..config import ExperimentConfig
from ..evaluate import check_collapse
from ..models import build_model
from .common import l1_penalty, report_val


def train_baseline(arch, num_features, train_loader, val_loader, cfg: ExperimentConfig, device):
    print("\n[Stage 1] Training baseline (undefended)...")
    t = cfg.train
    model = build_model(arch, num_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=t.lr, weight_decay=t.l2_lambda)
    for epoch in range(t.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Baseline] Epoch {epoch+1}/{t.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(inputs).squeeze(-1)
            loss = criterion(out, labels.float()) + t.l1_lambda * l1_penalty(model)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        report_val(model, val_loader, epoch, device)
    check_collapse(model, val_loader, "baseline", cfg, device)
    return model
