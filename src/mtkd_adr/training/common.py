"""Helpers shared by the training stages."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..evaluate import evaluate_clean


def l1_penalty(model: nn.Module) -> torch.Tensor:
    return sum(p.abs().sum() for p in model.parameters())


def report_val(model, val_loader, epoch: int, device) -> None:
    m = evaluate_clean(model, val_loader, device)
    print(f"  Epoch {epoch+1}: Val Acc {m['accuracy']*100:.2f}% | Macro-F1 {m['macro_f1']:.4f} | "
          f"Attack-Recall {m['attack_recall']:.4f} | Benign-Recall {m['benign_recall']:.4f}")
