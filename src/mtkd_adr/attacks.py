"""White box attacks and the single adversarial crafting routine used everywhere.

One function, :func:`craft_adv`, is used by teacher hardening, student distillation and
evaluation, so training and evaluation always share the same threat model definition.

Threat model
------------
* Only malicious samples are perturbed (the attacker wants to hide attacks).
* Only ``feature_indices`` (flow timing features) may change; every other column,
  including one hot encoded categoricals, is restored to its clean value.
* Optional realism check (``mape_threshold``): if any non near zero feature of a
  perturbed row moved by more than the threshold percent, the whole row is rolled
  back to clean. Note this makes the attack WEAKER, which is why evaluation reports
  the unclamped ``masked`` threat model as well.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence

import foolbox as fb
import torch
import torch.nn as nn
from foolbox.attacks import FGSM, PGD, LinfBasicIterativeAttack

from .config import AttackConfig
from .constants import ATTACK_LABEL, BENIGN_LABEL, BOUNDS  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class ThreatModel:
    feature_indices: Sequence[int]
    n_features: int
    mape_threshold: Optional[float] = None
    disable_cudnn: bool = False   # required for LSTM backward while in eval mode on GPU

    def without_mape(self) -> "ThreatModel":
        return replace(self, mape_threshold=None)

    def with_mape(self, threshold: Optional[float]) -> "ThreatModel":
        return replace(self, mape_threshold=threshold)


def make_attacks(cfg: AttackConfig) -> Dict[str, object]:
    return {
        "fgsm": FGSM(),
        "ifgsm": LinfBasicIterativeAttack(abs_stepsize=cfg.ifgsm_abs_stepsize, steps=cfg.ifgsm_steps),
        "pgd": PGD(rel_stepsize=cfg.pgd_rel_stepsize, steps=cfg.pgd_steps,
                   random_start=cfg.pgd_random_start),
    }


class BinaryWrapper(nn.Module):
    """Turns a single logit into two class logits [attack, benign] for foolbox."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit = self.model(x).squeeze(-1)
        return torch.stack([-logit, logit], dim=1)


def make_fmodel(model: nn.Module, device: torch.device) -> fb.PyTorchModel:
    """Foolbox model that SHARES weights with ``model`` (so it always sees the latest weights)."""
    wrapped = BinaryWrapper(model).to(device)
    was_training = model.training
    wrapped.eval()   # foolbox warns if handed a module in training mode
    fmodel = fb.PyTorchModel(wrapped, bounds=BOUNDS)
    model.train(was_training)   # constructing the foolbox model must not change the caller's mode
    # Keep our own handle: foolbox does not expose the wrapped module publicly.
    fmodel.torch_module = wrapped
    return fmodel


def craft_adv(attack, fmodel: fb.PyTorchModel, inputs: torch.Tensor, labels: torch.Tensor,
              epsilon: float, threat: ThreatModel) -> torch.Tensor:
    """Return ``inputs`` with malicious rows adversarially perturbed under ``threat``.

    The model is put in eval mode while the attack runs (BatchNorm must use running
    statistics, otherwise the attack both corrupts them and attacks a different network
    than the one that is evaluated) and its previous mode is restored afterwards.
    """
    adv = inputs.clone()
    is_mal = labels == ATTACK_LABEL
    if int(is_mal.sum()) == 0:
        return adv

    x_mal = inputs[is_mal]
    y_mal = labels[is_mal].long()

    net = fmodel.torch_module          # BinaryWrapper, always kept in eval mode itself
    inner = net.model                  # the real network whose mode the caller controls
    was_training = inner.training
    net.eval()
    cudnn_cm = (torch.backends.cudnn.flags(enabled=False)
                if threat.disable_cudnn else contextlib.nullcontext())
    try:
        with cudnn_cm, torch.enable_grad():
            _, adv_mal, _ = attack(fmodel, x_mal, y_mal, epsilons=epsilon)
    finally:
        inner.train(was_training)

    adv_mal = adv_mal.detach()
    mask = torch.zeros(threat.n_features, device=adv_mal.device, dtype=adv_mal.dtype)
    mask[list(threat.feature_indices)] = 1.0
    adv_mal = x_mal + (adv_mal - x_mal) * mask

    if threat.mape_threshold is not None:
        nonzero = x_mal.abs() > 1e-2
        mape = torch.abs((adv_mal - x_mal) / (x_mal.abs() + 1e-8)) * 100.0
        ok = ((mape <= threat.mape_threshold) | ~nonzero).all(dim=1)
        adv_mal = torch.where(ok.unsqueeze(1), adv_mal, x_mal)

    adv[is_mal] = torch.clamp(adv_mal, BOUNDS[0], BOUNDS[1])
    return adv
