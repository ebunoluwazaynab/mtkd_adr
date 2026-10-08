import torch
import torch.nn as nn

from mtkd_adr.attacks import ThreatModel, craft_adv, make_attacks, make_fmodel
from mtkd_adr.config import AttackConfig
from mtkd_adr.constants import ATTACK_LABEL, BENIGN_LABEL
from mtkd_adr.models import CNNModel

DEV = torch.device("cpu")
N_FEAT = 12
PERT = (1, 3, 4)


def _batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, N_FEAT, generator=g) * 0.8 + 0.1
    y = torch.tensor([ATTACK_LABEL, BENIGN_LABEL] * (n // 2))
    return x, y


def _model():
    torch.manual_seed(0)
    m = CNNModel(N_FEAT)
    # make it slightly trained so gradients are informative
    return m


def test_only_malicious_rows_and_only_masked_columns_change():
    x, y = _batch()
    m = _model().eval()
    fm = make_fmodel(m, DEV)
    threat = ThreatModel(PERT, N_FEAT, mape_threshold=None)
    for name, atk in make_attacks(AttackConfig(pgd_steps=5, ifgsm_steps=5)).items():
        adv = craft_adv(atk, fm, x, y, 0.1, threat)
        diff = (adv != x)
        assert not diff[y == BENIGN_LABEL].any(), name
        other = [i for i in range(N_FEAT) if i not in PERT]
        assert not diff[:, other].any(), name
        assert diff[y == ATTACK_LABEL][:, list(PERT)].any(), f"{name} was a no op"
        assert adv.min() >= 0.0 and adv.max() <= 1.0
        assert (adv - x).abs().max() <= 0.1 + 1e-6, name


def test_mape_rolls_back_large_changes():
    x, y = _batch()
    m = _model().eval()
    fm = make_fmodel(m, DEV)
    atk = make_attacks(AttackConfig())["fgsm"]
    free = craft_adv(atk, fm, x, y, 0.3, ThreatModel(PERT, N_FEAT, None))
    clamped = craft_adv(atk, fm, x, y, 0.3, ThreatModel(PERT, N_FEAT, 1.0))  # 1 percent
    assert (free != x).any()
    assert not (clamped != x).any(), "every row exceeds 1% change so all must be rolled back"


def test_batchnorm_statistics_untouched_and_mode_restored():
    x, y = _batch()
    m = _model()
    m.train()
    fm = make_fmodel(m, DEV)
    bn = m.cnn_backbone[1]
    before = bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone()
    craft_adv(make_attacks(AttackConfig())["pgd"], fm, x, y, 0.1, ThreatModel(PERT, N_FEAT, None))
    assert m.training, "training mode must be restored after crafting"
    assert torch.equal(before[0], bn.running_mean) and torch.equal(before[1], bn.running_var)
    assert torch.equal(before[2], bn.num_batches_tracked)


def test_no_malicious_rows_is_identity():
    x, _ = _batch()
    y = torch.full((len(x),), BENIGN_LABEL)
    m = _model().eval()
    fm = make_fmodel(m, DEV)
    adv = craft_adv(make_attacks(AttackConfig())["fgsm"], fm, x, y, 0.1, ThreatModel(PERT, N_FEAT, None))
    assert torch.equal(adv, x)
