import pytest
import torch

from mtkd_adr.models import MODEL_REGISTRY, build_model
from mtkd_adr.training.mtkd import compute_adaptive_weights, distillation_loss, feature_distillation_loss


@pytest.mark.parametrize("arch", list(MODEL_REGISTRY))
def test_forward_shapes(arch):
    m = build_model(arch, 10).eval()
    x = torch.rand(4, 10)
    logit = m(x)
    assert logit.shape == (4, 1)
    logit2, feats = m(x, return_features=True)
    assert torch.allclose(logit, logit2) and feats.shape[0] == 4


def test_unknown_arch():
    with pytest.raises(ValueError):
        build_model("transformer", 10)


def test_distillation_pieces():
    s = torch.randn(8, 1, requires_grad=True)
    ts = [torch.randn(8, 1), torch.randn(8, 1)]
    w = compute_adaptive_weights(s.detach(), ts)
    assert abs(sum(w) - 1.0) < 1e-5 and all(x > 0 for x in w)
    kd = distillation_loss(s, ts, w, 4.0)
    assert torch.isfinite(kd) and kd.requires_grad
    sf = torch.randn(8, 5, requires_grad=True)
    fl = feature_distillation_loss(sf, [torch.randn(8, 5), torch.randn(8, 5)], w)
    assert torch.isfinite(fl)


def test_collapse_detection_flags_constant_predictor():
    import os
    from torch.utils.data import DataLoader, TensorDataset
    from mtkd_adr.config import load_config
    from mtkd_adr.evaluate import check_collapse
    from conftest import CONFIGS

    class AlwaysBenign(torch.nn.Module):
        def forward(self, x, return_features=False):
            return torch.full((x.shape[0], 1), 10.0)

    x = torch.rand(40, 5)
    y = torch.tensor([0, 1] * 20)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    cfg = load_config(os.path.join(CONFIGS, "cicids2017.yaml"))
    collapsed, m = check_collapse(AlwaysBenign(), loader, "const", cfg, torch.device("cpu"))
    assert collapsed and m["attack_recall"] == 0.0
