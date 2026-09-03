from __future__ import annotations

import random

import numpy as np
import torch

from src.cosyvoice_v4 import seed_everything
from src.losses_v4 import campplus_calibration_loss
from src.models.campplus_residual_adapter import CampPlusResidualAdapter


def _model() -> CampPlusResidualAdapter:
    return CampPlusResidualAdapter(torch.randn(192), torch.rand(192) + 0.1, hidden_dim=32, dropout=0.0)


def test_adapter_is_exact_identity_at_initialisation() -> None:
    model = _model().eval()
    embedding, quality = torch.randn(4, 192), torch.rand(4, 2)
    output, diagnostics = model(embedding, quality)
    torch.testing.assert_close(output, embedding, rtol=0.0, atol=2e-6)
    assert torch.count_nonzero(diagnostics["residual_z"]) == 0


def test_alpha_zero_is_exact_fallback_after_training_changes() -> None:
    model = _model().eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.residual[-1].weight)
        torch.nn.init.normal_(model.residual[-1].bias)
    embedding, quality = torch.randn(4, 192), torch.rand(4, 2)
    output, _ = model(embedding, quality, alpha=0.0)
    torch.testing.assert_close(output, embedding, rtol=0.0, atol=2e-6)


def test_v4_loss_is_finite_and_backpropagates() -> None:
    model = _model().train()
    degraded, clean, quality = torch.randn(8, 192), torch.randn(8, 192), torch.rand(8, 2)
    predicted, diagnostics = model(degraded, quality)
    clean_output, _ = model(clean, torch.ones_like(quality))
    losses = campplus_calibration_loss(
        predicted, clean, clean, clean_output, clean, model.standardise(clean), diagnostics
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.residual[-1].weight.grad is not None


def test_seed_everything_resets_all_local_rngs() -> None:
    seed_everything(2026)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(2026)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])
