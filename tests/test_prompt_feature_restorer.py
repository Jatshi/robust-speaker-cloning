import torch

from src.models.prompt_feature_restorer import PromptFeatureRestorer
from src.prompt_feature_loss import restoration_loss


def test_prompt_feature_restorer_starts_as_identity() -> None:
    model = PromptFeatureRestorer(torch.randn(80), torch.rand(80) + 0.1, channels=32, blocks=2, dropout=0).eval()
    feature, quality = torch.randn(3, 80, 41), torch.rand(3, 2)
    output, diagnostics = model(feature, quality)
    torch.testing.assert_close(output, feature, rtol=0, atol=2e-6)
    assert torch.count_nonzero(diagnostics["residual_z"]) == 0


def test_alpha_zero_remains_noop_after_weight_change() -> None:
    model = PromptFeatureRestorer(torch.zeros(80), torch.ones(80), channels=32, blocks=2, dropout=0).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.output.weight)
    feature, quality = torch.randn(2, 80, 31), torch.rand(2, 2)
    output, _ = model(feature, quality, alpha=0)
    torch.testing.assert_close(output, feature, rtol=0, atol=2e-6)


def test_prompt_feature_loss_backpropagates() -> None:
    model = PromptFeatureRestorer(torch.zeros(80), torch.ones(80), channels=32, blocks=2, dropout=0).train()
    degraded, clean, quality = torch.randn(4, 80, 37), torch.randn(4, 80, 37), torch.rand(4, 2)
    predicted, diagnostics = model(degraded, quality)
    clean_output, _ = model(clean, torch.ones_like(quality))
    losses = restoration_loss(predicted, clean, model.standardise(clean), clean_output, clean, diagnostics)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.output.weight.grad is not None
