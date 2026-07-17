import torch

from src.degradation import DEGRADATION_TYPES, simulate
from src.models.losses import combined_loss
from src.models.robust_speaker_encoder import LightweightBWENet, RobustSpeakerEncoder


def test_v2_models_preserve_required_shapes():
    encoder = RobustSpeakerEncoder(); bwe = LightweightBWENet()
    mel, quality = torch.randn(2, 80, 64), torch.ones(2, 2)
    assert encoder(mel, quality).shape == (2, 192)
    assert bwe(mel).shape == mel.shape


def test_joint_loss_is_finite_and_uses_bwe_branch():
    anchor = torch.nn.functional.normalize(torch.randn(2, 192), dim=-1)
    losses = combined_loss(anchor, anchor, anchor, torch.arange(2), torch.tensor([True, False]), torch.randn(2, 80, 64), torch.randn(2, 80, 64))
    assert torch.isfinite(losses["total"])
    assert losses["bwe_l1"] > 0


def test_all_seven_degradations_return_quality_metadata():
    waveform = torch.randn(16000).clamp(-1, 1)
    for kind in DEGRADATION_TYPES:
        degraded, metadata = simulate(waveform, kind, 1, 42)
        assert degraded.shape == waveform.shape
        assert "snr_db" in metadata and "bandwidth_hz" in metadata
