from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.degradation_v3 import DegradationAssets, DegradationPrerequisiteError, FormalDegradationEngine


def test_formal_engine_refuses_missing_real_assets():
    engine = FormalDegradationEngine(DegradationAssets(), allow_synthetic_fallback=False)
    with pytest.raises(DegradationPrerequisiteError):
        engine.apply(torch.zeros(16000), "real_noise", 1, 42)


def test_real_noise_and_rir_assets_are_used(tmp_path: Path):
    noise_path, rir_path = tmp_path / "noise.wav", tmp_path / "rir.wav"
    sf.write(noise_path, np.random.default_rng(1).normal(size=16000).astype("float32"), 16000)
    rir = np.zeros(1000, dtype="float32"); rir[0], rir[200] = 1.0, 0.3
    sf.write(rir_path, rir, 16000)
    engine = FormalDegradationEngine(DegradationAssets(noise=(noise_path,), speech=(noise_path,), music=(noise_path,), rir=(rir_path,)))
    waveform = torch.sin(2 * torch.pi * 220 * torch.arange(16000) / 16000)
    noisy, noise_metadata = engine.apply(waveform, "real_noise", 1, 7)
    reverberant, rir_metadata = engine.apply(waveform, "reverb_noise", 1, 7)
    assert noisy.shape == waveform.shape == reverberant.shape
    assert noise_metadata["source"] == "real_asset" and rir_metadata["source"] == "real_asset"
