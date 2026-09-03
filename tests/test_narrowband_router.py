import torch

from src.quality import high_band_energy_ratio


def test_high_band_ratio_separates_lowpass_signal() -> None:
    sample_rate = 16000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    wide = torch.sin(2 * torch.pi * 1000 * time) + 0.2 * torch.sin(2 * torch.pi * 6000 * time)
    narrow = torch.sin(2 * torch.pi * 1000 * time)
    assert high_band_energy_ratio(narrow) < 1e-4
    assert high_band_energy_ratio(wide) > 1e-3
