"""Shared quality estimation used by both evaluation and deployment."""
from __future__ import annotations

import torch


def high_band_energy_ratio(waveform: torch.Tensor, sample_rate: int = 16000, cutoff_hz: float = 4200.0) -> float:
    """Observable narrow-band cue used by the safe prompt-feature router."""
    mono = waveform.float().reshape(-1)
    window = torch.hann_window(1024, device=mono.device)
    spectrum = torch.stft(mono, n_fft=1024, hop_length=256, window=window, return_complex=True).abs().square().mean(dim=-1)
    frequencies = torch.linspace(0.0, sample_rate / 2, spectrum.numel(), device=mono.device)
    total = spectrum[(frequencies >= 300.0) & (frequencies <= 7800.0)].sum().clamp_min(1e-12)
    return float(spectrum[frequencies >= cutoff_hz].sum() / total)


def estimate_quality(waveform: torch.Tensor, sample_rate: int = 16000) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate SNR and occupied bandwidth without access to clean speech.

    SNR is estimated from the 20th-percentile short-time frame energy; this is
    intentionally a small deterministic estimator, not an oracle.  Formal
    reports must label the returned values as estimates.
    """
    mono = waveform.float().reshape(-1)
    if mono.numel() < 512:
        mono = torch.nn.functional.pad(mono, (0, 512 - mono.numel()))
    frames = mono.unfold(0, 512, 256)
    frame_power = frames.square().mean(dim=-1).clamp_min(1e-10)
    noise_power = torch.quantile(frame_power, 0.2)
    signal_power = torch.quantile(frame_power, 0.8)
    snr_db = 10.0 * torch.log10((signal_power - noise_power).clamp_min(1e-10) / noise_power)

    window = torch.hann_window(min(mono.numel(), 4096), device=mono.device)
    spectrum = torch.fft.rfft(mono[: window.numel()] * window).abs().square()
    cumulative = spectrum.cumsum(0) / spectrum.sum().clamp_min(1e-10)
    occupied_bin = torch.searchsorted(cumulative, torch.tensor(0.99, device=mono.device)).clamp_max(spectrum.numel() - 1)
    bandwidth_hz = occupied_bin.float() * sample_rate / (2.0 * max(1, spectrum.numel() - 1))
    normalized = torch.stack([(snr_db / 30.0).clamp(0.0, 1.0), (bandwidth_hz / 8000.0).clamp(0.0, 1.0)])
    return normalized, {"estimated_snr_db": float(snr_db), "estimated_bandwidth_hz": float(bandwidth_hz)}


def perturb_quality(quality: torch.Tensor, std: tuple[float, float], training: bool) -> torch.Tensor:
    """Expose the encoder to estimator error instead of only oracle metadata."""
    if not training or max(std) <= 0:
        return quality
    scale = quality.new_tensor(std)
    return (quality + torch.randn_like(quality) * scale).clamp(0.0, 1.0)
