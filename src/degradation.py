"""V2 的七类在线、可复现实音频退化；不在磁盘上复制 WAV。"""
from __future__ import annotations

import math
import random

import torch
import torchaudio

DEGRADATION_TYPES = ("white_noise", "pink_noise", "hvac_noise", "cafe_noise", "reverb_noise", "telephone", "compression")


def _normalise(waveform: torch.Tensor) -> torch.Tensor:
    return waveform / waveform.abs().amax().clamp_min(1e-4) * 0.95


def _noise_like(waveform: torch.Tensor, kind: str, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    if kind == "pink_noise":
        spectrum = torch.fft.rfft(noise)
        frequencies = torch.arange(1, spectrum.shape[-1] + 1, dtype=waveform.dtype)
        noise = torch.fft.irfft(spectrum / frequencies.sqrt(), n=waveform.shape[-1])
    elif kind == "hvac_noise":
        time = torch.arange(waveform.numel(), dtype=waveform.dtype) / 16000
        noise = sum(torch.sin(2 * math.pi * frequency * time + random.random()) for frequency in (180, 250, 420))
        noise = noise + 0.08 * torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    elif kind == "cafe_noise":
        noise = torchaudio.functional.lowpass_biquad(noise, 16000, 4000)
    return noise / noise.pow(2).mean().sqrt().clamp_min(1e-7)


def _with_snr(waveform: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    scale = waveform.pow(2).mean().sqrt() / (10 ** (snr_db / 20))
    return waveform + noise * scale


def simulate(waveform: torch.Tensor, kind: str, severity: int, seed: int, sample_rate: int = 16000) -> tuple[torch.Tensor, dict[str, float | str | bool]]:
    """返回退化波形及真实传入编码器的质量元数据。"""
    generator = torch.Generator().manual_seed(seed)
    severity = max(0, min(2, severity)); snr = (15.0, 10.0, 5.0)[severity]
    metadata: dict[str, float | str | bool] = {"type": kind, "snr_db": snr, "bandwidth_hz": 8000.0, "telephone": False}
    if kind in {"white_noise", "pink_noise", "hvac_noise", "cafe_noise"}:
        output = _with_snr(waveform, _noise_like(waveform, kind, generator), snr)
    elif kind == "reverb_noise":
        rt60 = (0.3, 0.5, 0.8)[severity]
        impulse_length = int(sample_rate * rt60)
        impulse = torch.exp(-torch.arange(impulse_length, dtype=waveform.dtype) / max(1, impulse_length / 6))
        impulse[0] = 1.0
        reverberant = torch.nn.functional.conv1d(waveform[None, None], impulse[None, None], padding=impulse_length - 1)[0, 0][: waveform.numel()]
        output = _with_snr(0.7 * waveform + 0.3 * reverberant, _noise_like(waveform, "white_noise", generator), snr)
        metadata["rt60_s"] = rt60
    elif kind == "telephone":
        narrow = torchaudio.functional.resample(waveform, sample_rate, 8000)
        output = torchaudio.functional.resample(narrow, 8000, sample_rate)
        output = torchaudio.functional.highpass_biquad(output, sample_rate, 300)
        output = _with_snr(output, _noise_like(output, "hvac_noise", generator), snr)
        metadata.update({"bandwidth_hz": 3400.0, "telephone": True})
    elif kind == "compression":
        cutoff = (6500, 5500, 4200)[severity]
        output = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff)
        levels = float((256, 128, 64)[severity])
        output = torch.round(output * levels) / levels
        metadata["bandwidth_hz"] = float(cutoff)
    else:
        raise ValueError(f"unknown degradation: {kind}")
    return _normalise(output.clamp(-1, 1)), metadata
