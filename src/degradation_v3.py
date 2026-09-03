"""Real-asset and real-codec degradations for formal V3 experiments."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from src.degradation import _normalise, _with_snr, simulate

FORMAL_DEGRADATION_TYPES = ("real_noise", "babble_noise", "music_noise", "reverb_noise", "telephone", "mp3", "opus")


class DegradationPrerequisiteError(RuntimeError):
    """Raised when a formal run would silently fall back to synthetic data."""


@dataclass(frozen=True)
class DegradationAssets:
    noise: tuple[Path, ...] = ()
    speech: tuple[Path, ...] = ()
    music: tuple[Path, ...] = ()
    rir: tuple[Path, ...] = ()

    @classmethod
    def from_manifest(cls, path: Path | None, partition: str | None = None) -> "DegradationAssets":
        if path is None:
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        def selected(category: str) -> tuple[Path, ...]:
            return tuple(Path(row["path"]) for row in data.get(category, []) if partition is None or row.get("split") == partition)
        return cls(noise=selected("noise"), speech=selected("speech"), music=selected("music"), rir=selected("rir"))


def _load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    return torchaudio.functional.resample(waveform, source_rate, sample_rate) if source_rate != sample_rate else waveform


def _fit_length(value: torch.Tensor, samples: int, generator: random.Random) -> torch.Tensor:
    if value.numel() < samples:
        repeats = (samples + value.numel() - 1) // max(1, value.numel())
        value = value.repeat(repeats)
    start = generator.randrange(max(1, value.numel() - samples + 1))
    return value[start : start + samples]


def _fft_convolve(signal: torch.Tensor, impulse: torch.Tensor) -> torch.Tensor:
    length = signal.numel() + impulse.numel() - 1
    fft_length = 1 << (length - 1).bit_length()
    result = torch.fft.irfft(torch.fft.rfft(signal, fft_length) * torch.fft.rfft(impulse, fft_length), fft_length)
    return result[: signal.numel()]


def _ffmpeg_roundtrip(waveform: torch.Tensor, sample_rate: int, codec: str, bitrate: str | None = None) -> torch.Tensor:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise DegradationPrerequisiteError("ffmpeg is required for real codec degradations")
    extension = {"libmp3lame": ".mp3", "aac": ".m4a", "libopus": ".ogg", "pcm_alaw": ".wav", "pcm_mulaw": ".wav"}[codec]
    with tempfile.TemporaryDirectory(prefix="rsc-codec-") as directory:
        root = Path(directory); source, encoded, decoded = root / "source.wav", root / f"encoded{extension}", root / "decoded.wav"
        sf.write(source, waveform.detach().cpu().numpy(), sample_rate)
        command = [executable, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1"]
        if codec in {"pcm_alaw", "pcm_mulaw"}:
            command += ["-ar", "8000"]
        command += ["-c:a", codec]
        if bitrate:
            command += ["-b:a", bitrate]
        command.append(str(encoded))
        subprocess.run(command, check=True, capture_output=True)
        subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", "-i", str(encoded), "-ar", str(sample_rate), "-ac", "1", str(decoded)], check=True, capture_output=True)
        result, decoded_rate = sf.read(decoded, dtype="float32")
        if decoded_rate != sample_rate:
            raise RuntimeError(f"ffmpeg decoded at unexpected rate {decoded_rate}")
        return torch.from_numpy(result).reshape(-1)


class FormalDegradationEngine:
    """Use MUSAN/DNS-like WAV assets, measured RIRs and actual ffmpeg codecs.

    When ``allow_synthetic_fallback`` is false, missing assets stop the run so a
    formal report can never be accidentally produced from toy degradations.
    """

    def __init__(self, assets: DegradationAssets, allow_synthetic_fallback: bool = False, sample_rate: int = 16000) -> None:
        self.assets = assets
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.sample_rate = sample_rate

    def _asset_or_fallback(self, category: str, waveform: torch.Tensor, kind: str, severity: int, seed: int) -> tuple[torch.Tensor, dict]:
        paths = getattr(self.assets, category)
        if not paths:
            if self.allow_synthetic_fallback:
                legacy_kind = {"real_noise": "white_noise", "babble_noise": "cafe_noise", "music_noise": "pink_noise", "mp3": "compression", "opus": "compression"}.get(kind, kind)
                value, metadata = simulate(waveform, legacy_kind, severity, seed, self.sample_rate)
                return value, dict(metadata) | {"source": "synthetic_fallback"}
            raise DegradationPrerequisiteError(f"formal degradation '{kind}' requires at least one {category} asset")
        path = paths[seed % len(paths)]
        return _load_mono(path, self.sample_rate), {"asset_path": str(path), "source": "real_asset"}

    def apply(self, waveform: torch.Tensor, kind: str, severity: int, seed: int) -> tuple[torch.Tensor, dict]:
        severity = min(2, max(0, int(severity))); snr_db = (15.0, 10.0, 5.0)[severity]
        generator = random.Random(seed)
        if kind in {"real_noise", "babble_noise", "music_noise"}:
            category = {"real_noise": "noise", "babble_noise": "speech", "music_noise": "music"}[kind]
            noise, metadata = self._asset_or_fallback(category, waveform, kind, severity, seed)
            if metadata["source"] == "synthetic_fallback":
                return noise, metadata
            noise = _fit_length(noise, waveform.numel(), generator)
            if kind == "babble_noise":
                talkers = [noise]
                for offset in (1, 2):
                    path = self.assets.speech[(seed + offset) % len(self.assets.speech)]
                    talkers.append(_fit_length(_load_mono(path, self.sample_rate), waveform.numel(), random.Random(seed + offset)))
                noise = torch.stack(talkers).sum(dim=0)
                metadata["talkers"] = 3
            noise = noise / noise.square().mean().sqrt().clamp_min(1e-7)
            return _normalise(_with_snr(waveform, noise, snr_db)), metadata | {"type": kind, "snr_db": snr_db, "bandwidth_hz": 8000.0, "telephone": False}
        if kind == "reverb_noise":
            rir, metadata = self._asset_or_fallback("rir", waveform, kind, severity, seed)
            if metadata["source"] == "synthetic_fallback":
                return rir, metadata
            rir = rir[: self.sample_rate * 2]; rir = rir / rir.square().sum().sqrt().clamp_min(1e-7)
            wet = _fft_convolve(waveform, rir)
            ratio = (0.25, 0.5, 0.75)[severity]
            return _normalise((1 - ratio) * waveform + ratio * wet), metadata | {"type": kind, "snr_db": 30.0, "bandwidth_hz": 8000.0, "telephone": False, "wet_ratio": ratio}
        if kind == "telephone":
            codec = ("pcm_alaw", "pcm_mulaw", "pcm_alaw")[severity]
            output = _ffmpeg_roundtrip(waveform, self.sample_rate, codec)
            return _normalise(_fit_length(output, waveform.numel(), generator)), {"type": kind, "source": "ffmpeg", "codec": codec, "snr_db": 30.0, "bandwidth_hz": 3400.0, "telephone": True}
        if kind in {"mp3", "opus"}:
            codec = "libmp3lame" if kind == "mp3" else "libopus"
            bitrate = (("64k", "32k", "16k") if kind == "mp3" else ("32k", "16k", "8k"))[severity]
            output = _ffmpeg_roundtrip(waveform, self.sample_rate, codec, bitrate)
            return _normalise(_fit_length(output, waveform.numel(), generator)), {"type": kind, "source": "ffmpeg", "codec": codec, "bitrate": bitrate, "snr_db": 30.0, "bandwidth_hz": (7000.0, 6500.0, 6000.0)[severity], "telephone": False}
        raise ValueError(f"unknown degradation kind: {kind}")
