"""将 V2 的 42k 在线退化预计算为紧凑 float16 mel 缓存，而非重复 WAV。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as multiprocessing
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.degradation import DEGRADATION_TYPES, simulate

FRAMES, MELS, SAMPLES = 251, 80, 64000
_MEL = None


def _initialise_worker() -> None:
    global _MEL
    torch.set_num_threads(1); _MEL = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, win_length=512, hop_length=256, n_mels=MELS)


def _crop(path: str, seed: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path); waveform = waveform.mean(dim=0)
    if sample_rate != 16000: waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.numel() > SAMPLES: waveform = waveform[seed % (waveform.numel() - SAMPLES + 1):][:SAMPLES]
    return torch.nn.functional.pad(waveform, (0, max(0, SAMPLES - waveform.numel())))


def _as_mel(waveform: torch.Tensor) -> np.ndarray:
    mel = torch.log(_MEL(waveform).clamp_min(1e-6))  # type: ignore[misc]
    return mel[:, :FRAMES].numpy().astype(np.float16, copy=False)


def _process(item: tuple[int, str, str]) -> tuple[int, str, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    index, speaker, path = item; clean = _crop(path, index + 42); degraded_mels, qualities, telephone = [], [], []
    for kind_index, kind in enumerate(DEGRADATION_TYPES):
        degraded, metadata = simulate(clean, kind, (index + kind_index) % 3, 100_000 + index * 17 + kind_index)
        degraded_mels.append(_as_mel(degraded)); qualities.append([float(metadata["snr_db"]) / 30, float(metadata["bandwidth_hz"]) / 8000]); telephone.append(bool(metadata["telephone"]))
    return index, speaker, _as_mel(clean), degraded_mels, np.asarray(qualities, np.float32), np.asarray(telephone, bool)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); speakers: dict[str, list[dict]] = json.loads(args.manifest.read_text(encoding="utf-8")); items = [(index, speaker, row["wav_path"]) for index, (speaker, row) in enumerate((entry for speaker, rows in speakers.items() for entry in ((speaker, row) for row in rows)))]
    args.output.mkdir(parents=True, exist_ok=True); count, variants = len(items), len(DEGRADATION_TYPES)
    clean = np.memmap(args.output / "clean_mels.f16", dtype=np.float16, mode="w+", shape=(count, MELS, FRAMES))
    degraded = np.memmap(args.output / "degraded_mels.f16", dtype=np.float16, mode="w+", shape=(count * variants, MELS, FRAMES))
    quality = np.memmap(args.output / "quality.f32", dtype=np.float32, mode="w+", shape=(count * variants, 2)); telephone = np.memmap(args.output / "telephone.bool", dtype=np.bool_, mode="w+", shape=(count * variants,))
    speaker_rows = [None] * count
    with multiprocessing.Pool(args.workers, initializer=_initialise_worker) as pool:
        for completed, result in enumerate(pool.imap_unordered(_process, items), 1):
            index, speaker, clean_mel, variants_mel, qualities, flags = result; clean[index] = clean_mel; start = index * variants; degraded[start:start + variants] = np.stack(variants_mel); quality[start:start + variants] = qualities; telephone[start:start + variants] = flags; speaker_rows[index] = speaker
            if completed % 100 == 0 or completed == count: print(f"cached {completed}/{count} clean utterances ({completed * variants}/{count * variants} degradations)", flush=True)
    for array in (clean, degraded, quality, telephone): array.flush()
    (args.output / "metadata.json").write_text(json.dumps({"count": count, "speakers": speaker_rows, "variants": variants, "frames": FRAMES, "mels": MELS}), encoding="utf-8")


if __name__ == "__main__": main()
