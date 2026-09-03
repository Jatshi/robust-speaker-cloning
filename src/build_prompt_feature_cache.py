"""Cache paired features from CosyVoice's exact 24 kHz prompt frontend."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.degradation_v3 import DegradationAssets, FORMAL_DEGRADATION_TYPES, FormalDegradationEngine
from src.quality import estimate_quality

SAMPLES_16K = 64000
# Matcha's implementation reflect-pads by (n_fft-hop)/2 before STFT, yielding
# exactly 200 frames for a 4-second 24 kHz waveform despite center=False.
FRAMES = 200
MELS = 80
_ENGINE = None
_MEL_SPECTROGRAM = None


def _initialise_worker(cosyvoice_root: str, asset_manifest: str) -> None:
    global _ENGINE, _MEL_SPECTROGRAM
    torch.set_num_threads(1)
    root = Path(cosyvoice_root)
    for directory in (root, root / "third_party" / "Matcha-TTS"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from matcha.utils.audio import mel_spectrogram  # pylint: disable=import-outside-toplevel

    _MEL_SPECTROGRAM = mel_spectrogram
    _ENGINE = FormalDegradationEngine(
        DegradationAssets.from_manifest(Path(asset_manifest), "train"), allow_synthetic_fallback=False
    )


def _crop(path: str, seed: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.numel() > SAMPLES_16K:
        start = seed % (waveform.numel() - SAMPLES_16K + 1)
        waveform = waveform[start : start + SAMPLES_16K]
    return torch.nn.functional.pad(waveform, (0, max(0, SAMPLES_16K - waveform.numel())))


def _feature(waveform_16k: torch.Tensor) -> np.ndarray:
    waveform_24k = torchaudio.functional.resample(waveform_16k, 16000, 24000).reshape(1, -1)
    feature = _MEL_SPECTROGRAM(  # type: ignore[misc]
        waveform_24k,
        n_fft=1920,
        num_mels=MELS,
        sampling_rate=24000,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=8000,
        center=False,
    ).squeeze(0)
    if feature.shape != (MELS, FRAMES):
        raise RuntimeError(f"unexpected CosyVoice prompt feature shape {tuple(feature.shape)}")
    return feature.numpy().astype(np.float16, copy=False)


def _process(item: tuple[int, str, dict]) -> tuple:
    index, speaker, row = item
    clean_waveform = _crop(row["wav_path"], 42 + index)
    clean_feature = _feature(clean_waveform)
    degraded_features, qualities, audit = [], [], []
    for kind_index, kind in enumerate(FORMAL_DEGRADATION_TYPES):
        severity = (index + kind_index) % 3
        seed = 700_000 + index * 17 + kind_index
        degraded, metadata = _ENGINE.apply(clean_waveform, kind, severity, seed)  # type: ignore[union-attr]
        degraded_features.append(_feature(degraded))
        qualities.append(estimate_quality(degraded)[0].numpy())
        audit.append(
            {
                "source_index": index,
                "speaker": speaker,
                "wav_path": row["wav_path"],
                "kind": kind,
                "severity": severity,
                "seed": seed,
                "degradation": metadata,
            }
        )
    return index, speaker, row["wav_path"], clean_feature, degraded_features, np.asarray(qualities), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--degradation-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()
    manifest: dict[str, list[dict]] = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = [(index, speaker, row) for index, (speaker, row) in enumerate((pair for speaker, rows in manifest.items() for pair in ((speaker, row) for row in rows)))]
    if args.max_items:
        items = items[: args.max_items]
    count, variants = len(items), len(FORMAL_DEGRADATION_TYPES)
    args.output.mkdir(parents=True, exist_ok=True)
    clean = np.memmap(args.output / "clean_features.f16", np.float16, "w+", shape=(count, MELS, FRAMES))
    degraded = np.memmap(args.output / "degraded_features.f16", np.float16, "w+", shape=(count * variants, MELS, FRAMES))
    quality = np.memmap(args.output / "quality.f32", np.float32, "w+", shape=(count * variants, 2))
    speakers: list[str | None] = [None] * count
    paths: list[str | None] = [None] * count
    audit_rows: list[dict | None] = [None] * (count * variants)
    with multiprocessing.Pool(
        args.workers,
        initializer=_initialise_worker,
        initargs=(str(args.cosyvoice_root.resolve()), str(args.degradation_assets.resolve())),
    ) as pool:
        for completed, result in enumerate(pool.imap_unordered(_process, items), 1):
            index, speaker, path, clean_feature, degraded_features, qualities, audit = result
            start = index * variants
            clean[index] = clean_feature
            degraded[start : start + variants] = np.stack(degraded_features)
            quality[start : start + variants] = qualities
            speakers[index], paths[index] = speaker, path
            audit_rows[start : start + variants] = audit
            if completed % 25 == 0 or completed == count:
                print(json.dumps({"event": "prompt_feature_cache", "completed": completed, "total": count}), flush=True)
    for array in (clean, degraded, quality):
        array.flush()
    metadata = {
        "schema_version": 4,
        "feature_space": "cosyvoice2_matcha_mel_24khz_nfft1920_hop480_fmax8000_center_false",
        "count": count,
        "variants": variants,
        "mels": MELS,
        "frames": FRAMES,
        "speakers": speakers,
        "paths": paths,
        "degradation_types": list(FORMAL_DEGRADATION_TYPES),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "audit.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in audit_rows) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
