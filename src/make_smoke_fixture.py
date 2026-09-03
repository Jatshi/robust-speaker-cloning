"""Create a tiny deterministic cache for CPU-only training pipeline checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); cache = args.output / "feature_cache"; cache.mkdir(parents=True, exist_ok=True)
    speakers, variants, frames, mels = 12, 7, 32, 80
    generator = np.random.default_rng(42)
    clean = np.memmap(cache / "clean_mels.f16", np.float16, "w+", shape=(speakers, mels, frames))
    degraded = np.memmap(cache / "degraded_mels.f16", np.float16, "w+", shape=(speakers * variants, mels, frames))
    quality = np.memmap(cache / "quality.f32", np.float32, "w+", shape=(speakers * variants, 2))
    telephone = np.memmap(cache / "telephone.bool", np.bool_, "w+", shape=(speakers * variants,))
    clean[:] = generator.normal(size=clean.shape); degraded[:] = np.repeat(clean, variants, axis=0) + generator.normal(0, 0.1, size=degraded.shape)
    quality[:] = generator.uniform(0.2, 1.0, size=quality.shape); telephone[:] = np.tile([False] * 5 + [True, False], speakers)
    for array in (clean, degraded, quality, telephone): array.flush()
    names = [f"S{index:03d}" for index in range(speakers)]
    (cache / "metadata.json").write_text(json.dumps({"schema_version": 3, "count": speakers, "speakers": names, "variants": variants, "frames": frames, "mels": mels, "degradation_backend": "synthetic_smoke_only"}), encoding="utf-8")
    manifest = {speaker: [{"audio_id": speaker, "wav_path": f"/{speaker}.wav", "text": "smoke"}] for speaker in names}
    manifest_path = args.output / "manifest.json"; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    torch.save({f"/{speaker}.wav": torch.nn.functional.normalize(torch.randn(192), dim=0) for speaker in names}, args.output / "teacher.pt")
    print(json.dumps({"manifest": str(manifest_path), "feature_cache": str(cache), "teacher_cache": str(args.output / 'teacher.pt')}))


if __name__ == "__main__": main()
