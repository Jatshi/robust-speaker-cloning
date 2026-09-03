"""Build a compact cache in the exact raw CAMPPlus space used by CosyVoice."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.degradation_v3 import DegradationAssets, FORMAL_DEGRADATION_TYPES, FormalDegradationEngine
from src.quality import estimate_quality

SAMPLES = 64000
DIMENSION = 192


def _add_cosyvoice_path(root: Path) -> None:
    for directory in (root, root / "third_party" / "Matcha-TTS"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def _crop(path: Path, seed: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.numel() > SAMPLES:
        start = seed % (waveform.numel() - SAMPLES + 1)
        waveform = waveform[start : start + SAMPLES]
    return torch.nn.functional.pad(waveform, (0, max(0, SAMPLES - waveform.numel())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--degradation-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-split", choices=("train", "selection", "test"), default="train")
    parser.add_argument("--max-items", type=int, default=0, help="bounded smoke only")
    args = parser.parse_args()

    _add_cosyvoice_path(args.cosyvoice_root)
    from cosyvoice.utils.onnx import EmbeddingExtractor  # pylint: disable=import-outside-toplevel

    manifest: dict[str, list[dict]] = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = [(speaker, row) for speaker, rows in manifest.items() for row in rows]
    if args.max_items:
        items = items[: args.max_items]
    variants = len(FORMAL_DEGRADATION_TYPES)
    args.output.mkdir(parents=True, exist_ok=True)
    clean = np.memmap(args.output / "clean_embeddings.f32", np.float32, "w+", shape=(len(items), DIMENSION))
    degraded = np.memmap(
        args.output / "degraded_embeddings.f32", np.float32, "w+", shape=(len(items) * variants, DIMENSION)
    )
    quality = np.memmap(args.output / "quality.f32", np.float32, "w+", shape=(len(items) * variants, 2))
    extractor = EmbeddingExtractor(str(args.model_root / "campplus.onnx"))
    engine = FormalDegradationEngine(
        DegradationAssets.from_manifest(args.degradation_assets, args.asset_split), allow_synthetic_fallback=False
    )
    audit: list[dict] = []
    speakers: list[str] = []
    paths: list[str] = []
    audio_ids: list[str] = []

    def embed(waveform: torch.Tensor) -> np.ndarray:
        value = extractor.inference(waveform.reshape(1, -1).cpu()).float().reshape(-1)
        if value.numel() != DIMENSION:
            raise RuntimeError(f"unexpected CAMPPlus dimension: {value.numel()}")
        return value.numpy().astype(np.float32, copy=False)

    for index, (speaker, row) in enumerate(items):
        clean_waveform = _crop(Path(row["wav_path"]), 42 + index)
        clean[index] = embed(clean_waveform)
        speakers.append(speaker)
        paths.append(row["wav_path"])
        audio_ids.append(row.get("audio_id", Path(row["wav_path"]).stem))
        for kind_index, kind in enumerate(FORMAL_DEGRADATION_TYPES):
            severity = (index + kind_index) % 3
            seed = 400_000 + index * 17 + kind_index
            degraded_waveform, degradation_metadata = engine.apply(clean_waveform, kind, severity, seed)
            cache_index = index * variants + kind_index
            degraded[cache_index] = embed(degraded_waveform)
            quality[cache_index] = estimate_quality(degraded_waveform)[0].numpy()
            audit.append(
                {
                    "cache_index": cache_index,
                    "source_index": index,
                    "speaker": speaker,
                    "wav_path": row["wav_path"],
                    "kind": kind,
                    "severity": severity,
                    "seed": seed,
                    "degradation": degradation_metadata,
                }
            )
        if (index + 1) % 50 == 0 or index + 1 == len(items):
            print(json.dumps({"event": "campplus_cache", "completed": index + 1, "total": len(items)}), flush=True)

    for array in (clean, degraded, quality):
        array.flush()
    metadata = {
        "schema_version": 4,
        "embedding_space": "raw_cosyvoice_campplus_no_l2_normalization",
        "count": len(items),
        "variants": variants,
        "dimension": DIMENSION,
        "degradation_types": list(FORMAL_DEGRADATION_TYPES),
        "speakers": speakers,
        "paths": paths,
        "audio_ids": audio_ids,
        "asset_split": args.asset_split,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "audit.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in audit) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
