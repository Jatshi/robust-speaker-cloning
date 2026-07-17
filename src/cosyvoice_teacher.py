"""使用 CosyVoice 实际 CampPlus 前端建立干净条件蒸馏缓存。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio


def _add_cosyvoice_path(cosyvoice_root: Path) -> None:
    for directory in (cosyvoice_root, cosyvoice_root / "third_party" / "Matcha-TTS"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def extract_embeddings(manifest: Path, cosyvoice_root: Path, model_root: Path, output: Path) -> None:
    _add_cosyvoice_path(cosyvoice_root)
    from cosyvoice.utils.onnx import EmbeddingExtractor  # pylint: disable=import-outside-toplevel

    rows = json.loads(manifest.read_text(encoding="utf-8"))
    extractor = EmbeddingExtractor(str(model_root / "campplus.onnx")); results: dict[str, torch.Tensor] = {}
    flat = [item for items in rows.values() for item in items]
    for index, item in enumerate(flat, 1):
        waveform, sample_rate = torchaudio.load(item["wav_path"]); waveform = waveform.mean(dim=0)
        if sample_rate != 16000: waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        results[item["wav_path"]] = torch.nn.functional.normalize(extractor.inference(waveform[None]).float(), dim=0)
        if index % 100 == 0 or index == len(flat): print(f"teacher {index}/{len(flat)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True); torch.save(results, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); extract_embeddings(args.manifest, args.cosyvoice_root, args.model_root, args.output)
