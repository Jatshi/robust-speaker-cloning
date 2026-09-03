"""Bounded real-model smoke test for the CosyVoice conditioning contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile
import torch
import torchaudio

from src.cosyvoice_infer import CosyVoiceConditionedSynthesizer


def _add_cosyvoice_path(cosyvoice_root: Path) -> None:
    for directory in (cosyvoice_root, cosyvoice_root / "third_party" / "Matcha-TTS"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def _prompt_condition(prompt_wav: Path, model_root: Path, cosyvoice_root: Path) -> torch.Tensor:
    _add_cosyvoice_path(cosyvoice_root)
    from cosyvoice.utils.onnx import EmbeddingExtractor  # pylint: disable=import-outside-toplevel

    waveform, sample_rate = torchaudio.load(prompt_wav)
    waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    condition = EmbeddingExtractor(str(model_root / "campplus.onnx")).inference(waveform[None]).float()
    if condition.numel() != 192 or not torch.isfinite(condition).all():
        raise RuntimeError(f"invalid CampPlus condition: shape={tuple(condition.shape)}")
    return condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    condition = _prompt_condition(args.prompt_wav, args.model_root, args.cosyvoice_root)
    synthesizer = CosyVoiceConditionedSynthesizer(args.model_root, args.cosyvoice_root)
    baseline = args.output / "baseline.wav"
    conditioned = args.output / "conditioned.wav"
    synthesizer.synthesize_baseline(args.text, args.prompt_text, args.prompt_wav, baseline)
    synthesizer.synthesize_with_condition(
        args.text, args.prompt_text, args.prompt_wav, condition, conditioned
    )

    report = {"condition_shape": list(condition.shape), "outputs": {}}
    for name, path in (("baseline", baseline), ("conditioned", conditioned)):
        info = soundfile.info(path)
        if info.frames <= 0 or info.samplerate != synthesizer.model.sample_rate:
            raise RuntimeError(f"invalid synthesized audio: {path}: {info}")
        report["outputs"][name] = {
            "path": str(path),
            "sample_rate": info.samplerate,
            "frames": info.frames,
            "seconds": info.duration,
        }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
