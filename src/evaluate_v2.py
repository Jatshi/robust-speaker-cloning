"""V2 的真实 CosyVoice 端到端评测：baseline 与鲁棒条件逐样本对比。"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import soundfile
import torch
import torchaudio

try:  # SpeechBrain 1.x
    from speechbrain.inference.speaker import EncoderClassifier
except ModuleNotFoundError:  # SpeechBrain 0.5.x available in the training image
    from speechbrain.pretrained import EncoderClassifier

from src.cosyvoice_infer import CosyVoiceConditionedSynthesizer
from src.degradation import DEGRADATION_TYPES, simulate
from src.models.robust_speaker_encoder import LightweightBWENet, RobustSpeakerEncoder


def _mel(waveform: torch.Tensor) -> torch.Tensor:
    transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, win_length=512, hop_length=256, n_mels=80)
    return torch.log(transform(waveform).clamp_min(1e-6))


def _xvector(model: EncoderClassifier, path: Path, device: torch.device) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path); waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000: waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return torch.nn.functional.normalize(model.encode_batch(waveform.to(device))[0, 0], dim=0).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True); parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--samples-per-type", type=int, default=3)
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True); encoder = RobustSpeakerEncoder().to(device).eval(); bwe = LightweightBWENet().to(device).eval()
    encoder.load_state_dict(checkpoint["encoder"]); bwe.load_state_dict(checkpoint["bwe"])
    speakers: dict[str, list[dict]] = json.loads(args.manifest.read_text(encoding="utf-8")); samples = [item for items in speakers.values() for item in items][:len(DEGRADATION_TYPES) * args.samples_per_type]
    synthesizer = CosyVoiceConditionedSynthesizer(args.model_root, args.cosyvoice_root)
    # SpeechBrain 0.5.x keeps a separate internal ``device`` field; setting
    # only ``.to(device)`` leaves encode_batch inputs on CPU.
    xvector = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", savedir=str(args.output / "ecapa"),
        run_opts={"device": str(device)},
    ).eval()
    generated = args.output / "generated"; generated.mkdir(parents=True, exist_ok=True); rows = []
    for index, item in enumerate(samples):
        clean, sample_rate = torchaudio.load(item["wav_path"]); clean = clean.mean(dim=0)
        if sample_rate != 16000: clean = torchaudio.functional.resample(clean, sample_rate, 16000)
        kind = DEGRADATION_TYPES[index % len(DEGRADATION_TYPES)]; degraded, quality_info = simulate(clean, kind, index % 3, 10_000 + index)
        quality = torch.tensor([[float(quality_info["snr_db"]) / 30, float(quality_info["bandwidth_hz"]) / 8000]], device=device)
        with torch.no_grad():
            degraded_mel = _mel(degraded)[None].to(device); enhanced = bwe(degraded_mel) if quality_info["telephone"] else degraded_mel
            condition = encoder(enhanced, quality)[0].cpu()
        prompt_file = Path(tempfile.mkstemp(suffix=".wav")[1]); soundfile.write(prompt_file, degraded.numpy(), 16000)
        baseline, robust = generated / f"{item['audio_id']}_{kind}_baseline.wav", generated / f"{item['audio_id']}_{kind}_robust.wav"
        try:
            synthesizer.synthesize_baseline(item["text"], item["text"], prompt_file, baseline)
            synthesizer.synthesize_with_condition(item["text"], item["text"], prompt_file, condition, robust)
            clean_file = Path(tempfile.mkstemp(suffix=".wav")[1]); soundfile.write(clean_file, clean.numpy(), 16000)
            try:
                target = _xvector(xvector, clean_file, device); base = _xvector(xvector, baseline, device); result = _xvector(xvector, robust, device)
            finally: clean_file.unlink(missing_ok=True)
            rows.append({"audio_id": item["audio_id"], "degradation": kind, "baseline_xvector_similarity": float(target @ base), "robust_xvector_similarity": float(target @ result), "delta": float(target @ result - target @ base)})
        finally: prompt_file.unlink(missing_ok=True)
        print(f"evaluated {index + 1}/{len(samples)} {kind}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(args.output / "evaluation.csv", index=False)
    summary = {"samples": len(frame), "baseline": float(frame.baseline_xvector_similarity.mean()), "robust": float(frame.robust_xvector_similarity.mean()), "delta": float(frame.delta.mean()), "by_degradation": frame.groupby("degradation").mean(numeric_only=True).reset_index().to_dict(orient="records")}
    (args.output / "evaluation.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
