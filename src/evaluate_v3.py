"""Leakage-resistant three-way end-to-end evaluation for V3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ModuleNotFoundError:
    from speechbrain.pretrained import EncoderClassifier

from src.baselines import MetricGANPlusEnhancer
from src.cosyvoice_infer import CosyVoiceConditionedSynthesizer
from src.degradation_v3 import DegradationAssets, FormalDegradationEngine
from src.metrics import mel_lsd_db, paired_summary
from src.models.robust_speaker_encoder import LightweightBWENet, RobustSpeakerEncoder, apply_soft_bwe
from src.quality import estimate_quality
from src.utmos import UTMOS22StrongScorer


def _load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    return torchaudio.functional.resample(waveform, source_rate, sample_rate) if source_rate != sample_rate else waveform


def _mel(waveform: torch.Tensor) -> torch.Tensor:
    transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, win_length=512, hop_length=256, n_mels=80)
    return torch.log(transform(waveform).clamp_min(1e-6))


def _xvector(model: EncoderClassifier, path: Path, device: torch.device) -> torch.Tensor:
    waveform = _load_audio(path).reshape(1, -1).to(device)
    return torch.nn.functional.normalize(model.encode_batch(waveform)[0, 0], dim=0).cpu()


def _write(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.detach().cpu().numpy(), 16000)


def _load_protocol(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_csv(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _validate_formal_protocol(protocol: list[dict]) -> None:
    if len(protocol) != 105 or len({row["sample_id"] for row in protocol}) != 105:
        raise ValueError("formal protocol must contain exactly 105 unique samples")
    counts = pd.Series([row["degradation"] for row in protocol]).value_counts().to_dict()
    if len(counts) != 7 or set(counts.values()) != {15}:
        raise ValueError(f"formal protocol must contain 15 samples for each of 7 degradations, got {counts}")
    for row in protocol:
        if row["prompt_audio_id"] == row["reference_audio_id"] or row["prompt_text"] == row["synthesis_text"]:
            raise ValueError(f"formal protocol leakage in {row['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--degradation-assets", type=Path)
    parser.add_argument("--asset-split", choices=("selection", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--skip-denoiser", action="store_true")
    parser.add_argument("--skip-utmos", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.formal and (args.allow_synthetic_fallback or args.skip_denoiser or args.skip_utmos or args.max_samples):
        raise SystemExit("--formal forbids synthetic fallback, skipped baselines/metrics and truncated evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder, bwe = RobustSpeakerEncoder().to(device).eval(), LightweightBWENet().to(device).eval()
    encoder.load_state_dict(checkpoint["encoder"]); bwe.load_state_dict(checkpoint["bwe"])
    synthesizer = CosyVoiceConditionedSynthesizer(args.model_root, args.cosyvoice_root)
    engine = FormalDegradationEngine(DegradationAssets.from_manifest(args.degradation_assets, args.asset_split), args.allow_synthetic_fallback)
    speaker_model = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", savedir=str(args.output / "models" / "ecapa"),
        run_opts={"device": str(device)},
    ).eval()
    enhancer = None if args.skip_denoiser else MetricGANPlusEnhancer(args.output / "models" / "metricgan-plus", device)
    utmos = None if args.skip_utmos else UTMOS22StrongScorer(device)

    protocol = _load_protocol(args.protocol)
    if args.formal:
        _validate_formal_protocol(protocol)
    if args.max_samples:
        protocol = protocol[: args.max_samples]
    result_path = args.output / "evaluation.csv"
    rows = pd.read_csv(result_path).to_dict(orient="records") if args.resume and result_path.exists() else []
    completed = {str(row["sample_id"]) for row in rows}
    audio_root = args.output / "audio"
    for index, item in enumerate(protocol, 1):
        if item["sample_id"] in completed:
            continue
        prompt = _load_audio(Path(item["prompt_wav_path"]))
        degraded, degradation_metadata = engine.apply(prompt, item["degradation"], item["severity"], item["seed"])
        quality, estimated_metadata = estimate_quality(degraded)
        prompt_mel, degraded_mel = _mel(prompt), _mel(degraded)
        frames = min(prompt_mel.shape[-1], degraded_mel.shape[-1])
        with torch.inference_mode():
            predicted = bwe(degraded_mel[:, :frames][None].to(device))
            enhanced, gate = apply_soft_bwe(degraded_mel[:, :frames][None].to(device), predicted, quality[None].to(device))
            condition = encoder(enhanced, quality[None].to(device))[0].cpu()

        sample_root = audio_root / item["sample_id"]
        degraded_path = sample_root / "degraded_prompt.wav"; _write(degraded_path, degraded)
        baseline_path, robust_path = sample_root / "baseline.wav", sample_root / "robust.wav"
        synthesizer.synthesize_baseline(item["synthesis_text"], item["prompt_text"], degraded_path, baseline_path)
        synthesizer.synthesize_with_condition(item["synthesis_text"], item["prompt_text"], degraded_path, condition, robust_path)
        denoised_path = sample_root / "denoised_prompt.wav"
        denoised_output_path = sample_root / "denoiser_baseline.wav"
        if enhancer is not None:
            _write(denoised_path, enhancer(degraded))
            synthesizer.synthesize_baseline(item["synthesis_text"], item["prompt_text"], denoised_path, denoised_output_path)

        reference_embedding = _xvector(speaker_model, Path(item["reference_wav_path"]), device)
        baseline_similarity = float(reference_embedding @ _xvector(speaker_model, baseline_path, device))
        robust_similarity = float(reference_embedding @ _xvector(speaker_model, robust_path, device))
        row = dict(item) | {
            "baseline_xvector_similarity": baseline_similarity,
            "robust_xvector_similarity": robust_similarity,
            "delta": robust_similarity - baseline_similarity,
            "denoiser_xvector_similarity": float(reference_embedding @ _xvector(speaker_model, denoised_output_path, device)) if enhancer else None,
            "baseline_utmos": utmos.score(baseline_path) if utmos else None,
            "robust_utmos": utmos.score(robust_path) if utmos else None,
            "denoiser_utmos": utmos.score(denoised_output_path) if utmos and enhancer else None,
            "utmos_implementation": utmos.implementation if utmos else None,
            "degraded_mel_lsd_db": mel_lsd_db(degraded_mel[:, :frames].numpy(), prompt_mel[:, :frames].numpy()),
            "bwe_mel_lsd_db": mel_lsd_db(predicted[0].cpu().numpy(), prompt_mel[:, :frames].numpy()),
            "bwe_gate": float(gate.item()),
            "degradation_metadata": json.dumps(degradation_metadata, ensure_ascii=False, sort_keys=True),
            "quality_metadata": json.dumps(estimated_metadata, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row); _atomic_csv(rows, result_path)
        print(json.dumps({"event": "evaluated", "index": index, "total": len(protocol), "sample_id": item["sample_id"], "delta": row["delta"]}, ensure_ascii=False), flush=True)

    summary = paired_summary(pd.DataFrame(rows))
    summary.update({
        "samples": len(rows),
        "protocol_path": str(args.protocol),
        "checkpoint_path": str(args.checkpoint),
        "formal": args.formal,
        "three_way_baseline": enhancer is not None,
        "utmos_implementation": utmos.implementation if utmos else None,
    })
    (args.output / "evaluation.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
