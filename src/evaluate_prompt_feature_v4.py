"""End-to-end evaluation of direct CosyVoice prompt-feature restoration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.cosyvoice_v4 import CosyVoiceNativeCampPlus
from src.degradation_v3 import DegradationAssets, FormalDegradationEngine
from src.evaluate_v3 import _validate_formal_protocol
from src.evaluation_v4_common import atomic_csv, load_audio, load_protocol, speaker_model, write_prompt, xvector
from src.metrics import paired_summary
from src.models.prompt_feature_restorer import build_feature_restorer_from_checkpoint
from src.quality import estimate_quality, high_band_energy_ratio
from src.utmos import UTMOS22StrongScorer


def _alphas(value: str) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if not result or any(value < 0 or value > 1 for value in result):
        raise argparse.ArgumentTypeError("alphas must lie within [0, 1]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--degradation-assets", type=Path, required=True)
    parser.add_argument("--asset-split", choices=("selection", "test"), default="selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=_alphas, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--skip-utmos", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--high-band-ratio-threshold", type=float, default=0.0, help="0 disables safe narrow-band routing")
    args = parser.parse_args()
    if args.formal and (len(args.alphas) != 1 or args.asset_split != "test" or args.skip_utmos):
        raise SystemExit("formal evaluation requires one selected alpha, test assets and UTMOS")
    protocol = load_protocol(args.protocol)
    if args.formal:
        _validate_formal_protocol(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    restorer = build_feature_restorer_from_checkpoint(checkpoint, device).eval()
    synthesizer = CosyVoiceNativeCampPlus(args.model_root, args.cosyvoice_root)
    engine = FormalDegradationEngine(DegradationAssets.from_manifest(args.degradation_assets, args.asset_split), False)
    evaluator = speaker_model(args.output, device)
    utmos = None if args.skip_utmos else UTMOS22StrongScorer(device)
    result_path = args.output / "evaluation.csv"
    rows = pd.read_csv(result_path).to_dict(orient="records") if args.resume and result_path.exists() else []
    completed = {(str(row["sample_id"]), float(row["alpha"])) for row in rows}
    for index, item in enumerate(protocol, 1):
        sample_root = args.output / "audio" / item["sample_id"]
        degraded, metadata = engine.apply(load_audio(Path(item["prompt_wav_path"])), item["degradation"], item["severity"], item["seed"])
        degraded_path = sample_root / "degraded_prompt.wav"
        write_prompt(degraded_path, degraded)
        baseline_path = sample_root / "baseline.wav"
        if not baseline_path.exists():
            synthesizer.synthesize_baseline(item["synthesis_text"], item["prompt_text"], degraded_path, baseline_path, item["seed"])
        reference = xvector(evaluator, Path(item["reference_wav_path"]), device)
        baseline_similarity = float(reference @ xvector(evaluator, baseline_path, device))
        feature = synthesizer.extract_prompt_feature(degraded_path).to(device)
        quality, quality_metadata = estimate_quality(degraded)
        high_band_ratio = high_band_energy_ratio(degraded)
        with torch.inference_mode():
            predicted, diagnostics = restorer(feature, quality[None].to(device))
        for alpha in args.alphas:
            if (item["sample_id"], alpha) in completed:
                continue
            effective_alpha = alpha if args.high_band_ratio_threshold <= 0 or high_band_ratio < args.high_band_ratio_threshold else 0.0
            if effective_alpha == 0:
                output_path, similarity = baseline_path, baseline_similarity
            else:
                corrected = feature + effective_alpha * (predicted - feature)
                output_path = sample_root / f"feature-alpha-{alpha:.2f}-effective-{effective_alpha:.2f}.wav"
                synthesizer.synthesize_prompt_feature(item["synthesis_text"], item["prompt_text"], degraded_path, corrected.cpu(), output_path, item["seed"])
                similarity = float(reference @ xvector(evaluator, output_path, device))
            baseline_utmos = utmos.score(baseline_path) if utmos else None
            calibrated_utmos = baseline_utmos if output_path == baseline_path else utmos.score(output_path) if utmos else None
            rows.append(
                dict(item)
                | {
                    "alpha": alpha,
                    "effective_alpha": effective_alpha,
                    "high_band_energy_ratio": high_band_ratio,
                    "router_triggered": effective_alpha > 0,
                    "baseline_xvector_similarity": baseline_similarity,
                    "calibrated_xvector_similarity": similarity,
                    "delta": similarity - baseline_similarity,
                    "baseline_utmos": baseline_utmos,
                    "calibrated_utmos": calibrated_utmos,
                    "feature_gate": float(diagnostics["gate"].item()),
                    "feature_shift_l1": float((predicted - feature).abs().mean()),
                    "degradation_metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    "quality_metadata": json.dumps(quality_metadata, ensure_ascii=False, sort_keys=True),
                }
            )
            atomic_csv(rows, result_path)
        print(json.dumps({"event": "feature_v4_evaluated", "index": index, "total": len(protocol)}), flush=True)
    frame = pd.DataFrame(rows)
    summaries = {
        str(float(alpha)): paired_summary(group, baseline_column="baseline_xvector_similarity", robust_column="calibrated_xvector_similarity")
        for alpha, group in frame.groupby("alpha", sort=True)
    }
    summary = {"samples": len(protocol), "rows": len(frame), "alphas": args.alphas, "high_band_ratio_threshold": args.high_band_ratio_threshold, "router_trigger_rate": float(frame["router_triggered"].mean()), "formal": args.formal, "checkpoint": str(args.checkpoint), "utmos_implementation": utmos.implementation if utmos else None, "summaries": summaries}
    (args.output / "evaluation.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
