"""End-to-end alpha selection and formal evaluation for V4."""
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
from src.models.campplus_residual_adapter import build_adapter_from_checkpoint
from src.quality import estimate_quality
from src.utmos import UTMOS22StrongScorer


def _parse_alphas(value: str) -> list[float]:
    alphas = [float(item) for item in value.split(",")]
    if not alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must be a comma-separated subset of [0, 1]")
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--degradation-assets", type=Path, required=True)
    parser.add_argument("--asset-split", choices=("selection", "test"), default="selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=_parse_alphas, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--skip-utmos", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.formal and (len(args.alphas) != 1 or args.asset_split != "test" or args.skip_utmos):
        raise SystemExit("formal V4 requires one pre-selected alpha, test assets and UTMOS")

    protocol = load_protocol(args.protocol)
    if args.formal:
        _validate_formal_protocol(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    adapter = build_adapter_from_checkpoint(checkpoint, device).eval()
    synthesizer = CosyVoiceNativeCampPlus(args.model_root, args.cosyvoice_root)
    engine = FormalDegradationEngine(DegradationAssets.from_manifest(args.degradation_assets, args.asset_split), False)
    evaluator = speaker_model(args.output, device)
    utmos = None if args.skip_utmos else UTMOS22StrongScorer(device)
    result_path = args.output / "evaluation.csv"
    rows = pd.read_csv(result_path).to_dict(orient="records") if args.resume and result_path.exists() else []
    completed = {(str(row["sample_id"]), float(row["alpha"])) for row in rows}

    for index, item in enumerate(protocol, 1):
        sample_root = args.output / "audio" / item["sample_id"]
        degraded, metadata = engine.apply(
            load_audio(Path(item["prompt_wav_path"])), item["degradation"], item["severity"], item["seed"]
        )
        degraded_path = sample_root / "degraded_prompt.wav"
        write_prompt(degraded_path, degraded)
        baseline_path = sample_root / "baseline.wav"
        if not baseline_path.exists():
            synthesizer.synthesize_baseline(
                item["synthesis_text"], item["prompt_text"], degraded_path, baseline_path, item["seed"]
            )
        reference = xvector(evaluator, Path(item["reference_wav_path"]), device)
        baseline_similarity = float(reference @ xvector(evaluator, baseline_path, device))
        raw_embedding = synthesizer.extract_raw_embedding(degraded_path).to(device)
        quality, quality_metadata = estimate_quality(degraded)
        with torch.inference_mode():
            predicted, diagnostics = adapter(raw_embedding[None], quality[None].to(device), alpha=1.0)
        for alpha in args.alphas:
            if (item["sample_id"], alpha) in completed:
                continue
            if alpha == 0.0:
                calibrated_path = baseline_path
                calibrated_similarity = baseline_similarity
            else:
                condition = raw_embedding + alpha * (predicted[0] - raw_embedding)
                calibrated_path = sample_root / f"calibrated-alpha-{alpha:.2f}.wav"
                synthesizer.synthesize_raw_embedding(
                    item["synthesis_text"],
                    item["prompt_text"],
                    degraded_path,
                    condition.cpu(),
                    calibrated_path,
                    item["seed"],
                )
                calibrated_similarity = float(reference @ xvector(evaluator, calibrated_path, device))
            row = dict(item) | {
                "alpha": alpha,
                "baseline_xvector_similarity": baseline_similarity,
                "calibrated_xvector_similarity": calibrated_similarity,
                "delta": calibrated_similarity - baseline_similarity,
                "baseline_utmos": utmos.score(baseline_path) if utmos else None,
                "calibrated_utmos": utmos.score(calibrated_path) if utmos else None,
                "adapter_gate": float(diagnostics["gate"].item()),
                "raw_embedding_norm": float(raw_embedding.norm()),
                "predicted_embedding_norm": float(predicted[0].norm()),
                "embedding_shift_norm": float((predicted[0] - raw_embedding).norm()),
                "degradation_metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                "quality_metadata": json.dumps(quality_metadata, ensure_ascii=False, sort_keys=True),
            }
            rows.append(row)
            atomic_csv(rows, result_path)
        print(json.dumps({"event": "v4_evaluated", "index": index, "total": len(protocol)}), flush=True)

    frame = pd.DataFrame(rows)
    summaries = {}
    for alpha, group in frame.groupby("alpha", sort=True):
        summaries[str(float(alpha))] = paired_summary(
            group,
            baseline_column="baseline_xvector_similarity",
            robust_column="calibrated_xvector_similarity",
        )
    summary = {
        "samples": len(protocol),
        "rows": len(frame),
        "alphas": args.alphas,
        "checkpoint": str(args.checkpoint),
        "formal": args.formal,
        "utmos_implementation": utmos.implementation if utmos else None,
        "summaries": summaries,
    }
    (args.output / "evaluation.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
