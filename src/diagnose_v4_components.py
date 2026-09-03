"""Oracle component ablation: embedding vs tokens vs acoustic prompt features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.cosyvoice_v4 import CosyVoiceNativeCampPlus
from src.degradation_v3 import DegradationAssets, FormalDegradationEngine
from src.evaluation_v4_common import atomic_csv, load_audio, load_protocol, speaker_model, write_prompt, xvector
from src.metrics import paired_summary


SYSTEMS = {
    "degraded_baseline": set(),
    "clean_embedding_only": {"embedding"},
    "clean_tokens_only": {"tokens"},
    "clean_features_only": {"features"},
    "clean_tokens_features": {"tokens", "features"},
    "full_clean_oracle": {"embedding", "tokens", "features"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cosyvoice-root", type=Path, required=True)
    parser.add_argument("--degradation-assets", type=Path, required=True)
    parser.add_argument("--asset-split", choices=("selection", "test"), default="selection")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthesizer = CosyVoiceNativeCampPlus(args.model_root, args.cosyvoice_root)
    engine = FormalDegradationEngine(DegradationAssets.from_manifest(args.degradation_assets, args.asset_split), False)
    evaluator = speaker_model(args.output, device)
    rows: list[dict] = []
    for index, item in enumerate(load_protocol(args.protocol), 1):
        sample_root = args.output / "audio" / item["sample_id"]
        degraded, metadata = engine.apply(
            load_audio(Path(item["prompt_wav_path"])), item["degradation"], item["severity"], item["seed"]
        )
        degraded_path = sample_root / "degraded_prompt.wav"
        write_prompt(degraded_path, degraded)
        reference = xvector(evaluator, Path(item["reference_wav_path"]), device)
        similarities = {}
        for name, clean_components in SYSTEMS.items():
            output_path = sample_root / f"{name}.wav"
            synthesizer.synthesize_mixed_prompt(
                item["synthesis_text"],
                item["prompt_text"],
                degraded_path,
                Path(item["prompt_wav_path"]),
                clean_components,
                output_path,
                item["seed"],
            )
            similarities[name] = float(reference @ xvector(evaluator, output_path, device))
        rows.append(
            dict(item)
            | similarities
            | {"degradation_metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True)}
        )
        atomic_csv(rows, args.output / "component_diagnostic.csv")
        print(json.dumps({"event": "v4_component_diagnostic", "index": index, "sample_id": item["sample_id"]}), flush=True)
    frame = pd.DataFrame(rows)
    comparisons = {
        system: paired_summary(frame, baseline_column="degraded_baseline", robust_column=system)
        for system in SYSTEMS
        if system != "degraded_baseline"
    }
    overall = {
        system: comparison["comparisons"][0]
        for system, comparison in comparisons.items()
    }
    summary = {
        "samples": len(frame),
        "oracle_only": True,
        "best_component": max(overall, key=lambda key: overall[key]["mean_delta"]),
        "overall": overall,
        "comparisons": comparisons,
    }
    (args.output / "component_diagnostic.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
