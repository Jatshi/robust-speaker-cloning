"""Decisive pre-training test for CAMPPlus scale and embedding headroom."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.cosyvoice_v4 import CosyVoiceNativeCampPlus
from src.degradation_v3 import DegradationAssets, FormalDegradationEngine
from src.evaluation_v4_common import atomic_csv, load_audio, load_protocol, speaker_model, waveform_mae, write_prompt, xvector
from src.metrics import paired_summary


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
        raw_degraded = synthesizer.extract_raw_embedding(degraded_path)
        raw_clean = synthesizer.extract_raw_embedding(Path(item["prompt_wav_path"]))
        conditions = {
            "official_baseline": None,
            "raw_reinject": raw_degraded,
            "unitnorm_reinject": torch.nn.functional.normalize(raw_degraded, dim=0),
            "clean_raw_oracle": raw_clean,
        }
        reference = xvector(evaluator, Path(item["reference_wav_path"]), device)
        paths: dict[str, Path] = {}
        similarities: dict[str, float] = {}
        for name, condition in conditions.items():
            output_path = sample_root / f"{name}.wav"
            if condition is None:
                synthesizer.synthesize_baseline(
                    item["synthesis_text"], item["prompt_text"], degraded_path, output_path, item["seed"]
                )
            else:
                synthesizer.synthesize_raw_embedding(
                    item["synthesis_text"], item["prompt_text"], degraded_path, condition, output_path, item["seed"]
                )
            paths[name] = output_path
            similarities[name] = float(reference @ xvector(evaluator, output_path, device))
        row = dict(item) | similarities | {
            "raw_reinject_waveform_mae": waveform_mae(paths["official_baseline"], paths["raw_reinject"]),
            "degraded_embedding_norm": float(raw_degraded.norm()),
            "unitnorm_embedding_norm": float(torch.nn.functional.normalize(raw_degraded, dim=0).norm()),
            "clean_embedding_norm": float(raw_clean.norm()),
            "degradation_metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)
        atomic_csv(rows, args.output / "diagnostic.csv")
        print(json.dumps({"event": "v4_diagnostic", "index": index, "sample_id": item["sample_id"]}), flush=True)

    frame = pd.DataFrame(rows)
    comparisons = {}
    for column in ("raw_reinject", "unitnorm_reinject", "clean_raw_oracle"):
        comparisons[column] = paired_summary(
            frame, baseline_column="official_baseline", robust_column=column
        )
    raw_delta = float((frame["raw_reinject"] - frame["official_baseline"]).mean())
    unit_delta = float((frame["unitnorm_reinject"] - frame["raw_reinject"]).mean())
    oracle_delta = float((frame["clean_raw_oracle"] - frame["official_baseline"]).mean())
    summary = {
        "samples": len(frame),
        "raw_reinject_minus_baseline": raw_delta,
        "unitnorm_minus_raw_reinject": unit_delta,
        "clean_oracle_minus_baseline": oracle_delta,
        "mean_raw_reinject_waveform_mae": float(frame["raw_reinject_waveform_mae"].mean()),
        "gate_interface_consistent": abs(raw_delta) <= 0.02,
        "gate_unitnorm_non_equivalent": abs(unit_delta) >= 0.02,
        "gate_embedding_headroom": oracle_delta > 0.0,
        "proceed_to_training": abs(raw_delta) <= 0.02 and oracle_delta > 0.0,
        "comparisons": comparisons,
    }
    (args.output / "diagnostic.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
