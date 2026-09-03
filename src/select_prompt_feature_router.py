"""Lock a deployable narrow-band router using selection data only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from src.evaluation_v4_common import load_audio
from src.quality import high_band_energy_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.evaluation)
    frame = frame[frame["alpha"] == args.alpha].copy()
    frame["high_band_energy_ratio"] = [
        high_band_energy_ratio(load_audio(args.audio_root / sample_id / "degraded_prompt.wav"))
        for sample_id in frame["sample_id"]
    ]
    telephone = frame[frame["degradation"] == "telephone"]
    others = frame[frame["degradation"] != "telephone"]
    telephone_max = float(telephone["high_band_energy_ratio"].max())
    other_min = float(others["high_band_energy_ratio"].min())
    separated = telephone_max < other_min
    threshold = math.sqrt(telephone_max * other_min) if separated else float("nan")
    telephone_delta = float(telephone["delta"].mean())
    telephone_win_rate = float((telephone["delta"] > 0).mean())
    passed = separated and telephone_delta > 0 and telephone_win_rate == 1.0
    report = {
        "gate_passed": passed,
        "selection_only": True,
        "alpha": args.alpha,
        "threshold": threshold,
        "cutoff_hz": 4200.0,
        "telephone_max_ratio": telephone_max,
        "nontelephone_min_ratio": other_min,
        "telephone_mean_delta": telephone_delta,
        "telephone_win_rate": telephone_win_rate,
        "rule": "apply restoration iff high-band energy ratio above 4.2 kHz is below threshold",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
