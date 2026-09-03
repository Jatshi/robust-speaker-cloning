"""Select a safe interpolation coefficient with alpha=0 as a no-op fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.metrics import paired_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-win-rate", type=float, default=0.50)
    args = parser.parse_args()
    frame = pd.read_csv(args.evaluation)
    candidates = []
    for alpha, group in frame.groupby("alpha", sort=True):
        summary = paired_summary(
            group,
            baseline_column="baseline_xvector_similarity",
            robust_column="calibrated_xvector_similarity",
        )["comparisons"][0]
        candidates.append({"alpha": float(alpha), **summary})
    eligible = [
        row
        for row in candidates
        if row["alpha"] > 0 and row["mean_delta"] > 0 and row["win_rate"] >= args.minimum_win_rate
    ]
    if eligible:
        selected = max(eligible, key=lambda row: (row["mean_delta"], row["win_rate"]))
        passed = True
    else:
        selected = next(row for row in candidates if row["alpha"] == 0.0)
        passed = False
    report = {
        "gate_passed": passed,
        "selected_alpha": selected["alpha"],
        "selection_rule": "max mean delta among alpha>0 with mean_delta>0 and win_rate>=minimum; otherwise alpha=0",
        "minimum_win_rate": args.minimum_win_rate,
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
