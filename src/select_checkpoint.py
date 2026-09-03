"""Select a checkpoint by held-out end-to-end ECAPA, not training loss alone."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True); parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True); args = parser.parse_args()
    rows = []
    for candidate in json.loads(args.candidates.read_text(encoding="utf-8")):
        evaluation = args.evaluation_root / f"epoch-{candidate['epoch']:03d}" / "evaluation.csv"
        frame = pd.read_csv(evaluation)
        rows.append(candidate | {"selection_samples": len(frame), "robust_xvector_mean": float(frame.robust_xvector_similarity.mean())})
    winner = max(rows, key=lambda row: row["robust_xvector_mean"])
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(winner["checkpoint"], args.output_checkpoint)
    report = {"criterion": "held-out selection-speaker end-to-end robust ECAPA mean", "winner": winner, "candidates": rows}
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(report))


if __name__ == "__main__": main()
