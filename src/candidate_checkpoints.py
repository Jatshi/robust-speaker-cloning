"""Choose low-validation-loss candidates for end-to-end model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True); parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    ordered = sorted(history, key=lambda row: float(row["validation"]["total"]))[: args.top_k]
    candidates = [{"epoch": row["epoch"], "validation_total": row["validation"]["total"], "checkpoint": str(args.checkpoint_dir / f"epoch-{row['epoch']:03d}.pt")} for row in ordered]
    if any(not Path(row["checkpoint"]).exists() for row in candidates): raise FileNotFoundError("candidate epoch checkpoint missing")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(json.dumps(candidates))


if __name__ == "__main__": main()
