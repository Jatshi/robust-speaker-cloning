"""仅在 V2 训练和端到端评测完整时允许打包。"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--output", type=Path, default=Path("robust-speaker-cloning-v2.tar.gz")); args = parser.parse_args(); root = args.root.resolve()
    required = ("outputs/checkpoints/best.pt", "outputs/checkpoints/last.pt", "outputs/checkpoints/history.json", "outputs/evaluation/evaluation.csv", "outputs/evaluation/evaluation.summary.json", "app.py", "README.md")
    absent = [item for item in required if not (root / item).exists()]
    if absent: raise FileNotFoundError("V2 incomplete: " + ", ".join(absent))
    archive = args.output if args.output.is_absolute() else root / args.output
    with tarfile.open(archive, "w:gz") as bundle:
        for item in ("src", "scripts", "app.py", "requirements.txt", "README.md", "outputs/checkpoints", "outputs/evaluation"):
            bundle.add(root / item, arcname=item)
    print(archive)


if __name__ == "__main__": main()
