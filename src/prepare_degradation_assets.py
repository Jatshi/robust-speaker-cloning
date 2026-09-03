"""Index user-provided noise and RIR corpora with explicit provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(roots: list[Path], source: str) -> list[dict]:
    paths = sorted({path.resolve() for root in roots for suffix in ("*.wav", "*.flac") for path in root.rglob(suffix)})
    result = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest(); bucket = int(checksum[:8], 16) % 100
        split = "train" if bucket < 80 else "selection" if bucket < 90 else "test"
        result.append({"path": str(path), "sha256": checksum, "source": source, "split": split})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-root", action="append", type=Path, default=[])
    parser.add_argument("--speech-root", action="append", type=Path, default=[])
    parser.add_argument("--music-root", action="append", type=Path, default=[])
    parser.add_argument("--rir-root", action="append", type=Path, default=[])
    parser.add_argument("--noise-source", default="MUSAN/DNS (record exact local subset in DATA_CARD.md)")
    parser.add_argument("--rir-source", default="OpenSLR RIRS_NOISES (record exact local subset in DATA_CARD.md)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"noise": _rows(args.noise_root, args.noise_source), "speech": _rows(args.speech_root, "MUSAN speech"), "music": _rows(args.music_root, "MUSAN music"), "rir": _rows(args.rir_root, args.rir_source)}
    if any(not payload[key] for key in ("noise", "speech", "music", "rir")):
        raise SystemExit("Formal assets require distinct noise, speech, music and RIR roots")
    missing_partitions = [(category, split) for category, rows in payload.items() for split in ("train", "selection", "test") if not any(row["split"] == split for row in rows)]
    if missing_partitions:
        raise SystemExit(f"Not enough assets for deterministic train/selection/test partitions: {missing_partitions}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) for key, value in payload.items()}))


if __name__ == "__main__":
    main()
