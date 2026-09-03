"""Convert tabular corpora and merge domains without speaker-ID collisions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def tabular_to_manifest(path: Path, domain: str) -> dict[str, list[dict]]:
    frame = pd.read_csv(path)
    required = {"speaker", "audio_id", "wav_path", "text"}; missing = required - set(frame.columns)
    if missing: raise ValueError(f"tabular manifest missing columns: {sorted(missing)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        speaker = f"{domain}:{row['speaker']}"
        grouped[speaker].append({"audio_id": str(row["audio_id"]), "wav_path": str(Path(row["wav_path"]).resolve()), "text": str(row["text"]), "domain": domain})
    return dict(grouped)


def merge_manifests(named_manifests: list[tuple[str, dict[str, list[dict]]]]) -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {}
    for domain, manifest in named_manifests:
        for speaker, rows in manifest.items():
            key = speaker if speaker.startswith(f"{domain}:") else f"{domain}:{speaker}"
            if key in merged: raise ValueError(f"duplicate domain-qualified speaker: {key}")
            merged[key] = [dict(row) | {"domain": domain} for row in rows]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(); subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert"); convert.add_argument("--input", type=Path, required=True); convert.add_argument("--domain", required=True); convert.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge"); merge.add_argument("--input", action="append", required=True, help="DOMAIN=manifest.json"); merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "convert": result = tabular_to_manifest(args.input, args.domain)
    else:
        named = []
        for specification in args.input:
            domain, path = specification.split("=", 1); named.append((domain, json.loads(Path(path).read_text(encoding="utf-8"))))
        result = merge_manifests(named)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"speakers": len(result), "utterances": sum(map(len, result.values())), "output": str(args.output)}))


if __name__ == "__main__": main()
