"""构建 V2 的 AISHELL 说话人清单，不复制原始音频。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_manifest(aishell_root: Path, output: Path, max_per_speaker: int = 15) -> dict[str, list[dict[str, str]]]:
    transcript_path = aishell_root / "transcript" / "aishell_transcript_v0.8.txt"
    transcripts = {parts[0]: "".join(parts[1:]) for line in transcript_path.read_text(encoding="utf-8").splitlines()
                   if (parts := line.split()) and len(parts) > 1}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for wav in sorted((aishell_root / "wav").glob("*/*/*.wav")):
        if wav.stem in transcripts:
            grouped[wav.parent.name].append({"audio_id": wav.stem, "wav_path": str(wav), "text": transcripts[wav.stem]})
    selected = {speaker: rows[:max_per_speaker] for speaker, rows in sorted(grouped.items()) if len(rows) >= 3}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"speakers": len(selected), "clean_utterances": sum(map(len, selected.values()))}, ensure_ascii=False))
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aishell-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-speaker", type=int, default=15)
    arguments = parser.parse_args()
    build_manifest(arguments.aishell_root, arguments.output, arguments.max_per_speaker)
