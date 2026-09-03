"""Build a balanced, leakage-resistant end-to-end evaluation protocol."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.degradation_v3 import FORMAL_DEGRADATION_TYPES
from src.splits import split_speakers

DEFAULT_SYNTHESIS_TEXTS = (
    "人工智能系统需要在复杂环境中保持稳定和可信。",
    "今天的实验重点是验证模型在真实退化下的表现。",
    "请把窗边的蓝色文件夹放到会议桌上。",
    "可靠的工程结论必须能够由原始证据复现。",
    "下周三上午我们将讨论下一阶段的测试计划。",
)


def build_eval_protocol(
    manifest: dict[str, list[dict]],
    samples_per_type: int,
    seed: int = 2026,
    synthesis_texts: tuple[str, ...] = DEFAULT_SYNTHESIS_TEXTS,
) -> list[dict]:
    """Use different utterances for degraded prompt and clean identity reference."""
    candidates = [(speaker, rows) for speaker, rows in manifest.items() if len(rows) >= 2]
    if not candidates:
        raise ValueError("evaluation requires speakers with at least two clean utterances")
    generator = random.Random(seed)
    generator.shuffle(candidates)
    rows: list[dict] = []
    for kind_index, kind in enumerate(FORMAL_DEGRADATION_TYPES):
        for sample_index in range(samples_per_type):
            speaker, utterances = candidates[(kind_index * samples_per_type + sample_index) % len(candidates)]
            prompt_index = (kind_index + sample_index) % len(utterances)
            reference_index = (prompt_index + 1) % len(utterances)
            prompt, reference = utterances[prompt_index], utterances[reference_index]
            synthesis_text = synthesis_texts[(kind_index + sample_index) % len(synthesis_texts)]
            if synthesis_text == prompt.get("text"):
                synthesis_text = synthesis_texts[(kind_index + sample_index + 1) % len(synthesis_texts)]
            rows.append({
                "sample_id": f"{speaker}-{kind}-{sample_index:03d}",
                "speaker": speaker,
                "prompt_audio_id": prompt["audio_id"],
                "prompt_wav_path": prompt["wav_path"],
                "prompt_text": prompt["text"],
                "reference_audio_id": reference["audio_id"],
                "reference_wav_path": reference["wav_path"],
                "synthesis_text": synthesis_text,
                "degradation": kind,
                "severity": sample_index % 3,
                "seed": seed + kind_index * 10000 + sample_index,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-type", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--speaker-split", choices=("selection", "test", "all"), default="test")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.speaker_split != "all":
        selected = split_speakers(list(manifest), args.split_seed)[args.speaker_split]
        manifest = {speaker: rows for speaker, rows in manifest.items() if speaker in selected}
    protocol = build_eval_protocol(manifest, args.samples_per_type, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in protocol) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(protocol), "samples_per_type": args.samples_per_type, "speaker_split": args.speaker_split, "speakers": len(manifest), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
