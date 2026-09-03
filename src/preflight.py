"""Fail-fast checks before paid AutoDL preparation, training or evaluation."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch


def _codec_names() -> str:
    if not shutil.which("ffmpeg"):
        return ""
    return subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("prepare", "train", "eval"), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd()); parser.add_argument("--minimum-free-gb", type=float, default=60)
    args = parser.parse_args(); root = args.project_root.resolve(); checks: dict[str, dict] = {}
    def check(name: str, passed: bool, detail: str) -> None: checks[name] = {"passed": bool(passed), "detail": detail}
    check("project", (root / "src/models/robust_speaker_encoder.py").exists(), str(root))
    free_gb = shutil.disk_usage(root).free / 1024**3; check("disk", free_gb >= args.minimum_free_gb, f"{free_gb:.1f} GiB free")
    check("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3; check("gpu_memory", memory_gb >= 20, f"{memory_gb:.1f} GiB")
    codecs = _codec_names(); required_codecs = ("libmp3lame", "libopus", "pcm_alaw", "pcm_mulaw")
    check("ffmpeg_codecs", all(codec in codecs for codec in required_codecs), ",".join(codec for codec in required_codecs if codec in codecs) or "missing")
    cosyvoice_cli, cosyvoice_frontend = root / "CosyVoice/cosyvoice/cli/cosyvoice.py", root / "CosyVoice/cosyvoice/cli/frontend.py"
    if cosyvoice_cli.exists() and cosyvoice_frontend.exists():
        cli_text, frontend_text = cosyvoice_cli.read_text(encoding="utf-8"), cosyvoice_frontend.read_text(encoding="utf-8")
        contract = all(token in cli_text + frontend_text for token in ("def AutoModel", "frontend_zero_shot", "llm_embedding", "flow_embedding"))
        check("cosyvoice_interface", contract, "AutoModel + zero-shot + dual 192-D embedding fields")
    else:
        check("cosyvoice_interface", False, "pinned CosyVoice checkout not found")
    model_files = (root / "models/CosyVoice2-0.5B/cosyvoice2.yaml", root / "models/CosyVoice2-0.5B/campplus.onnx")
    check("cosyvoice_model", all(path.is_file() for path in model_files), ", ".join(str(path) for path in model_files))
    if args.stage in {"train", "eval"}:
        for relative in ("data/speaker_to_files.json", "data/cosyvoice_teacher.pt", "data/feature_cache/metadata.json"):
            check(relative, (root / relative).exists(), str(root / relative))
        metadata_path = root / "data/feature_cache/metadata.json"
        backend = json.loads(metadata_path.read_text(encoding="utf-8")).get("degradation_backend") if metadata_path.exists() else None
        check("formal_cache_backend", backend == "real_assets_and_ffmpeg", str(backend))
        assets_path = root / "data/degradation_assets.json"
        assets = json.loads(assets_path.read_text(encoding="utf-8")) if assets_path.exists() else {}
        asset_splits_ok = all(any(row.get("split") == split for row in assets.get(key, [])) for key in ("noise", "speech", "music", "rir") for split in ("train", "selection", "test"))
        check("formal_assets", asset_splits_ok, str(assets_path))
    if args.stage == "eval":
        for relative in ("outputs/checkpoints/selected.pt", "data/eval_protocol.jsonl", "models/CosyVoice2-0.5B", "CosyVoice"):
            check(relative, (root / relative).exists(), str(root / relative))
        protocol_path = root / "data/eval_protocol.jsonl"
        protocol = [json.loads(line) for line in protocol_path.read_text(encoding="utf-8").splitlines() if line] if protocol_path.exists() else []
        check("formal_protocol", len(protocol) == 105 and len({row.get("sample_id") for row in protocol}) == 105, f"{len(protocol)} rows")
    passed = all(row["passed"] for row in checks.values()); report = {"passed": passed, "stage": args.stage, "checks": checks}
    (root / "outputs").mkdir(exist_ok=True); (root / "outputs" / f"preflight-{args.stage}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed: raise SystemExit(2)


if __name__ == "__main__": main()
