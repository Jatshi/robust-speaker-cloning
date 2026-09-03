"""Create a compact, hash-addressed V3 evidence bundle after formal completion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict:
    """Record Git provenance when available without rejecting deployed snapshots."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    if revision.returncode != 0:
        return {
            "available": False,
            "revision": None,
            "dirty": None,
            "note": "deployment snapshot does not contain Git metadata",
        }
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    )
    return {
        "available": True,
        "revision": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "note": None,
    }


def _files_under(root: Path, sources: list[str]) -> list[str]:
    files: set[str] = set()
    for relative in sources:
        path = root / relative
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
        elif path.is_dir():
            files.update(item.relative_to(root).as_posix() for item in path.rglob("*") if item.is_file())
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("robust-speaker-cloning-v3-evidence.tar.gz")); args = parser.parse_args()
    root = args.root.resolve()
    required = [
        "outputs/checkpoints/selected.pt", "outputs/checkpoints/history.json", "outputs/selection/selection.json",
        "outputs/evaluation-v3/evaluation.csv", "outputs/evaluation-v3/evaluation.summary.json",
        "data/eval_protocol.jsonl", "data/feature_cache/metadata.json", "data/feature_cache/degradation_metadata.jsonl",
        "third_party.lock", "README.md",
    ]
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing: raise FileNotFoundError("formal V3 evidence incomplete: " + ", ".join(missing))
    summary = json.loads((root / "outputs/evaluation-v3/evaluation.summary.json").read_text(encoding="utf-8"))
    if not summary.get("formal") or int(summary.get("samples", 0)) != 105:
        raise ValueError("evaluation summary is not a 105-sample formal run")
    archive_sources = [
        *required, "src", "scripts", "docs", "tests", "app.py", "requirements.txt",
        "requirements-autodl.txt", "requirements-cosyvoice-runtime.txt", "pyproject.toml",
    ]
    hashed_files = _files_under(root, archive_sources)
    provenance = _git_provenance(root)
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": provenance,
        "files": {relative: _sha256(root / relative) for relative in hashed_files},
    }
    manifest_path = root / "outputs" / "evidence_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = args.output if args.output.is_absolute() else root / args.output
    with tarfile.open(archive, "w:gz") as bundle:
        for relative in [*archive_sources, "outputs/evidence_manifest.json"]:
            bundle.add(root / relative, arcname=relative)
    print(json.dumps({"archive": str(archive), "sha256": _sha256(archive), "git": provenance}))


if __name__ == "__main__": main()
