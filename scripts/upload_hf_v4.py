"""Upload the V4 checkpoint and model card without persisting an access token."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit("Hugging Face token must be supplied on stdin")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo="v4/prompt_feature_restorer.pt",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Add V4 narrowband prompt feature restorer",
    )
    api.upload_file(
        path_or_fileobj=args.model_card,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Document V4 evidence and limitations",
    )
    info = api.model_info(args.repo_id, revision=args.revision)
    files = {item.rfilename for item in info.siblings}
    required = {"README.md", "v4/prompt_feature_restorer.pt"}
    missing = sorted(required - files)
    if missing:
        raise SystemExit(f"upload verification failed; missing: {missing}")
    print(f"verified repo={args.repo_id} revision={info.sha} files={sorted(required)}")


if __name__ == "__main__":
    main()
