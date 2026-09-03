#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning}
PHASE=${1:-all}
cd "$PROJECT_DIR"
screen -S robust_clone_v3 -X quit >/dev/null 2>&1 || true
screen -dmS robust_clone_v3 bash -lc "cd '$PROJECT_DIR' && PROJECT_DIR='$PROJECT_DIR' bash scripts/run_v3.sh '$PHASE'"
echo "started screen session robust_clone_v3; inspect with: screen -r robust_clone_v3"
