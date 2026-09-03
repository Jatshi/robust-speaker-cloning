#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning}
COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
# ModelScope exposes this repository through the named `master` revision rather
# than a Git commit SHA.  Pin the accepted server-side revision and write a
# content-addressed SHA256 manifest after download so the exact resolved
# snapshot remains auditable even if the branch later moves.
COSYVOICE_MODEL_REVISION=${COSYVOICE_MODEL_REVISION:-master}
cd "$PROJECT_DIR"

retry_network() {
  local attempt=1
  local max_attempts=4
  until "$@"; do
    if (( attempt >= max_attempts )); then
      echo "network command failed after ${max_attempts} attempts: $*" >&2
      return 1
    fi
    echo "network command failed (attempt ${attempt}/${max_attempts}); retrying..." >&2
    sleep $((attempt * 5))
    attempt=$((attempt + 1))
  done
}

if ! command -v ffmpeg >/dev/null || ! command -v aria2c >/dev/null || ! command -v unzip >/dev/null; then
  apt-get update
  apt-get install -y ffmpeg aria2 unzip
fi
python -m pip install --upgrade pip==24.2 wheel setuptools==70.3.0
python -m pip install -r requirements-autodl.txt

if [[ ! -d CosyVoice/.git ]]; then
  retry_network git -c http.version=HTTP/1.1 clone --filter=blob:none https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
fi
if ! git -C CosyVoice cat-file -e "${COSYVOICE_COMMIT}^{commit}" 2>/dev/null; then
  retry_network git -c http.version=HTTP/1.1 -C CosyVoice fetch --depth 1 origin "$COSYVOICE_COMMIT"
fi
git -C CosyVoice checkout --detach "$COSYVOICE_COMMIT"
if git -C CosyVoice submodule status --recursive | grep -q '^-'; then
  retry_network git -c http.version=HTTP/1.1 -C CosyVoice submodule update --init --recursive
fi
# openai-whisper's pinned source distribution still imports pkg_resources in
# its build hook; use the compatible, explicitly pinned global build tooling.
python -m pip install --no-build-isolation -r requirements-cosyvoice-runtime.txt

mkdir -p models
if [[ ! -f models/CosyVoice2-0.5B/campplus.onnx ]]; then
  retry_network modelscope download --model iic/CosyVoice2-0.5B --revision "$COSYVOICE_MODEL_REVISION" --local_dir models/CosyVoice2-0.5B
fi
MODEL_DIR="$PROJECT_DIR/models/CosyVoice2-0.5B"
python - "$MODEL_DIR" "$COSYVOICE_MODEL_REVISION" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

model_dir = Path(sys.argv[1])
metadata = {
    "model_id": "iic/CosyVoice2-0.5B",
    "requested_revision": sys.argv[2],
    "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
    "reproducibility": "Exact file contents are locked by MODEL_SHA256SUMS.txt",
}
(model_dir / "MODEL_SNAPSHOT.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
(
  cd "$MODEL_DIR"
  find . -type f \
    ! -name 'MODEL_SHA256SUMS.txt' \
    -print0 | sort -z | xargs -0 sha256sum > MODEL_SHA256SUMS.txt
)

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m src.preflight --stage prepare --project-root "$PROJECT_DIR" --minimum-free-gb 20
