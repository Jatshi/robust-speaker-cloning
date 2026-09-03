#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-$PROJECT_DIR/data/downloads}
mkdir -p "$DOWNLOAD_DIR" "$PROJECT_DIR/data"
AISHELL_PUBLIC_ARCHIVE=${AISHELL_PUBLIC_ARCHIVE:-/root/autodl-pub/Aishell/data_aishell.gz}
AISHELL_PUBLIC_RESOURCE=${AISHELL_PUBLIC_RESOURCE:-/root/autodl-pub/Aishell/resource_aishell.gz}

download() {
  local url=$1 output=$2
  [[ -f "$output" ]] || aria2c -x 16 -s 16 -k 1M --continue=true --file-allocation=none -d "$(dirname "$output")" -o "$(basename "$output")" "$url"
}

if [[ -r "$AISHELL_PUBLIC_ARCHIVE" ]]; then
  AISHELL_ARCHIVE=$AISHELL_PUBLIC_ARCHIVE
else
  AISHELL_ARCHIVE=$DOWNLOAD_DIR/data_aishell.tgz
  download https://www.openslr.org/resources/33/data_aishell.tgz "$AISHELL_ARCHIVE"
fi

# The AutoDL public archive is local and read-only. Extract it in parallel with
# the two network downloads; never copy the 15 GB archive to the paid disk.
aishell_pid=
if [[ ! -d "$PROJECT_DIR/data/data_aishell" ]]; then
  tar -xzf "$AISHELL_ARCHIVE" -C "$PROJECT_DIR/data" &
  aishell_pid=$!
fi
download https://www.openslr.org/resources/17/musan.tar.gz "$DOWNLOAD_DIR/musan.tar.gz"
download https://www.openslr.org/resources/28/rirs_noises.zip "$DOWNLOAD_DIR/rirs_noises.zip"

[[ -z "$aishell_pid" ]] || wait "$aishell_pid"
if [[ -r "$AISHELL_PUBLIC_RESOURCE" && ! -d "$PROJECT_DIR/data/resource_aishell" ]]; then
  tar -xzf "$AISHELL_PUBLIC_RESOURCE" -C "$PROJECT_DIR/data"
fi
# AISHELL distributes speaker audio as nested S*.tar.gz archives.
if ! find "$PROJECT_DIR/data/data_aishell/wav" -mindepth 3 -name '*.wav' -print -quit | grep -q .; then
  find "$PROJECT_DIR/data/data_aishell/wav" -maxdepth 1 -name '*.tar.gz' -print0 | xargs -0 -r -P 8 -I{} tar -xzf "{}" -C "$PROJECT_DIR/data/data_aishell/wav"
fi
[[ -d "$PROJECT_DIR/data/musan" ]] || tar -xzf "$DOWNLOAD_DIR/musan.tar.gz" -C "$PROJECT_DIR/data"
[[ -d "$PROJECT_DIR/data/RIRS_NOISES" ]] || unzip -q "$DOWNLOAD_DIR/rirs_noises.zip" -d "$PROJECT_DIR/data"

python -m src.prepare_degradation_assets \
  --noise-root "$PROJECT_DIR/data/musan/noise" \
  --speech-root "$PROJECT_DIR/data/musan/speech" \
  --music-root "$PROJECT_DIR/data/musan/music" \
  --rir-root "$PROJECT_DIR/data/RIRS_NOISES" \
  --output "$PROJECT_DIR/data/degradation_assets.json"
