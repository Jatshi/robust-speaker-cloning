#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning-v2}
AISHELL_ROOT=${AISHELL_ROOT:-$PROJECT_DIR/data/data_aishell}
MODEL_ROOT=${MODEL_ROOT:-$PROJECT_DIR/models/CosyVoice2-0.5B}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-$PROJECT_DIR/CosyVoice}
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

python -m src.preprocess_aishell --aishell-root "$AISHELL_ROOT" --output data/speaker_to_files.json --max-per-speaker 15
python -m src.cosyvoice_teacher --manifest data/speaker_to_files.json --cosyvoice-root "$COSYVOICE_ROOT" --model-root "$MODEL_ROOT" --output data/cosyvoice_teacher.pt
python -m src.build_feature_cache --manifest data/speaker_to_files.json --output data/feature_cache --workers 8
python -m src.train_v2 --manifest data/speaker_to_files.json --teacher-cache data/cosyvoice_teacher.pt --feature-cache data/feature_cache --output outputs/checkpoints --epochs 20 --batch-size 16 --workers 8
python -m src.evaluate_v2 --manifest data/speaker_to_files.json --checkpoint outputs/checkpoints/best.pt --model-root "$MODEL_ROOT" --cosyvoice-root "$COSYVOICE_ROOT" --output outputs/evaluation --samples-per-type 3
python -m src.package_artifacts
