#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning}
AISHELL_ROOT=${AISHELL_ROOT:-$PROJECT_DIR/data/data_aishell}
MODEL_ROOT=${MODEL_ROOT:-$PROJECT_DIR/models/CosyVoice2-0.5B}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-$PROJECT_DIR/CosyVoice}
PHASE=${1:-all}
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"
mkdir -p outputs/logs outputs/checkpoints

run_phase() {
  local name=$1
  shift
  if [[ -f "outputs/.done-$name" ]]; then echo "skip completed phase: $name"; return; fi
  "$@" 2>&1 | tee "outputs/logs/$name.log"
  touch "outputs/.done-$name"
}

if [[ "$PHASE" == all || "$PHASE" == manifest ]]; then
  run_phase manifest python -m src.preprocess_aishell --aishell-root "$AISHELL_ROOT" --output data/speaker_to_files.json --max-per-speaker 15
fi
if [[ "$PHASE" == all || "$PHASE" == teacher ]]; then
  run_phase teacher python -m src.cosyvoice_teacher --manifest data/speaker_to_files.json --cosyvoice-root "$COSYVOICE_ROOT" --model-root "$MODEL_ROOT" --output data/cosyvoice_teacher.pt
fi
if [[ "$PHASE" == all || "$PHASE" == cache ]]; then
  run_phase cache python -m src.build_feature_cache --manifest data/speaker_to_files.json --output data/feature_cache --workers 8 --degradation-assets data/degradation_assets.json --formal-real-degradations
fi
if [[ "$PHASE" == all || "$PHASE" == train ]]; then
  python -m src.preflight --stage train --project-root "$PROJECT_DIR" --minimum-free-gb 10
  resume_args=(); [[ -f outputs/checkpoints/last.pt ]] && resume_args=(--resume outputs/checkpoints/last.pt)
  python -m src.train_v2 --manifest data/speaker_to_files.json --teacher-cache data/cosyvoice_teacher.pt --feature-cache data/feature_cache --output outputs/checkpoints --epochs 20 --batch-size 16 --workers 8 --memory-bank-size 4096 "${resume_args[@]}" 2>&1 | tee -a outputs/logs/train.log
fi
if [[ "$PHASE" == all || "$PHASE" == protocol ]]; then
  run_phase protocol python -m src.eval_protocol --manifest data/speaker_to_files.json --output data/eval_protocol.jsonl --samples-per-type 15 --speaker-split test
fi
if [[ "$PHASE" == all || "$PHASE" == select ]]; then
  run_phase select bash scripts/select_v3.sh
fi
if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
  python -m src.preflight --stage eval --project-root "$PROJECT_DIR" --minimum-free-gb 5
  python -m src.evaluate_v3 --protocol data/eval_protocol.jsonl --checkpoint outputs/checkpoints/selected.pt --model-root "$MODEL_ROOT" --cosyvoice-root "$COSYVOICE_ROOT" --degradation-assets data/degradation_assets.json --output outputs/evaluation-v3 --resume --formal 2>&1 | tee -a outputs/logs/eval.log
fi
