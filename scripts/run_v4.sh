#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning}
AISHELL_ROOT=${AISHELL_ROOT:-$PROJECT_DIR/data/data_aishell}
MODEL_ROOT=${MODEL_ROOT:-$PROJECT_DIR/models/CosyVoice2-0.5B}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-$PROJECT_DIR/CosyVoice}
PHASE=${1:-all}
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"
mkdir -p outputs/v4/logs outputs/v4/checkpoints

run_phase() {
  local name=$1
  shift
  if [[ -f "outputs/v4/.done-$name" ]]; then
    echo "skip completed V4 phase: $name"
    return
  fi
  "$@" 2>&1 | tee "outputs/v4/logs/$name.log"
  touch "outputs/v4/.done-$name"
}

require_diagnostic_gate() {
  python - <<'PY'
import json
from pathlib import Path
path = Path("outputs/v4/diagnostic/diagnostic.summary.json")
if not path.exists() or not json.loads(path.read_text())["proceed_to_training"]:
    raise SystemExit("V4 Gate A failed or is missing; refusing to spend GPU time on training")
PY
}

if [[ "$PHASE" == all || "$PHASE" == prepare ]]; then
  bash scripts/bootstrap_autodl.sh
  bash scripts/download_public_data.sh
fi
if [[ "$PHASE" == all || "$PHASE" == manifest ]]; then
  run_phase manifest python -m src.preprocess_aishell \
    --aishell-root "$AISHELL_ROOT" --output data/speaker_to_files_v4.json --max-per-speaker 5
fi
if [[ "$PHASE" == all || "$PHASE" == diagnostic ]]; then
  run_phase diagnostic_protocol python -m src.eval_protocol \
    --manifest data/speaker_to_files_v4.json --output data/v4_diagnostic_protocol.jsonl \
    --samples-per-type 1 --speaker-split selection --seed 4026
  run_phase diagnostic python -m src.diagnose_v4_interface \
    --protocol data/v4_diagnostic_protocol.jsonl --model-root "$MODEL_ROOT" \
    --cosyvoice-root "$COSYVOICE_ROOT" --degradation-assets data/degradation_assets.json \
    --asset-split selection --output outputs/v4/diagnostic
fi
if [[ "$PHASE" == all || "$PHASE" == cache ]]; then
  require_diagnostic_gate
  run_phase embedding_cache python -m src.build_v4_embedding_cache \
    --manifest data/speaker_to_files_v4.json --cosyvoice-root "$COSYVOICE_ROOT" \
    --model-root "$MODEL_ROOT" --degradation-assets data/degradation_assets.json \
    --output data/v4_embedding_cache --asset-split train
fi
if [[ "$PHASE" == all || "$PHASE" == train ]]; then
  require_diagnostic_gate
  resume_args=()
  [[ -f outputs/v4/checkpoints/last.pt ]] && resume_args=(--resume outputs/v4/checkpoints/last.pt)
  python -m src.train_v4 --manifest data/speaker_to_files_v4.json \
    --embedding-cache data/v4_embedding_cache --output outputs/v4/checkpoints \
    --epochs 30 --batch-size 256 --workers 4 "${resume_args[@]}" \
    2>&1 | tee -a outputs/v4/logs/train.log
fi
if [[ "$PHASE" == all || "$PHASE" == select ]]; then
  require_diagnostic_gate
  run_phase selection_protocol python -m src.eval_protocol \
    --manifest data/speaker_to_files_v4.json --output data/v4_selection_protocol.jsonl \
    --samples-per-type 2 --speaker-split selection --seed 5026
  run_phase selection_eval python -m src.evaluate_v4 \
    --protocol data/v4_selection_protocol.jsonl --checkpoint outputs/v4/checkpoints/best.pt \
    --model-root "$MODEL_ROOT" --cosyvoice-root "$COSYVOICE_ROOT" \
    --degradation-assets data/degradation_assets.json --asset-split selection \
    --output outputs/v4/selection --alphas 0,0.25,0.5,0.75,1 --skip-utmos
  run_phase select_alpha python -m src.select_v4_alpha \
    --evaluation outputs/v4/selection/evaluation.csv \
    --output outputs/v4/selection/selected_alpha.json
fi
if [[ "$PHASE" == all || "$PHASE" == formal ]]; then
  python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("outputs/v4/selection/selected_alpha.json").read_text())
if not report["gate_passed"]:
    raise SystemExit("V4 selection gate failed; alpha=0 won, so formal test is intentionally skipped")
PY
  if [[ -f data/eval_protocol.jsonl ]]; then
    formal_protocol=data/eval_protocol.jsonl
    echo "reuse V3 formal protocol for a paired V3/V4 comparison: $formal_protocol"
  else
    run_phase formal_protocol python -m src.eval_protocol \
      --manifest data/speaker_to_files_v4.json --output data/v4_test_protocol.jsonl \
      --samples-per-type 15 --speaker-split test --seed 2026
    formal_protocol=data/v4_test_protocol.jsonl
  fi
  alpha=$(python -c 'import json; print(json.load(open("outputs/v4/selection/selected_alpha.json"))["selected_alpha"])')
  python -m src.evaluate_v4 --protocol "$formal_protocol" \
    --checkpoint outputs/v4/checkpoints/best.pt --model-root "$MODEL_ROOT" \
    --cosyvoice-root "$COSYVOICE_ROOT" --degradation-assets data/degradation_assets.json \
    --asset-split test --output outputs/v4/formal --alphas "$alpha" --formal --resume \
    2>&1 | tee -a outputs/v4/logs/formal.log
fi
