#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning}
MODEL_ROOT=${MODEL_ROOT:-$PROJECT_DIR/models/CosyVoice2-0.5B}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-$PROJECT_DIR/CosyVoice}
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

python -m src.candidate_checkpoints --history outputs/checkpoints/history.json --checkpoint-dir outputs/checkpoints --top-k 3 --output outputs/selection/candidates.json
python -m src.eval_protocol --manifest data/speaker_to_files.json --output data/selection_protocol.jsonl --samples-per-type 2 --speaker-split selection

while IFS= read -r line; do
  epoch=${line%%|*}; checkpoint=${line#*|}
  python -m src.evaluate_v3 --protocol data/selection_protocol.jsonl --checkpoint "$checkpoint" --model-root "$MODEL_ROOT" --cosyvoice-root "$COSYVOICE_ROOT" --degradation-assets data/degradation_assets.json --asset-split selection --output "outputs/selection/epoch-$epoch" --skip-denoiser --skip-utmos --resume </dev/null
  # The evaluator and its third-party dependencies must not inherit the
  # candidate pipe. A library reading stdin would otherwise consume the next
  # candidate line and silently skip a checkpoint.
done < <(python -c "import json; [print(f\"{r['epoch']:03d}|{r['checkpoint']}\") for r in json.load(open('outputs/selection/candidates.json'))]")

python -m src.select_checkpoint --candidates outputs/selection/candidates.json --evaluation-root outputs/selection --output-checkpoint outputs/checkpoints/selected.pt --output-report outputs/selection/selection.json
