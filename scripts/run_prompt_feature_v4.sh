#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/robust-speaker-cloning}
MODEL_ROOT=${MODEL_ROOT:-$PROJECT_DIR/models/CosyVoice2-0.5B}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-$PROJECT_DIR/CosyVoice}
PHASE=${1:-all}
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"
mkdir -p outputs/prompt-feature-v4/logs outputs/prompt-feature-v4/checkpoints

require_oracle_gate() {
  python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/v4/component-diagnostic/component_diagnostic.summary.json')
s=json.loads(p.read_text())
r=s['overall']['clean_features_only']
if not (r['mean_delta'] > 0 and r['win_rate'] == 1.0 and r['ci95_low'] > 0):
    raise SystemExit('clean prompt-feature Oracle gate failed')
print({'oracle_feature_delta':r['mean_delta'],'win_rate':r['win_rate'],'ci95':[r['ci95_low'],r['ci95_high']]})
PY
}

if [[ "$PHASE" == all || "$PHASE" == cache ]]; then
  require_oracle_gate
  python -m src.build_prompt_feature_cache \
    --manifest data/speaker_to_files_v4.json --cosyvoice-root "$COSYVOICE_ROOT" \
    --degradation-assets data/degradation_assets.json \
    --output data/prompt_feature_cache_v4 --workers 8 \
    2>&1 | tee outputs/prompt-feature-v4/logs/cache.log
fi
if [[ "$PHASE" == all || "$PHASE" == train ]]; then
  require_oracle_gate
  resume_args=()
  [[ -f outputs/prompt-feature-v4/checkpoints/last.pt ]] && resume_args=(--resume outputs/prompt-feature-v4/checkpoints/last.pt)
  python -m src.train_prompt_feature_restorer \
    --manifest data/speaker_to_files_v4.json --feature-cache data/prompt_feature_cache_v4 \
    --output outputs/prompt-feature-v4/checkpoints --epochs 25 --batch-size 64 --workers 4 \
    "${resume_args[@]}" 2>&1 | tee -a outputs/prompt-feature-v4/logs/train.log
fi
if [[ "$PHASE" == all || "$PHASE" == select ]]; then
  if [[ ! -f data/v4_selection_protocol.jsonl ]]; then
    python -m src.eval_protocol --manifest data/speaker_to_files_v4.json \
      --output data/v4_selection_protocol.jsonl --samples-per-type 2 \
      --speaker-split selection --seed 5026
  fi
  python -m src.evaluate_prompt_feature_v4 \
    --protocol data/v4_selection_protocol.jsonl \
    --checkpoint outputs/prompt-feature-v4/checkpoints/best.pt \
    --model-root "$MODEL_ROOT" --cosyvoice-root "$COSYVOICE_ROOT" \
    --degradation-assets data/degradation_assets.json --asset-split selection \
    --output outputs/prompt-feature-v4/selection --alphas 0,0.25,0.5,0.75,1 \
    --skip-utmos 2>&1 | tee outputs/prompt-feature-v4/logs/selection.log
  python -m src.select_v4_alpha \
    --evaluation outputs/prompt-feature-v4/selection/evaluation.csv \
    --output outputs/prompt-feature-v4/selection/selected_alpha.json
fi
if [[ "$PHASE" == all || "$PHASE" == formal ]]; then
  python - <<'PY'
import json
r=json.load(open('outputs/prompt-feature-v4/selection/selected_alpha.json'))
if not r['gate_passed']:
    raise SystemExit('selection gate failed; formal test intentionally skipped')
PY
  alpha=$(python -c 'import json; print(json.load(open("outputs/prompt-feature-v4/selection/selected_alpha.json"))["selected_alpha"])')
  python -m src.evaluate_prompt_feature_v4 \
    --protocol data/eval_protocol.jsonl \
    --checkpoint outputs/prompt-feature-v4/checkpoints/best.pt \
    --model-root "$MODEL_ROOT" --cosyvoice-root "$COSYVOICE_ROOT" \
    --degradation-assets data/degradation_assets.json --asset-split test \
    --output outputs/prompt-feature-v4/formal --alphas "$alpha" --formal --resume \
    2>&1 | tee -a outputs/prompt-feature-v4/logs/formal.log
fi
