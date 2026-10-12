#!/usr/bin/env bash
# Stage-5 round-2 smoke test on the frozen protocol-v2 320-question cohort.
# Run: bash scripts/run_stage5_round2_smoke.sh [posterior-gv|posterior-g|posterior-g-ess|hard|soft]
# This deliberately refuses to overwrite ANY existing round-2 arm artifacts.
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-posterior-gv}"
ROOT="${ROOT:-/content/drive/MyDrive/gv-smc-artifacts/protocol-v2}"
MODEL="google/gemma-2-2b-it"
SEED=2026

case "$MODE" in
  posterior-g|posterior-gv|posterior-g-ess)
    PREV="$ROOT/single-round/checkpoints/$MODE"
    ;;
  hard|soft)
    PREV="$ROOT/single-round/checkpoints/${MODE}-matched-n320-seed2026"
    ;;
  *)
    echo "Unsupported arm: $MODE" >&2
    exit 2
    ;;
esac

DIR="$ROOT/iterative-posterior-v1/seed$SEED/$MODE/round2"
CFT_QUESTIONS="$ROOT/splits/cft_train.jsonl"
DEV_QUESTIONS="$ROOT/splits/development.jsonl"
CALIBRATOR="$ROOT/posterior_base_v2_pooled"
RAW="$DIR/train_gv_n320.jsonl"
FEATURES="$DIR/train_features_n320.jsonl"
WEIGHTS="$DIR/train_frozen_weights_n320.jsonl"
CKPT="$DIR/checkpoint"
EVAL="$DIR/development_gv_n512.jsonl"

# Check prerequisites before creating any output. In particular,
# an existing/partially completed round directory is not overwritten.
for f in "$PREV/adapter_config.json" "$PREV/gvsmc_training_summary.json" \
         "$CFT_QUESTIONS" "$DEV_QUESTIONS" \
         "$CALIBRATOR/q_g_model.joblib" "$CALIBRATOR/q_gv_model.joblib" \
         "$CALIBRATOR/summary.json" \
         "$ROOT/splits/calibration_train.jsonl" \
         "$ROOT/splits/calibration_val.jsonl" \
         "$ROOT/splits/legacy_train.jsonl" \
         "$ROOT/splits/official_confirmation.jsonl"; do
  if [ ! -f "$f" ]; then
    echo "Required input missing: $f" >&2
    exit 1
  fi
done
if [ -e "$DIR" ]; then
  echo "Round output directory already exists; will not overwrite: $DIR" >&2
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime required for Stage-5 CFT")
print("Stage-5 GPU:", torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p "$DIR"
echo "Stage 5 round 2: $MODE"
echo "Generator + validator checkpoint: $PREV"
echo "New round output: $DIR"

# 1. Generate new G->V trajectories under the PREVIOUS round model.
python scripts/run_triviaqa_gv.py \
  --model "$MODEL" --adapter "$PREV" --device cuda \
  --question-file "$CFT_QUESTIONS" --limit 320 \
  --generation-mode greedy \
  --generation-seed 33002 --order-seed 34002 \
  --validator-sample-seed 35002 \
  --paired-order-diagnostic \
  --output "$RAW"

# 2. Score generator candidate preferences under the SAME adapter.
python scripts/score_gv_generator_preferences.py \
  --data "$RAW" --output "$FEATURES" \
  --model "$MODEL" --adapter "$PREV" \
  --device cuda --require-cuda

# 3. Apply SAME FROZEN base-calibrated qG and qGV, no refitting.
# Gold/EM fields are stripped from the training output.
python scripts/apply_frozen_cft_posteriors.py \
  --data "$FEATURES" \
  --model-dir "$CALIBRATOR" \
  --expected-question-file "$CFT_QUESTIONS" \
  --forbid-question-file "$ROOT/splits/calibration_train.jsonl" \
  --forbid-question-file "$ROOT/splits/calibration_val.jsonl" \
  --forbid-question-file "$ROOT/splits/legacy_train.jsonl" \
  --forbid-question-file "$ROOT/splits/development.jsonl" \
  --forbid-question-file "$ROOT/splits/official_confirmation.jsonl" \
  --output "$WEIGHTS"

# 4. Train a FRESH base-model LoRA on the newly weighted trajectories.
python scripts/train_cft.py \
  --data "$WEIGHTS" --mode "$MODE" --model "$MODEL" \
  --output-dir "$CKPT" --epochs 2 \
  --updates-per-epoch 41 --seed "$SEED"

# 5. Evaluate only on frozen, previously inspected DEVELOPMENT questions.
python scripts/run_triviaqa_gv.py \
  --model "$MODEL" --adapter "$CKPT" --device cuda \
  --question-file "$DEV_QUESTIONS" --limit 512 \
  --generation-mode greedy \
  --generation-seed 43 --order-seed 44 \
  --validator-sample-seed 45 \
  --paired-order-diagnostic \
  --output "$EVAL"

echo "Stage-5 round-2 $MODE finished. Artifacts preserved under: $DIR"
