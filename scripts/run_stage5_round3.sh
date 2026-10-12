#!/usr/bin/env bash
# Stage-5 round-3 continuation of the validated 320-question cohort.
# Run: bash scripts/run_stage5_round3.sh [posterior-gv|posterior-g|posterior-g-ess|hard|soft]
# This deliberately refuses to overwrite ANY existing round-3 arm artifacts.
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-posterior-gv}"
ROOT="${ROOT:-/content/drive/MyDrive/gv-smc-artifacts/protocol-v2}"
MODEL="google/gemma-2-2b-it"
SEED=2026

case "$MODE" in
  posterior-g|posterior-gv|posterior-g-ess|hard|soft) ;;
  *) echo "Unsupported arm: $MODE" >&2; exit 2 ;;
esac

# Round 3 begins from each arm's COMPLETE round-2 adapter,
# not from the historical single-round adapter or another arm.
PREV="$ROOT/iterative-posterior-v1/seed$SEED/$MODE/round2/checkpoint"
PREV_EVAL="$ROOT/iterative-posterior-v1/seed$SEED/$MODE/round2/development_gv_n512.jsonl"
PREV_EVAL_META="${PREV_EVAL%.jsonl}.summary.json"
DIR="$ROOT/iterative-posterior-v1/seed$SEED/$MODE/round3"
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
         "$PREV_EVAL" "$PREV_EVAL_META" \
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
# Verify that the parent checkpoint and 512-question evaluation truly
# correspond to the requested mode and the immediately preceding round.
python - "$PREV/gvsmc_training_summary.json" "$PREV_EVAL_META" "$MODE" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
train_path, eval_path, mode, root = sys.argv[1:]
train = json.loads(Path(train_path).read_text())
eval_meta = json.loads(Path(eval_path).read_text())
expected_data = Path(root) / "iterative-posterior-v1" / "seed2026" / mode / "round2" / "train_frozen_weights_n320.jsonl"
expected_questions = Path(root) / "splits" / "development.jsonl"
if train.get("mode") != mode or train.get("model") != "google/gemma-2-2b-it":
    raise ValueError("Round-2 training metadata does not match arm/model")
if Path(train.get("data","")).resolve() != expected_data.resolve():
    raise ValueError("Round-2 checkpoint has unexpected source data")
if train.get("seed") != 2026 or train.get("optimizer_updates") != 82:
    raise ValueError("Round-2 checkpoint training budget or seed disagrees")
if not (
    eval_meta.get("num_questions") == 512
    and Path(eval_meta.get("question_file","")).resolve() == expected_questions.resolve()
    and Path(eval_meta.get("adapter","")).resolve() == Path(train_path).parent.resolve()
):
    raise ValueError("Round-2 checkpoint/evaluation provenance mismatch")
print(f"Verified round-2 checkpoint and development evaluation: {mode}", flush=True)
PY

if [ -e "$DIR" ]; then
  echo "Round-3 output already exists; will not overwrite: $DIR" >&2
  exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime required for Stage-5 CFT")
print("Stage-5 GPU:", torch.cuda.get_device_name(0), flush=True)
PY

mkdir -p "$DIR"
echo "Stage 5 round 3: $MODE"
echo "Generator + validator checkpoint: $PREV"
echo "New round output: $DIR"

# 1. Generate new G->V trajectories under the PREVIOUS round model.
python scripts/run_triviaqa_gv.py \
  --model "$MODEL" --adapter "$PREV" --device cuda \
  --question-file "$CFT_QUESTIONS" --limit 320 \
  --generation-mode greedy \
  --generation-seed 33003 --order-seed 34003 \
  --validator-sample-seed 35003 \
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

echo "Stage-5 round-3 $MODE finished. Artifacts preserved under: $DIR"
