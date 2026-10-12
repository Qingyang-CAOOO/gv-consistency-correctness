"""Stage-5 iterative round-3 driver structural and shell syntax safeguards.

These checks do not load any HF model, read Drive, or execute GPU jobs.
"""
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_stage5_round3.sh"


def test_stage5_round3_script_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_stage5_round3_uses_matching_prior_round_adapter_for_both_measurements():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'PREV="$ROOT/iterative-posterior-v1/seed$SEED/$MODE/round2/checkpoint"' in text
    assert 'DIR="$ROOT/iterative-posterior-v1/seed$SEED/$MODE/round3"' in text
    assert '--model "$MODEL" --adapter "$PREV" --device cuda' in text
    assert '--model "$MODEL" --adapter "$PREV"' in text
    assert text.count('--adapter "$PREV"') == 2
    assert '--model-dir "$CALIBRATOR"' in text
    assert '--expected-question-file "$CFT_QUESTIONS"' in text
    assert '--question-file "$DEV_QUESTIONS" --limit 512' in text
    assert "33003" in text and "34003" in text and "35003" in text


def test_stage5_round3_protects_outputs_and_official_final_holdout():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'if [ -e "$DIR" ]; then' in text
    assert "refusing to overwrite" in text.lower() or "will not overwrite" in text.lower()
    assert '"$PREV/gvsmc_training_summary.json"' in text
    assert '"$PREV_EVAL_META"' in text
    assert '"$ROOT/splits/development.jsonl"' in text
    assert "official_final_holdout" not in text
    assert "--overwrite" not in text and "--force" not in text
