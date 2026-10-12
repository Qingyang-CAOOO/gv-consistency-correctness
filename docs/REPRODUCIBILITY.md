# Reproducibility guide

This guide supplements the **finalized README** (kept unchanged). It distinguishes
three levels of reproduction and lists the saved artifacts needed for each.
No model checkpoint, per-question experiment dataset, or final holdout is
redistributed by this source-code repository.

## 1. Code and unit tests (CPU-capable)

Use Python >=3.10. In a Linux/Colab environment, preserve the installed,
CUDA-compatible PyTorch build rather than overriding it with an unrelated
wheel.

~~~bash
python -m pip install -r requirements/colab.txt
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke_test.py
~~~

The frozen experimental repository passed **80 tests** in the original Colab
environment on October 11, 2026. That fact is provenance; a new checkout
must be tested separately. The smoke test checks only weighting primitives;
it does **not** prove that model inference, data acquisition, or all training
runs can be reproduced.

## 2. Small generator–validator run (requires GPU and data)

The runnable end-to-end entry point is
[`scripts/run_triviaqa_gv.py`](../scripts/run_triviaqa_gv.py).
It requires the third-party [RankAlign](https://github.com/juand-r/rankalign)
TriviaQA positive/negative JSONL source under
`external/rankalign/data/dataset/triviaqa_pn_test.jsonl`.
The upstream project and its data are **not** vendored here.

~~~bash
bash scripts/bootstrap_rankalign.sh
git -C external/rankalign rev-parse HEAD
python scripts/run_triviaqa_gv.py \
  --model google/gemma-2-2b-it --device cuda \
  --split test --limit 16 --generation-mode greedy \
  --data-seed 42 --generation-seed 43 --order-seed 44 \
  --validator-sample-seed 45 --paired-order-diagnostic \
  --output outputs/gv_pilot_n16.jsonl
~~~

Use an accessible Hugging Face model checkpoint and corresponding token if
required. Run this as a **fresh exploratory pilot**, not as reproduction of
the original frozen 512-question protocol. The RankAlign bootstrap used in
the earlier experiments cloned its then-current default branch. The exact
upstream commit was not recorded in the checked-in source documentation.
Thus cloning its current head alone does **not** guarantee byte-identical
historical data. Supply a known upstream `RANKALIGN_REF` commit when
available (the bootstrap supports this optional parameter).

## 3. Offline retrospective reproduction (original saved artifacts required)

All original controlled results were computed from model-generated pairs
and *frozen*, question-disjoint partition manifests. These artifacts live
outside Git. To reproduce their numerical summaries, obtain authorized
copies from the original experiment and restore the directory layout under
an arbitrary local root, for example:

~~~text
protocol-v2/
  official_confirmation_features_n512.jsonl
  posterior_base_v2_confirmation/
    frozen_confirmation_predictions.jsonl
    summary.json
  single-round/
    cft_train_gv_n320.jsonl
    cft_train_posterior_weights_n320.jsonl
    cft_train_posterior_weights_n320.summary.json
    eval/
      ... five arm-specific round-1 development JSONL files ...
  iterative-posterior-v1/
    seed2026/
      <arm>/round2/{train_gv_n320.jsonl,train_frozen_weights_n320.jsonl,
                     train_frozen_weights_n320.summary.json,
                     development_gv_n512.jsonl}
      <arm>/round3/{same core files}
  splits/
    cft_train.jsonl
    development.jsonl
    calibration_train.jsonl
    calibration_val.jsonl
    official_confirmation.jsonl
    ... other original partition manifests ...
~~~

Here `<arm>` means one of `hard`, `soft`, `posterior-g`,
`posterior-g-ess`, `posterior-gv`. The listed tree is a **schematic subset**;
the original program and audit scripts enforce additional metadata,
provenance hashes, and complete question alignment.

The retrospective false-retention audit reads exactly the saved confirmation
population (512 questions, 394 parsed, 177 EM-grounded) and all five Stage-5
training branches. It will raise if these frozen source counts or posterior
provenance checks differ; it is **not** a generic audit for an arbitrary new
dataset.

~~~bash
ROOT=/path/to/protocol-v2
python scripts/audit_false_retention.py \
  --root "$ROOT" --bootstrap 3000 --seed 2026 \
  --output outputs/false_retention_audit.json

python scripts/audit_stage5_dynamics.py \
  --root "$ROOT" --bootstrap 5000 --seed 2026 \
  --output outputs/stage5_dynamics_audit.json
~~~

Audit scripts refuse to overwrite an existing output. Use new output paths.
See [false-retention findings](stage5_false_retention_posthoc.md) and
[Stage-5 feedback protocol](stage5_iterative_cft_v1.md) for population-level
interpretation and precise artifact names. The previously inspected
512-question development set and the 512-question confirmation set have
different roles; neither is the **sealed 1,024-question final holdout**.

## 4. Re-running training (GPU, all preceding artifacts)

The original CFT pipeline is **not a single end-to-end command starting from
a clean repository**. Its component programs are retained for faithful
reconstruction of the completed experiment:

1. `run_triviaqa_gv.py`: generate G->V trajectories with the same checkpoint
   for the generator and validator.
2. `score_gv_generator_preferences.py`: score candidates using that same
   checkpoint.
3. `apply_frozen_cft_posteriors.py`: apply the frozen qG and qGV calibrators;
   check question identities and strip gold labels from training files.
4. `train_cft.py`: train a fresh base-model LoRA adapter with hard/soft/qG/
   qG-ESS/qGV weights, two epochs and 82 optimizer updates.
5. `run_triviaqa_gv.py`: evaluate on the fixed development questions.
6. `audit_five_arm_cft.py`, `audit_multiseed_cft.py` and
   `audit_stage5_dynamics.py`: evaluate matched questions and seed variation.

The supplied `run_stage5_round2_smoke.sh` and
`run_stage5_round3.sh` are **continuations**: they require the original
round-1 and round-2 checkpoints, frozen fitted calibrators, question files
and existing training outputs. Their `ROOT` default is a
historical **Colab/Drive** path, not a downloadable public resource.
Set `ROOT` to a populated local artifact directory before running.
The older `run_iterative_cft_experiment.py` is a separate hard/soft
historical workflow and must not be substituted for the final five-arm
posterior protocol. Check saved metadata and protect existing artifacts
before GPU work.

The calibrated posterior models used gold-labeled *calibration* trajectories.
Thus posterior CFT is **supervised-calibrated self-training**, not label-free
EM. The same base-calibrated predictors were reused in rounds 2 and 3
without refitting; covariate shift is a limitation.

## 5. Reporting and provenance

Every experiment should record the repository commit, upstream RankAlign
commit, model identifier/revision, partition manifest and hashes, decoding
and randomized-order seeds, prompt version, LoRA hyperparameters, GPU
device/VRAM, dependency snapshot, per-question outcomes and summary metrics.

For comparisons, keep separate:
- parse-and-grounded EM yield **over all questions**;
- grounded fraction **conditional on successful parsing**;
- greedy randomized GV agreement, soft agreement and A/B order effects;
- label-informed posterior prediction versus actual downstream learning;
- multiple training seeds versus bootstrap variation across questions.

The confirmation subgroup contrasts and iterative-development CIs are
retrospective/exploratory; they should not be presented as independent
prospective tests, and source-weight differences alone are not causal
effects of the validator.

**Privacy/publication:** never commit Hugging Face/GitHub tokens, raw Drive
artifacts, fine-tuning checkpoints, private question manifests, or sealed
holdout results into an eventual public repository. Verify third-party
dataset and model redistribution rights before publishing any artifact.
