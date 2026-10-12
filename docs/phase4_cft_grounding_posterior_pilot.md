# Stage 3 pilot: correctness posterior for original CFT pairs

User-side Colab results, October 11 2026. Frozen disjoint question partitions:
256 calibration-train (197 parsed, 75 grounded) and 128 calibration-val
(94 parsed, 38 grounded). The generated pair is the **original**
designated-correct/designated-misleading GV trajectory, not the alternative
multi-hypothesis ranking task. Parsing success is ~76.95% and ~73.44%.

## Held-out pilot result (94 parsed validation questions)

| quantity | q_G | q_GV | improvement |
|---|---:|---:|---:|
| NLL (lower is better) | .48008 | .46617 | -.01391 |
| Brier (lower is better) | .16068 | .15577 | -.00491 |
| AUROC | .83271 | .85150 | +.01880 |
| Average precision | .79455 | .81043 | +.01588 |
| Grounded-pair selected precision | .58493 | .59580 | +.01087 |
| Fractional ESS | .71292 | .70080 | -.01211 |

Grounded-pair base rate on validation: .40426.
The uncalibrated order-marginalized GV consistency weight yielded weighted
grounded precision .47000 and fractional ESS .84251. This is **not a fair
unmatched-supervision contest** against calibrated predictors, which had
access to labeled calibration data.

Question bootstrap (2,000 resamples, q_GV minus q_G):
- NLL: -.01391, 95% CI [-.03482, +.00886].
- Brier: -.00491, 95% CI [-.01431, +.00513].
- Weighted grounded precision: +.01087, CI [-.00319, +.02513].
- AUROC: +.01880, CI [-.01703, +.05344].

All intervals cross zero. The pilot demonstrates feasibility of a strong
supervised q_G and suggests a small positive incremental V signal, but
**does not establish** incremental predictive benefit on held-out data.
Do not start expensive CFT-III training as a confirmed improvement.

The unconditional mean of soft GV consistency on validation was ~.782,
versus grounded-pair prevalence .404. Its NLL ~2.40 is **not** a calibration
failure relative to its own target: GV weights represent the probability
of self-consistency, *not* the probability of factual pair validity. Treat
the result only as evidence that the two targets are different.

## Diagnostic and low-cost data-enlargement experiments

First run a CPU-only reliability and state-level error audit using the saved
validation predictions and the 128-question validation feature file:

    python scripts/diagnose_cft_grounding_posterior.py \
      --validation-features "$ROOT/calibration_val_features_n128.jsonl" \
      --predictions "$ROOT/posterior_base_v1/calibration_validation_predictions.jsonl" \
      --output "$ROOT/posterior_base_v1/diagnostics.json" --show-examples 6

This reports the four EM-proxy pair states, mean q_G and q_GV by state, and
five-bin reliability. The 94-question validation has been examined already;
this audit is descriptive, not independent verification.

The 401 parsed historical base-model training pairs have already been
scored in both A/B orders. To test whether additional supervised training
data help the conditional model *without rerunning generator or verifier*
use one short generator-preference scoring run and merge feature files:

    python scripts/score_gv_generator_preferences.py \
      --data "$ROOT/base_train_both_orders_n512.jsonl" \
      --output "$ROOT/legacy_base_train_features_n512.jsonl" \
      --model google/gemma-2-2b-it --device cuda --require-cuda

    python scripts/merge_gv_calibration_features.py \
      --source "$ROOT/legacy_base_train_features_n512.jsonl" \
      --source "$ROOT/calibration_train_features_n256.jsonl" \
      --exclude "$ROOT/splits/cft_train.jsonl" \
      --exclude "$ROOT/splits/calibration_val.jsonl" \
      --output "$ROOT/calibration_train_pooled_n768.jsonl"

The pooled dataset contains 768 *questions* with approximately 598
parseable trajectories (401+197), subject to source validation. Both source
groups are independent of protocol-v2 CFT-train and calibration-val at the
question level. Historical base CFT data are NOT new independent calibration
questions; the stronger q fit is an exploratory data-size ablation, not a
fresh confirmatory result.

    python scripts/fit_cft_grounding_posterior.py \
      --train "$ROOT/calibration_train_pooled_n768.jsonl" \
      --validation "$ROOT/calibration_val_features_n128.jsonl" \
      --output-dir "$ROOT/posterior_base_v2_pooled" \
      --bootstrap 2000 --seed 2026

Do not hyperparameter-search on the repeatedly inspected 94-question
validation split, and do not compare it to the past fit as an independent
holdout confirmation.

## Main upcoming experiment

1. Get a sufficiently large *new* labeled calibration and evaluation
   corpus for QA (official TriviaQA or independent QA corpus, carefully
   deduplicated against all RankAlign question splits and historical
   experiments). Reserve a fresh evaluation block *before* fitting.
2. Test whether q_GV consistently improves **proper scoring rules** and
   grounded-pair selection over the supervision-matched q_G across
   domains/checkpoints.
3. Audit manual samples of the 00 and 10 exact-match states to understand
   alias and semantic false negatives.
4. Only then run CFT-III with frozen q_GV and q_G weighting from identical
   generated trajectories. Maintain hard/soft and order-marginalized
   single-round controls and matched compute.

Important: direct-answer likelihood scoring adds a second generator
measurement distinct from the paired-answer generation likelihood. The
q_GV comparison properly conditions on that feature set, not on a
fully observed model distribution over all possible answers.


## Pooled supervised pilot (October 11, 2026)

The 401 parsed historical base-model train pairs and 197 parsed new
calibration-training pairs were merged with overlap safeguards: **598 parsed
out of 768 training questions**. The *same previously inspected* 94 parsed
questions were used for exploratory validation. All 46 Colab tests passed
prior to adding the new posterior ESS / official holdout checks.

| Metric (94 parsed validation) | q_G | q_GV |
|---|---:|---:|
| NLL | .4615467 | .4487512 |
| Brier | .1546438 | .1503499 |
| AUROC | .8435150 | .8547932 |
| Weighted grounded precision | .5891828 | .6093217 |
| ESS fraction | .7190620 | .6839510 |

Paired question bootstrap (2,000 draws):
- GV minus G NLL: -.0127955, CI [-.0516155, +.0306032]
- Brier: -.0042939, CI [-.0214258, +.0145897]
- Weighted grounded precision: +.0201390, CI [-.0056353, +.0451613]
- AUROC: +.0112782, CI [-.0371241, +.0559563]

Both models improved point predictions after enlarging calibration training,
but additional validator benefit is **not established**. q_GV also has more
concentrated selection weights than q_G. The next cheap diagnostic matches
the *weights' ESS*, **not** calibration, by applying an exponent to q_G:

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_posterior_ess_matched.py \
      --predictions "$ROOT/posterior_base_v2_pooled/calibration_validation_predictions.jsonl" \
      --output "$ROOT/posterior_base_v2_pooled/ess_matched_audit.json" \
      --bootstrap 2000 --seed 2026

This reports q_GV selection precision minus q_G's ESS-matched precision
with question-bootstrap CI, on the same 94 inspected validation questions.
Exponentiated weights are NOT calibrated posterior probabilities.

## Fresh official TriviaQA confirmation and final holdout

The original RankAlign test file contains only 531 questions (512 already
inspected). For fresh **question-level** confirmation use official TriviaQA
*rc.nocontext validation* Parquet; this is the same benchmark but a much
larger source. Its publicly distributed Parquet file is about 7.34 MB.
The pinned source is mandarjoshi/trivia_qa, revision
0f7faf33a3908546c6fd5b73a660e0f8ff173c2f,
filename rc.nocontext/validation-00000-of-00001.parquet.

The manifest builder uses PyArrow to read just question, question_id,
and answer; it excludes **every RankAlign train and test question** by
normalized question identity before dividing eligible questions. This does
NOT guarantee that differently worded paraphrases / semantic duplicates or
pretraining exposure have been excluded.

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/make_official_triviaqa_evaluation_manifest.py \
      --rankalign-root external/rankalign \
      --output-dir "$ROOT/splits" \
      --confirm-size 512 --final-size 1024 --seed 312 \
      --inspect-only

Only if the source capacity and overlap summary look acceptable, re-run
without --inspect-only. This creates:
- splits/official_confirmation.jsonl (512 previously untouched questions):
  **one prospective evaluation** of locked q_G/q_GV, without re-fitting.
- splits/official_final_holdout.jsonl (1024 questions): **do not run** until
  the training method, evaluation criteria, and hyperparameters are frozen.
- splits/official_eval_manifest.json: immutable source revision, question
  hashes, exact overlap exclusions.

To evaluate the **frozen** pooled models, the confirmation subset must first
be scored with precisely the same two GV measurements and generator
likelihood features as calibration. This requires GPU inference:

    python scripts/run_triviaqa_gv.py \
      --model google/gemma-2-2b-it --device cuda \
      --question-file "$ROOT/splits/official_confirmation.jsonl" \
      --limit 512 --generation-seed 29001 --order-seed 29002 \
      --validator-sample-seed 29003 --paired-order-diagnostic \
      --output "$ROOT/official_confirmation_gv_n512.jsonl"

    python scripts/score_gv_generator_preferences.py \
      --data "$ROOT/official_confirmation_gv_n512.jsonl" \
      --output "$ROOT/official_confirmation_features_n512.jsonl" \
      --model google/gemma-2-2b-it --device cuda --require-cuda

    python scripts/evaluate_frozen_cft_grounding_posterior.py \
      --model-dir "$ROOT/posterior_base_v2_pooled" \
      --data "$ROOT/official_confirmation_features_n512.jsonl" \
      --output-dir "$ROOT/posterior_base_v2_confirmation" \
      --bootstrap 2000 --seed 2026

The evaluator loads saved q_G and q_GV models; it cannot refit them. Its
output is a first prospective test of the held-out correctness posterior
under the base-model generator, not a final claim of CFT-III benefit.
Treat this confirmation as expended for model selection after examining it;
preserve official_final_holdout for the eventual post-CFT comparison.

## Follow-up gate

Do NOT treat the previous bootstrap non-significance as proof that the
validator adds no signal. Equally, do NOT declare CFT-III superior because
its q_GV precision point estimate is higher than q_G on the repeated 94
validation questions. First inspect ESS-matched selection and the prospective
confirmation outcome.

The CFT-III objective remains an EM-**like**, supervised-calibrated
weighted MLE procedure, NOT standard exact EM. A q_G-only CFT arm is
essential as a supervision-matched control. No posterior-weighted LoRA run
is yet implemented.


## October 11, 2026 decision: ESS-adjusted validator effect

The ESS-matched diagnostic on the repeatedly inspected 94 parsed validation
questions was:

| Selection weighting | Grounded-pair precision | ESS fraction |
|---|---:|---:|
| q_G | .5891828 | .7190620 |
| q_G ^ 1.1129997 | .6053151 | .6839510 |
| q_GV | .6093217 | .6839510 |

The raw q_GV - q_G selection advantage +.020139 was reduced to
q_GV - q_G^(ESS-matched) **+.004007**, paired bootstrap 95% interval
[-.018887, +.024988]. Both are non-significant in this small reused
validation set. ESS matching is not a calibration operation.

The official TriviaQA validation-source inspection found 9,430 eligible
unique question identities after RankAlign overlap exclusion, including
7,550 duplicate source rows. To improve exact-match label quality, the
manifest builder now **unions nonconflicting aliases** across duplicate
question rows and **excludes conflicting official answer IDs / canonical
values**. The user should pull the new code and rerun --inspect-only.
The question eligibility count or manifest hashes can change under this
more conservative policy. Neither official split has been generated yet.

### Locked prospective confirmation protocol

Before examining **any** official-confirmation correctness labels:

- Freeze the already fitted pooled q_G and q_GV model files; do not retrain,
  choose new features or change regularization between the fit and the
  confirmation result. Record model SHA256 checksums if convenient.
- Generate one original CFT-style greedy answer pair per question using
  google/gemma-2-2b-it and the previously specified seeds. Compute both A/B
  positions, and the same direct-answer conditional likelihood features.
- **Primary endpoint:** question-level paired NLL difference
  (q_GV - q_G). Report point estimate and 95% paired bootstrap CI.
- Secondary endpoints: paired Brier difference, AUROC, conditional grounded-
  pair selection precision, and **ESS-matched** selection precision,
  alongside calibration/reliability and parse success.
- The 512-question confirmation is a *single prospective check*. A confidence
  interval excluding zero on the primary difference would support a
  positive incremental predictive validator signal within this benchmark.
  If it includes zero, record the result as inconclusive, not "no effect".
- Do not adjust the calibrated posterior on this set and then label it
  independent confirmation. Future improved methods need new untouched
  questions or cross-fitted results with clear development/confirmation roles.
- The official 1024-question final_holdout stays unused until a matched
  CFT-III training protocol and all model choices are frozen.

The label target remains a two-answer *exact-match grounded pair*, not
semantic truth. The 512-question confirmation is within-domain and may
contain semantic paraphrase overlap or pretraining exposure. This does
not establish downstream CFT-III gains.

The downstream CFT-III training comparison must include a supervision-
matched q_G control and ESS-aware ablations. A q_GV advantage in prediction
alone does not prove better MLE fine-tuning outcomes.
