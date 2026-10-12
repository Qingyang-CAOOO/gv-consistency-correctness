# Stage 5 v1 — iterative posterior-weighted GV CFT

Status: **round-2 q_GV smoke run complete (October 11, 2026); remaining controls not yet run**. Stage 4 completed with three LoRA training seeds and a small nonrobust downstream GV-posterior effect. Stage 5 tests the *dynamics* of repeated generation -> validation -> weighting -> fine-tuning, not a rescue of Stage-4 significance.

## Experimental question

Does validator-derived correctness information accumulate useful factual grounding across CFT iterations, or merely increase GV agreement and selection concentration? Predeclared primary outcome: all-question parse-and-grounded exact-match yield on the 512 repeatedly analyzed development questions. Report agreement, parsing, factual states, enrichment and ESS as distinct secondary outcomes.

## Iteration semantics

Use the original paper-style **reset-to-base** training semantics: round-t trajectories are generated and scored by the previous round's adapter; the new round-t LoRA is trained from a *fresh base model* using only those new trajectories and weights. The trainer already initializes a fresh base model, rather than continuing the previous LoRA state.

  theta_(m,t-1) -> generate D_(m,t) -> score/weight Phi_(m,t) -> fit fresh theta_(m,t)

This is adaptive learned-proposal filtering in a Feynman–Kac analogy, NOT textbook SMC: there are no persistent particles, no multiplicative weights across the same particles, and no ESS-triggered resampling at this stage.

## Methods and fixed initial state

Use the already trained **seed-2026 Stage-4 checkpoints as round 1**, all trained on exactly the same frozen 320-question CFT train source:

| Arm | Round-1 checkpoint relative to ROOT |
|---|---|
| posterior-g | single-round/checkpoints/posterior-g |
| posterior-gv | single-round/checkpoints/posterior-gv |
| posterior-g-ess | single-round/checkpoints/posterior-g-ess |
| hard | single-round/checkpoints/hard-matched-n320-seed2026 |
| soft | single-round/checkpoints/soft-matched-n320-seed2026 |

For rounds 2 and 3, compare these five arms on the same frozen training-question file (splits/cft_train.jsonl, 320 questions) and development-question file (splits/development.jsonl, 512 questions), preserving the train/dev separation. The primary comparison is GV posterior versus generator posterior. ESS-matched G controls weighting concentration, hard and soft are original consistency-filtering controls.

Hold fixed: Gemma-2-2b-it, LoRA r8/alpha16/dropout .05, seed 2026, two epochs, 41 optimizer updates per epoch, greedy generation, paired A/B validator scoring. Matching optimizer updates does not match selected sequence counts or FLOPs for hard filtering.

Training question-order and sampling seeds for round t: generation 33000+t, validator A/B order 34000+t, validator sample 35000+t, shared between all arms. Development evaluation seeds remain (43,44,45), as in Stage 4.

## Mandatory round-2 and round-3 pipeline for EACH arm

1. From the previous round adapter, generate fresh original paired-answer GV trajectories with scripts/run_triviaqa_gv.py and the frozen --question-file splits/cft_train.jsonl. Request both-order diagnostics and save new arm/round-specific raw JSONL.
2. Score EACH newly generated answer pair using scripts/score_gv_generator_preferences.py **with that same previous-round adapter**, never accidentally score fine-tuned samples against the untouched base model.
3. Run scripts/apply_frozen_cft_posteriors.py with model-dir ROOT/posterior_base_v2_pooled and expected-question-file ROOT/splits/cft_train.jsonl; forbid known calibration and development splits. No posterior refit. Drop gold/EM fields in the emitted training file, recalculate qG, qGV and qG-ESS weights on the **new** population. Do not reuse round-1 weights.
4. Run scripts/train_cft.py --mode ARM --data GOLD_BLIND_NEW_ROUND_DATA --epochs 2 --updates-per-epoch 41 --seed 2026, writing into a new checkpoint folder. train_cft starts afresh from the same base LM. Hard filtering can retain fewer examples.
5. Evaluate the newly trained checkpoint with scripts/run_triviaqa_gv.py using ROOT/splits/development.jsonl, 512 questions, paired order diagnostics and decoding seeds 43,44,45. Audit the exact same development question identities across methods and rounds.

Write all training trajectories, candidate-likelihood scores, posteriors, checkpoint summaries, evaluations and provenance to distinct ROOT/iterative-posterior-v1/seed2026/MODE/roundN paths. Do not overwrite existing files; preserve first-round checkpoints and all intermediate artifacts.

## Critical calibration and interpretation limitations

The frozen qG/qGV posterior estimators were fitted on **base-model** trajectories. As the current adapter changes, their predictions face *covariate shift* and need not remain calibrated. Track their weight distributions, ESS and truth enrichment at every round; do not quietly refit using round-2/3 CFT truth labels. Training-trajectory gold EM is permitted solely in separate exploratory diagnostics, never in training-feature selection.

After round 1 the arm populations (and therefore their raw trajectories) diverge. Each arm's ESS matching is within its own population; it does NOT mean the posterior-GV and posterior-G-ESS branches share identical effective populations in later rounds. Do not claim perfect causal identification from matched random seeds.

The previously executed three training seeds show meaningful optimizer-seed variation, and 512 development questions have been analyzed repeatedly. The seed-2026 iterative run is a **feasibility/dynamics pilot**. Do not pool seed x question outcomes as independent. Predeclare any Stage-5 training-seed replication policy before final evaluation.

## Existing code is not directly runnable for the new Stage 5

The repo already contains scripts/run_iterative_cft_experiment.py for historical hard/soft iterations (an earlier experimental direction; its separate planning note was omitted from the publication-focused repository). That existing driver uses historical --split train/test and Phase-1 checkpoints and **does not support** the frozen qG/qGV posterior pipeline, official protocol-v2 question manifests or adapter-aware generator-preference scoring. It must NOT be run unchanged on the new experimental population.

The underlying individual scripts already support the correct round-by-round steps. A **new safety-checked Stage-5 driver** should compose those tools with strict arm/round checkpoint provenance, collision protection and resume logic before the expensive GPU batch is launched. First run a small dry-run/preflight and a smoke-test arm, then proceed through rounds 2/3.

## Stopping and outcome analysis

Record at every round parse fraction, four factual states, all-question grounded-pair rate (primary), intended-correct EM, hard/soft agreement, validator order gap, truth enrichment, train ESS, selected trajectories, optimizer updates and elapsed time. Compare all arms against their OWN round-1 baseline and show GV-minus-G across rounds, including regressions. Higher self-consistency without factual gains is NOT a success.

No repeat-seed result or final test claim should be selected after inspecting the same development questions. The official 1,024-question final holdout stays sealed until the model-selection and final inference policy is locked.


## Smoke-run implementation now available

The single-arm, round-2 guarded runner is now implemented at
scripts/run_stage5_round2_smoke.sh. It performs the 5 stages above
and refuses to overwrite a pre-existing arm/round output folder
(including partially completed runs). It also validates that the
source checkpoint summary, question partitions and frozen posterior
joblib files exist before GPU work.

Run in Colab after pulling the branch and passing unit tests:

    bash scripts/run_stage5_round2_smoke.sh posterior-gv

This is a substantial GPU job: generation over 320 questions,
adapter-aware generator-preference scoring, frozen posterior
application, fresh 82-update LoRA training and 512-question dev eval.
If it completes, repeat the same script for posterior-g and
posterior-g-ess, then hard and soft controls. These are isolated
round-2 runs: each reads its **own** matching round-1 checkpoint.
No second-round or final-holdout data should be used for initial
calibration or model selection.

The script is **not** the complete rounds-2-and-3 orchestrator;
review one round-2 output/metadata first, then add a safe general
round-3 driver that loads each arm's saved round-2 checkpoint.
All code additions still require Colab validation.


## October 11, 2026: first round-2 GV-posterior run completed

User-side A100-SXM4-40GB Colab run passed **69 tests** and all five
round-2 pipeline steps for seed 2026 / posterior-gv: 320-question
new generation from the existing round-1 adapter, scoring those answers
with that SAME adapter, applying both FROZEN base-trained correctness
posteriors gold-blind, training a fresh 82-update LoRA using q_GV weights,
and evaluating on the unchanged 512-question development cohort.

### First observed round-1 to round-2 changes

| Measure | q_GV round 1 | q_GV round 2 |
|---|---:|---:|
| parsed / 512 | 476 | 499 |
| parse-and-grounded / 512 (primary) | 193 | 210 |
| all-question grounded yield | 37.6953125% | 41.015625% |
| grounded among parsed | 40.5462185% | 42.0841683% |
| hard GV agreement among parsed | 85.084034% | 87.374747% |

The observed round-2 minus round-1 primary point difference is
**+17/512 = +3.3203125 percentage points**. Paired uncertainty has
not yet been computed; inspect the saved, aligned question-level JSONL,
not merely the summary files.

The round-2 TRAIN generation parsed 292/320 pairs (91.25%), compared
with the original round-1 CFT training pool of 238/320 (74.375%).
Training exposure grew from 476 to 584 generator+validator sequences
while the optimizer budget remained 82 updates. Thus improvements
cannot be attributed solely to weights rather than format compliance
or additional effective training data; both are part of the iterative
feedback system. The round-2 raw generation produced 129 exactly
grounded pairs among the 292 parsed training pairs (44.18%), for
diagnostics only, never training supervision.

Frozen posterior application on the *new* GV-generated population:
q_G mean .4422308, q_GV mean .4717263, q_G ESS .8120604,
q_GV ESS .8154042, ESS-matching exponent .9876866.
The original Stage-4 q_GV mean was .4123796 and q_GV ESS .729868;
these distributions are not identically distributed across rounds.
**Do not interpret an increase in mean frozen posterior as verified
accuracy or improved calibration under distribution shift.** Model
SHA256s remained identical to Stage 4; no refit or gold labels were
used to construct the training file.

The A100 run's progress-bar stages were ~7:56 generator+validator train
generation, 0:53 candidate scoring, 8:51 LoRA training and 12:15 dev
evaluation. Save all outputs in:
`$ROOT/iterative-posterior-v1/seed2026/posterior-gv/round2`.
They have already been written; **do not rerun this smoke script
for posterior-gv** because the output directory is protected.

### Cheap paired within-arm audit

Before claiming an iterative improvement, quantify paired gains and
losses on the *same* 512 questions, with descriptive question-bootstrap
confidence intervals. This is development-only and one training seed:

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_single_round_cft.py \
      --arm "gv-round1=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --arm "gv-round2=$ROOT/iterative-posterior-v1/seed2026/posterior-gv/round2/development_gv_n512.jsonl" \
      --first gv-round1 --second gv-round2 \
      --output "$ROOT/iterative-posterior-v1/seed2026/posterior-gv/paired_round2_minus_round1_audit.json" \
      --bootstrap 5000 --seed 2026

If the output already exists, read it; do not overwrite.

### Next controlled round-2 comparisons

After the paired audit, run the existing protected single-arm round-2
script individually for **posterior-g** and **posterior-g-ess**, then
**hard** and **soft**. These scripts generate each arm's own round-2
feedback data from the matching *round-1* adapter and do not reuse
the GV arm's raw data:

    bash scripts/run_stage5_round2_smoke.sh posterior-g
    bash scripts/run_stage5_round2_smoke.sh posterior-g-ess
    bash scripts/run_stage5_round2_smoke.sh hard
    bash scripts/run_stage5_round2_smoke.sh soft

Only after those four matched round-2 evaluations should we compare
all arms and decide if the planned round-3 dynamics pilot is worth
running. The official 1024-question final holdout stays sealed.

## October 11, 2026: all five Stage-5 round-2 runs completed

The user's Colab A100-SXM4-40GB run completed all five arms:
hard, soft, generator-only posterior, GV posterior and ESS-matched
generator-only posterior, using the same frozen 320-question training
cohort and 512-question development evaluation cohort. Seed 2026.
Each new adapter came from a fresh base model with two epochs and
82 optimizer updates. Generator candidate likelihoods were scored
by the *previous-round matching adapter*. Original base-trained
posterior models were applied without any refitting or gold leakage.

| mode | r1 grounded / 512 | r2 parsed / 512 | r2 grounded / 512 | round-2 all-question yield | improvement (pp) | r2 hard GV agreement among parsed |
|---|---:|---:|---:|---:|---:|---:|
| hard | 191 | 501 | 214 | 41.7969% | +4.4922 | 86.0279% |
| soft | 184 | 507 | 212 | 41.4063% | +5.4688 | 84.4181% |
| posterior-g | 186 | 483 | 202 | 39.4531% | +3.1250 | 85.7143% |
| posterior-g-ess | 191 | 497 | 205 | 40.0391% | +2.7344 | 83.9034% |
| posterior-gv | 193 | 499 | 210 | 41.0156% | +3.3203 | 87.3747% |

After round 2, GV minus G yields +8/512 = +1.5625 pp, GV
minus G ESS yields +5/512 = +0.9766 pp, GV minus hard yields
-4/512 = -0.78125 pp, GV minus soft yields -2/512 = -0.390625
pp. These are descriptive single-seed results; exact question-
paired CIs for those ROUND-2 BETWEEN-METHOD comparisons have
not yet been computed.

The earlier q_GV r1-to-r2 WITHIN-ARM question-paired bootstrap
(5000 draws) was +17/512 = +3.3203 pp in all-question grounding,
95% percentile CI [+0.9766, +5.8594] pp; 28 questions gained
grounding, 11 lost. Conditional-on-parsing grounding rose by
+1.538 pp but its CI [-0.905, +3.981] pp includes zero.
Parse compliance rose +4.492 pp with CI excluding zero.
Hard GV agreement rose +2.291 pp, with CI including zero.
Soft-GV acceptance rose +2.300 pp with CI above zero.
The hard and soft GV weight truth-enrichment ratios actually
DECREASED by -0.0505 and -0.0437, with CIs crossing zero.
The rise in consistency should not be described as confirmed
better factual discrimination.

Round-2 training dynamics: parsed new training trajectories
were GV 292, G 292, G-ESS 294, hard 299 and soft 299. The hard
arm selected just 251/299 parsed trajectories; the other arms
used all their parsed trajectories (292 to 299), producing
different training sequence/token counts despite 82 updates.
Round 1 used 238 parsed trajectories across the cohort.
New generated populations diverge by arm after feedback; even
ESS matching within the same new population is not an exact
matching of effective sample size ACROSS independent branches.

### Next: CPU-only round-2 five-arm paired audit

The existing scripts/audit_five_arm_cft.py aligns question identity
and performs a vectorized paired bootstrap and exact conditional
McNemar test, on 512 previously inspected development questions.
Its q_GV-minus-G contrast is the primary descriptive stage-5
round-2 contrast; other comparisons exploratory. No GPU required.
Do not overwrite an existing audit file:

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    ITER="$ROOT/iterative-posterior-v1/seed2026"
    python scripts/audit_five_arm_cft.py \
      --arm "hard=$ITER/hard/round2/development_gv_n512.jsonl" \
      --arm "soft=$ITER/soft/round2/development_gv_n512.jsonl" \
      --arm "posterior-g=$ITER/posterior-g/round2/development_gv_n512.jsonl" \
      --arm "posterior-g-ess=$ITER/posterior-g-ess/round2/development_gv_n512.jsonl" \
      --arm "posterior-gv=$ITER/posterior-gv/round2/development_gv_n512.jsonl" \
      --target posterior-gv --bootstrap 5000 --seed 2026 \
      --output "$ITER/round2_five_arm_paired_audit.json"

### Round 3 continuation driver prepared, not executed

The new scripts/run_stage5_round3.sh accepts each of the five
modes as its only arm argument. It checks previous-round
adapter/config, training source provenance, seed/optimizer updates,
and prior development evaluation metadata. It refuses to overwrite
any preexisting round-3 directory. It uses each **matching round-2
adapter** to generate + score fresh round-3 data, freezes the same
q_G/q_GV correctness posteriors, fits a new LoRA from the base
model (82 updates), then evaluates on the same development cohort.
Round-3 training generation/order/validator seeds are 33003/34003/
35003; dev seeds remain 43/44/45. Round 3 has NOT run yet and
this new script still requires syntax-check/Colab validation.

Run the CPU audit and inspect paired contrasts BEFORE committing
GPU resources to all five round-3 arms. The sealed final holdout
must remain untouched.

## October 11, 2026: round-2 five-arm paired audit completed

The user executed the exact saved-JSONL question-aligned audit at the
round-2 paths from the preceding section. **72 unit tests passed**
(including the three new round-3 safety tests). The audit produced

    $ROOT/iterative-posterior-v1/seed2026/round2_five_arm_paired_audit.json

All 512 development question identities were aligned across the
five round-2 arms. Point metrics: hard 214/512 (41.80% grounded);
soft 212/512 (41.41%); posterior-g 202/512 (39.45%);
posterior-g-ess 205/512 (40.04%); posterior-gv 210/512 (41.02%).

For the all-question grounded-pair outcome, the paired **round-2**
GV-posterior minus comparator results (5000 question-bootstrap
resamples, percentile 95% CI and two-sided exact McNemar test) were:

| Comparator | GV - comparator (pp) | 95% paired bootstrap CI (pp) | gained/lost by GV | exact McNemar p |
|---|---:|---|---|---:|
| hard | -0.78 | [-2.93, +1.37] | 14/18 | .59661 |
| soft | -0.39 | [-2.15, +1.37] | 10/12 | .83181 |
| posterior-g (**primary**) | +1.56 | [-0.59, +3.52] | 18/10 | .18493 |
| posterior-g-ess | +0.98 | [-0.59, +2.54] | 11/6 | .33231 |

Every interval includes zero; every exact paired-test p-value exceeds
.05. Do **NOT** report incremental GV-posterior downstream
superiority. The original prospectively established frozen
*qGV versus qG predictive inference* finding is separate.

The round-1 to round-2 positive within-arm gains across all methods
are descriptive and partly reflect better parse compliance and more
newly trainable pairs. No isolated causal attribution to selection
weights is warranted. The qGV branch has the highest round-2 hard
GV agreement, 87.37%, but is NOT the top factual-grounded-yield arm.

**Stage-5 decision:** proceed with the preplanned third feedback round
as a dynamics pilot for the same five methods. No decision to run
the final 1024-question holdout has been made. The protected
scripts/run_stage5_round3.sh driver has passed CPU syntax and static
guardrail tests (72 overall), but actual GPU execution of round 3
remains to be validated. Preserve all existing round-2 files.

## October 11, 2026: Stage 5 all three rounds complete (seed 2026)

All five arm-specific round-3 generation, adapter-aware candidate scoring,
frozen-posterior reweighting, fresh-base 82-update LoRA training, and
512-question development evaluation jobs completed. A Colab disconnection
interrupted the original multi-arm session after the fourth arm, but
saved Drive artifacts were inspected and the not-yet-started soft arm
was subsequently run alone. All five round-3 evaluation JSONL files
and summaries are present.

### Development grounding trajectory (all-question parsed-and-grounded EM)

| Arm | R1 grounded/512 | R2 grounded/512 | R3 grounded/512 | R1->R3 change, pp |
|---|---:|---:|---:|---:|
| hard | 191 (37.3047%) | 214 (41.7969%) | 217 (42.3828%) | +5.0781 |
| soft | 184 (35.9375%) | 212 (41.4063%) | 221 (43.1641%) | +7.2266 |
| posterior-g | 186 (36.3281%) | 202 (39.4531%) | 214 (41.7969%) | +5.4688 |
| posterior-g-ess | 191 (37.3047%) | 205 (40.0391%) | 215 (41.9922%) | +4.6875 |
| posterior-gv | 193 (37.6953%) | 210 (41.0156%) | 213 (41.6016%) | +3.9063 |

All arms improved the end-to-end proxy across the full three-round
pilot. The **best R3 yield** was soft GV CFT (221/512). This does
NOT establish soft CFT superiority; the test is repeatedly used
development data and there is only **one iterative training seed**.

### Hard greedy GV agreement conditional on parsing (percent)

| Arm | R1 | R2 | R3 |
|---|---:|---:|---:|
| hard | 80.75 | 86.03 | 89.70 |
| soft | 81.86 | 84.42 | 85.77 |
| posterior-g | 83.54 | 85.71 | 89.18 |
| posterior-g-ess | 84.00 | 83.90 | 89.24 |
| posterior-gv | 85.08 | 87.37 | 85.86 |

Hard GV achieved the best R3 self-consistency, but soft GV had
the best R3 factual grounding. q_GV's R3 consistency declined
from R2 as its all-question grounding rose slightly. These
quantities are not interchangeable; do not claim a confirmed
feedback-induced hallucination mechanism based on the ranking alone.

### CPU-only round-3 paired exact test (completed)

Saved round-3 paired audit:

    $ROOT/iterative-posterior-v1/seed2026/round3_five_arm_paired_audit.json

Question alignment validated, n=512, 5000x paired bootstrap.
q_GV versus baselines for grounded-pair yield:

| Baseline | GV minus baseline, pp | 95% paired bootstrap CI, pp | GV gains/losses | exact two-sided McNemar p |
|---|---:|---|---|---:|
| hard | -0.78 | [-2.73,+1.17] | 12/16 | .57159 |
| soft | -1.56 | [-3.52,+0.39] | 11/19 | .20049 |
| posterior-g (PRIMARY) | -0.20 | [-1.76,+1.18] | 7/8 | 1.00000 |
| posterior-g-ess | -0.39 | [-1.95,+1.17] | 8/10 | .81453 |

ALL paired intervals contain zero, all p-values >.05. No
incremental GV-posterior CFT benefit is established in R3.
The q_GV-minus-q_G observed point contrast goes +1.3672,
+1.5625, -0.1953 pp from R1 to R3: **no progressively
accumulating advantage**. Three-seed Stage-4 single-round
effects also changed sign. Do not describe this study as
a successful factual-CFT improvement over controls.

### Scientific interpretation and research decision

1. The prospective, frozen, held-out confirmation of better
   q_GV versus q_G factual-validity *prediction* remains
   supported and should be presented separately.
2. A better conditional estimate of grounded-pair validity
   did not reliably improve downstream weighted paired-answer
   CFT across Stage-4 seeds or this Stage-5 feedback pilot.
3. Improved parse compliance and greater numbers of newly
   parsed training trajectories are potential contributors
   to across-round gains; the optimizer budget matched
   (82 updates) but training exposure did not.
4. The frozen posterior estimators were calibrated from
   base-model trajectories, then transported across evolving
   adapters and distinct answer populations. Posterior mean
   or training sequence NLL is not proof of retained
   predictive calibration or increased factual knowledge.
5. Stage 5 is a **completed one-seed feedback-dynamics
   pilot**, not a confirmatory iterative-method comparison.
   We should not run more rounds to pursue an advantageous
   R4/R5 checkpoint. The official 1024-question final
   holdout remains sealed.

Suggested next LOW-COST analyses before any GPU run:
(a) question-paired R1->R3 and R2->R3 transitions for
each arm, using saved JSONL; (b) per-arm/per-round
train selection truth enrichment and ESS using the
*raw generated trajectories* for diagnostics and the
separate gold-blind training-weight files (gold never
used for weights); (c) stratify longitudinal
improvement into parsing transitions versus factual
correctness among already-parseable questions.

A future component-target or direct-answer-QA ablation
should be explicitly positioned as a NEW exploratory
research question with a new frozen protocol, not as
positive retrospective evidence for Stage-5 hypotheses.

## Post-completion CPU-only audit: dynamics, enrichment and posterior transport

New script: `scripts/audit_stage5_dynamics.py`. All 15 evaluation
JSONL files and their 320-question *training-population* raw and
gold-blind weighted JSONL are read from Drive. No model loading,
generation, CUDA, posteriors refitting or final-holdout access is
performed. The command refuses to overwrite an existing output.

Colab with Google Drive mounted and repo bootstrapped:

    cd /content/gv-smc
    git pull --ff-only
    python -m pytest -q

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_stage5_dynamics.py \
      --root "$ROOT" \
      --seed 2026 --bootstrap 5000 \
      --output "$ROOT/iterative-posterior-v1/seed2026/stage5_dynamics_cpu_audit.json"

The complete saved report has:
- `eval_metrics`: all-question EM-proxy grounded-pair yield, parsing
  and hard consistency at every round for each of five arms.
- `paired_round_transitions`: same-question comparisons for 1->2,
  2->3 and 1->3, with exact gained/lost grounded counts and
  paired question-bootstrap intervals. It partitions gains and losses
  into **newly parseable** versus **already parseable** questions.
- `train_selection_diagnostics`: separate q_G, q_GV,
  q_G-ESS, hard and soft selection mass, ESS, weighted grounded
  precision, truth-enrichment ratio, plus base-trained posterior
  NLL/Brier/AUROC on each new model-generated **training** population.
- `posterior_hashes`: checks q_G and q_GV joblib SHA256
  provenance matches across all rounds, and checks no_refit /
  gold_blind metadata and no gold fields in weighted CFT files.

**Strong interpretation restriction:** Posterior diagnostics computed
against truth aliases on the 320 CFT *training questions* are
post-hoc, **in-sample with respect to the CFT adapters**, and
not independent calibration assessments. A discrepancy can suggest
distribution shift, but it does NOT causally diagnose it. The
probability predictor was originally fit on BASE trajectories
and fixed, whereas the generator-produced populations diverge by
method and round. Comparing q_G/q_GV NLL between such populations
also changes the target distribution and prevalence. q_G-ESS weights
are selection weights rather than calibrated probabilities.

The 512 development questions have already been repeatedly
inspected; question bootstrap intervals are descriptive,
conditional on ONE fixed training seed. Do not treat them as
confirmatory or pool as 1536 independent observations.
The official 1024-question final holdout is sealed.

The audit implementation was checked with synthetic CPU unit tests
before publication; rerun tests in Colab with the real repo,
then execute once. If a required input does not exist or fails
alignment/provenance checks, do not delete files or silently
substitute a historical cohort: inspect the specific path/error.

## October 11, 2026: completed CPU-only mechanistic audit (76 tests passed)

The user executed \`scripts/audit_stage5_dynamics.py\` successfully on
all 15 arm×round saved artifacts. The analysis verified same frozen
q_G and q_GV SHA256s across all rounds, no recalibration/refitting,
gold-blind CFT training files, and development/train question alignment.
Saved report:

    $ROOT/iterative-posterior-v1/seed2026/stage5_dynamics_cpu_audit.json

**Longitudinal question-paired grounding (fixed 512 dev questions)**

| Arm | R1→R2 net grounded | R2→R3 net grounded | R1→R3 net grounded | R1→R3 net from parsing | R1→R3 net among parseable both rounds |
|---|---:|---:|---:|---:|---:|
| hard | +23 | +3 | +26 | +8 | +18 |
| soft | +28 | +9 | +37 | +15 | +22 |
| posterior-g | +16 | +12 | +28 | +9 | +19 |
| posterior-g-ess | +14 | +10 | +24 | +11 | +13 |
| posterior-gv | +17 | +3 | +20 | +8 | +12 |

The parse-versus-factual decomposition is descriptive, not causal.
Among questions parseable in both r1/r3, grounded-pair correctness has
positive net change for all five arms. q_GV r2→r3: 18 gains / 15 losses,
+3/512 = +0.586 pp, question bootstrap 95% CI
[-1.5625,+2.734375] pp. Progress is **compatible with** saturation
but a single three-round seed does not demonstrate convergence.

**Post-hoc frozen q_GV minus q_G NLL on CFT TRAIN populations**
(negative favors q_GV; round-1 training population identical across
arms; new arm-specific populations for rounds 2/3):

| Arm | R1 | R2 | R3 |
|---|---:|---:|---:|
| hard | -.0259 | -.0168 | -.0037 |
| soft | -.0259 | -.0282 | -.0262 |
| posterior-g | -.0259 | -.0240 | -.0179 |
| posterior-g-ess | -.0259 | -.0184 | -.0227 |
| posterior-gv | -.0259 | -.0178 | -.0192 |

Corresponding q_GV minus q_G Brier differences were all negative,
including q_GV branch R1=-.0111, R2=-.0067, R3=-.0081.
Thus **relative post-hoc predictive discrimination survives** in
every observed arm×round training distribution, but the size changes
and it does NOT establish retained out-of-sample calibration.
Do not state "frozen posterior always calibrated" or "calibration
collapsed"; no independent adapted-model-label cohort was used.

**q_GV training-selection weight diagnostics by arm:**

| Arm | q_GV weighted grounded precision R1→R2→R3 | q_GV normalized ESS R1→R2→R3 |
|---|---|---|
| hard | .566→.563→.523 | .730→.774→.814 |
| soft | .566→.544→.566 | .730→.797→.810 |
| posterior-g | .566→.548→.540 | .730→.798→.830 |
| posterior-g-ess | .566→.547→.566 | .730→.806→.827 |
| posterior-gv | .566→.549→.545 | .730→.815→.832 |

These numbers refer to q_GV **weight diagnostics evaluated on each
arm's own generated TRAIN pairs**, even when that arm actually
trains with hard, soft, q_G or q_G-ESS. Weighted precision changes
with class prevalence, so require the report's \`precision_lift\`
before attributing changes to posterior selection quality.
Normalized ESS rises across all branches: weights become less
concentrated on each evolving generated population. This is a
candidate mechanism for weaker posterior-mediated selection, but
is not itself evidence that calibration worsened or that ESS caused
the downstream flattening.

### Interpretation and frozen pilot conclusion

The prospective unseen-question **inference** improvement from
q_GV relative to q_G is preserved as a separate, successful
experiment. In the iterative fine-tuning pilot, all arms gain
grounding, some owing to parse gains and some to question-level
changes among parseable answers, but the q_GV advantage is not
robust, does not accumulate, and reverses sign by round 3.
The post-hoc train NLL advantage survives while the downstream
CFT advantage vanishes. **A better predictive posterior is not
sufficient for improvement under this paired-answer CFT
objective.** A causal attribution between posterior drift,
overly homogeneous weights, and training-target mismatch cannot
be made from this audit.

Potential future exploratory follow-ups: quantify q_GV-vs-q_G
\`precision_lift\` and AUROC trends from the existing report;
design a direct-answer generator-only loss and/or separate
generator/validator-target ablations with matched data exposure;
and, if seeking transported-posterior calibration, use fresh
independent labeled questions generated by each adapted model
and freeze all analysis rules first. Do NOT unseal the 1024
official final holdout to choose such ablations retrospectively.
