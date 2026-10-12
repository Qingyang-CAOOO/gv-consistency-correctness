# Protocol-v2 reset: truth-enrichment audit and order-aware CFT

Status: implementation ready; no new GPU training has been launched by these commits.

## Research objective

Separate (1) parse compliance, (2) factual validity of the generator's
intended-correct/intended-misleading pair, (3) generator-validator agreement,
and (4) selection-induced enrichment of factual validity.

The existing TriviaQA test N=512 is a development set because it has been
inspected repeatedly. Do not use it as a pristine final test set. The RankAlign
test source has just 531 unique questions, so there is no useful untouched
final holdout in that file.

## Stage 0: verification and frozen question partitions

From the Colab project root, with RankAlign bootstrapped and Drive mounted:

    git pull --ff-only
    python -m pytest -q

    python scripts/make_triviaqa_split_manifest.py \
      --rankalign-root external/rankalign \
      --output-dir "/content/drive/MyDrive/gv-smc-artifacts/protocol-v2/splits"

The RankAlign corpus contains **1,611 unique train questions and only 531
unique test questions**. The historical CFT population used train seed 142,
while the earlier Phase-3 supervised calibration used train seed 42. Their
512-question subsets are overlapping but not identical. We reserve their
**union** before allocating *new* CFT or calibration questions.

Default new allocations: 320 CFT-train, 256 calibration-train, 128
calibration-validation. The precise remaining number is determined by the
overlap of the two historical cohorts; use `--inspect-only` before writing.
If a larger allocation is needed, obtain more *disjoint* source questions;
do not silently reuse historical questions.

The previously explored 512 questions from the 531-question RankAlign test
split remain the development set. By default `--final-test-size 0` writes
**no final holdout**: only 19 additional questions remain. We must later use
an untouched external evaluation source (for example, a question-deduplicated
portion of an official TriviaQA validation corpus), recorded in a separate
final-evaluation manifest. Never declare the remaining 19 questions sufficient
for paper-level inference.

Use `--inspect-only` to print exact capacity and verify before writing.
If insufficient data, the script fails explicitly: adjust partition sizes
only with documented choices, preserving the historical union and disjointness.
Do not change the saved manifest after any new experiment begins.

New inference scripts can run with an exact frozen partition by passing
\`--question-file <partition.jsonl>\` and the matching \`--limit\`.
The legacy seed-based behavior is unchanged when the flag is absent.

## Stage 1: audit existing hard/soft iterative checkpoints; no GPU required

    python scripts/audit_iterative_truth_enrichment.py \
      --artifact-root "/content/drive/MyDrive/gv-smc-artifacts/iterative-cft" \
      --rounds 3 \
      --test-limit 512 \
      --bootstrap 5000 \
      --seed 2026

Output:

    /content/drive/MyDrive/gv-smc-artifacts/iterative-cft/truth_enrichment_audit.json

The script audits the current base and hard/soft iteration 1--3 JSONL
records. It reports parse success, count of all four grounded-pair states,
end-to-end valid-and-correct rate, conditional correct-answer EM,
hard/soft-selected grounded-pair precision and recall, truth-enrichment
ratios, ESS fractions, and question bootstrap intervals. If paired question
identities match, it additionally computes paired hard-minus-soft intervals.

Definitions for parsed records: T=1 when intended_correct_em=1 and
intended_misleading_em=0; T=0 otherwise. This is a *proxy* for factual truth
based on RankAlign TriviaQA gold aliases and conservative exact matching.
Malformed generations are *missing for conditional factual state*; never
silently reclassify them as wrong. For end-to-end pipeline metrics, they
contribute zero to the valid-and-correct joint event by definition.

    LR_C = E[C | T=1, parsed] / E[C | T=0, parsed]
    LR_G = E[G | T=1, parsed] / E[G | T=0, parsed]
    selected_truth_Phi = sum(Phi_i * T_i) / sum(Phi_i)

The hard-weight field is **greedy-hard**, whereas sampled-hard is a distinct
Bernoulli control. The audit deliberately retains both columns.

The script reports a ratio as undefined when its denominator is zero.
Bootstrap sampling is over questions, not independent generator and
validator example components.

## Stage 2: score the opposite A/B order for existing *training* trajectories

After Stage 1, with Gemma2-2B-it access on a GPU:

    python scripts/augment_gv_both_orders.py \
      --data "/content/drive/MyDrive/gv-smc-artifacts/training-data/gemma2_2b_it_greedy_train_n512.jsonl" \
      --model google/gemma-2-2b-it \
      --output "/content/drive/MyDrive/gv-smc-artifacts/protocol-v2/base_train_both_orders_n512.jsonl"

This reuses the frozen generated answer pairs. It is not a new generation
sample and does not retrain. Subsequent iterative round data must be scored
using the adapter that originally generated each population; never score
such trajectories using the base model by accident. The script preserves
the source file and saves both order-specific prompts and probabilities.

The order-marginalized CFT objective is

    J = sum_i { [(GA_i+GB_i)/2] log p_theta(YG_i | XG_i)
              + (GA_i/2) log p_theta(A | XV_i,A)
              + (GB_i/2) log p_theta(B | XV_i,B) }
        / sum_i (GA_i+GB_i).

This is not the same as averaging the weight and training only one sampled
validator prompt. The mode is available in \`train_cft.py\` as
\`--mode soft-marginalized\`, with unit tests for both per-order terms.
Do not launch iterative training until the truth-enrichment audit and final
protocol review are complete.


## Stage 3: correctness-posterior calibration (new, no fine-tuning)

A frozen, question-disjoint split manifest has been created in Drive. It uses
256 calibration-train and 128 calibration-validation questions, both disjoint
from the 320 new CFT-train questions and the 861-question historical train union.
Do not overwrite the manifest or substitute a previously explored test split.

The binary target is grounded pair validity, conditional on the generator's
output being parseable: T=1 iff intended-correct EM is true and
intended-misleading EM is false. The both-incorrect / reversed / both-correct
cases are T=0. This is **conservative exact-match supervision**, not a
perfect truth oracle. The posterior is supervised-calibrated, not label free.

### Collect original CFT-format measurements (A100)

Run from /content/gv-smc, with Drive mounted:

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/run_triviaqa_gv.py \
      --model google/gemma-2-2b-it \
      --device cuda \
      --question-file "$ROOT/splits/calibration_train.jsonl" \
      --limit 256 --generation-seed 27001 --order-seed 27002 \
      --validator-sample-seed 27003 --paired-order-diagnostic \
      --output "$ROOT/calibration_train_gv_n256.jsonl"

    python scripts/run_triviaqa_gv.py \
      --model google/gemma-2-2b-it \
      --device cuda \
      --question-file "$ROOT/splits/calibration_val.jsonl" \
      --limit 128 --generation-seed 28001 --order-seed 28002 \
      --validator-sample-seed 28003 --paired-order-diagnostic \
      --output "$ROOT/calibration_val_gv_n128.jsonl"

These GPU measurements use **exactly the original CFT paired-answer
generator and validator prompts**, not Phase 3A's alternative candidate
ranking task. Every parsed example receives both validator positions.
The existing GV runner uses a tqdm per-question progress bar.

### Score generator-only preference (A100)

    python scripts/score_gv_generator_preferences.py \
      --data "$ROOT/calibration_train_gv_n256.jsonl" \
      --output "$ROOT/calibration_train_features_n256.jsonl" \
      --model google/gemma-2-2b-it --device cuda --require-cuda

    python scripts/score_gv_generator_preferences.py \
      --data "$ROOT/calibration_val_gv_n128.jsonl" \
      --output "$ROOT/calibration_val_features_n128.jsonl" \
      --model google/gemma-2-2b-it --device cuda --require-cuda

For each intended-correct and intended-misleading candidate, the generator
scores the conditional log probability under the same **direct-answer**
question prompt. We extract length-normalized likelihood and candidate
token-count ratio. No gold aliases are used to form these features.
The script shows model-scored pair counts, elapsed time, and ETA.
Existing output files are protected unless --overwrite is explicitly passed.

### Fit and evaluate the supervised binary posterior (CPU)

    python scripts/fit_cft_grounding_posterior.py \
      --train "$ROOT/calibration_train_features_n256.jsonl" \
      --validation "$ROOT/calibration_val_features_n128.jsonl" \
      --output-dir "$ROOT/posterior_base_v1" \
      --bootstrap 2000 --seed 2026

Models:
- q_G uses positive/negative candidate conditional average token logprob,
  and the log ratio of their token counts.
- q_GV uses exactly the same features **plus** the mean probability of
  selecting the generator's intended-correct answer across both A/B orders
  and the absolute order gap.
- Both use sklearn StandardScaler + L2-regularized LogisticRegression(C=1).
  No class weighting and no hyperparameter tuning on calibration validation.

The script checks question identity non-overlap, excludes unparsed
generations **from conditional posterior estimation**, saves fitted models,
per-question held-out posterior predictions, and reports held-out NLL,
Brier score, AUROC, weighted grounded precision, ESS, and paired question
bootstrap intervals for q_GV - q_G. It also compares constant-prior
predictions and uncalibrated order-marginalized soft GV potentials.

The models approximate q(T=1 | observed features,D=1); they are not
guaranteed calibrated simply because logistic regression is used. We
should inspect reliability plots and grouped out-of-split calibration
before committing to downstream CFT. With 256/128 questions and parsing
failures, this is an initial feasibility study; rare states (11 and 01)
are not separately learned.

### Gate before any CFT-III training

Only propose a posterior-weighted CFT run after:
1. The frozen-question split and all tests pass.
2. q_GV is competitive with q_G and has convincing held-out improvement
   in proper predictive scores (NLL/Brier), not just training accuracy.
3. Selection precision/lift for q_GV is compared to q_G and consistency
   weights on the *same held-out answer pairs*.
4. Calibration quality and uncertainty are inspected. If validation is
   inconclusive, expand from **new disjoint source questions** instead of
   tuning many variants on the 128-question validation set.

If the evidence supports moving forward, CFT-III will use the frozen
q_GV model to weight **new** CFT-train trajectories. The q_G model must
be used as a supervision-matched control. No CFT-III checkpoint is
trained by this stage.

## Stages 4--5 (not implemented yet)

1. Collect paired-answer calibration data on **calibration_train** and
   **calibration_val**, disjoint from the CFT training questions.
   Fit a generator-only prior versus generator+validator calibrated posterior
   of the *grounded* state (correct designated correct, incorrect designated
   misleading), and preserve all four factual pair-state diagnostics.
2. Run a matched **single-round** experiment with hard, soft sampled-order,
   soft order-marginalized, generator-only-calibrated, and posterior-weighted
   selection. A label-supervised oracle arm may be included only as an
   explicitly privileged upper-bound comparator.
3. Only after single-round results, run selected arms for three rounds,
   evaluate calibration shift and truth-enrichment drift, and finally use
   the untouched question pool or a separate dataset.

CFT-III will be *supervised-calibrated*, not label-free CFT. The existing
Phase-3A pairwise preference calibrator is useful precedent, but is not a
calibrated probability that the original generated two-answer CFT pair is
factually valid.


## Stage 4: matched single-round posterior CFT (implemented, not yet run)

On October 11, 2026, the locked pooled 598-example calibrator was prospectively
confirmed on the official 512-question question-disjoint confirmation partition:
394 parsed, 177 grounded pairs. Frozen q_GV improved on frozen q_G:

| Primary/secondary endpoint | q_G | q_GV | paired GV minus G, 95% CI |
|---|---:|---:|---|
| NLL | .597160 | .563113 | -.034047 [-.057660, -.011543] |
| Brier | .201548 | .188140 | -.013408 [-.022270, -.004903] |
| AUROC | .753235 | .786560 | +.033326 [.010799, .057477] |
| Weighted grounded precision | .583160 | .607329 | +.024168 [.011833, .037463] |

q_GV versus ESS-matched q_G selection precision difference remained
+.017151, CI [+ .006090, + .029055]. This prospectively demonstrates
incremental validator predictive signal for the EM-proxy target conditional
on parsing, but **does not prove improved fine-tuning performance**.
Confirmation is now a development result; DO NOT retune the frozen models
using it, and DO NOT touch official_final_holdout.

### Experimental arms

**Core matched three-way ablation (same 320 NEW CFT questions):**

- \`posterior-g\`: supervised-calibrated generator evidence only,
  frozen q_G(T=1).
- \`posterior-gv\`: the identical generator features plus both-order
  validator evidence, frozen q_GV(T=1).
- \`posterior-g-ess\`: generator-only weights power-transformed to the same
  **training-population** ESS as q_GV (no labels used in power matching);
  this is a concentration-matched baseline, *not calibrated*.

For complete paper-style baseline comparison, train additional \`hard\`,
\`soft\`, and optionally \`soft-marginalized\` arms on the exact same frozen
generator trajectories, matching epoch/update settings. Hard selects fewer
sequences than the other three; equal optimizer-update count does NOT imply
equal token- or FLOP-budget. Report both.

### Step A: generate one shared fresh CFT population (A100, progress + ETA)

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/run_triviaqa_gv.py \
      --model google/gemma-2-2b-it --device cuda \
      --question-file "$ROOT/splits/cft_train.jsonl" \
      --limit 320 --generation-seed 30001 --order-seed 30002 \
      --validator-sample-seed 30003 --paired-order-diagnostic \
      --output "$ROOT/single-round/cft_train_gv_n320.jsonl"

### Step B: generator likelihood features (A100, progress + ETA)

    python scripts/score_gv_generator_preferences.py \
      --data "$ROOT/single-round/cft_train_gv_n320.jsonl" \
      --output "$ROOT/single-round/cft_train_features_n320.jsonl" \
      --model google/gemma-2-2b-it --device cuda --require-cuda

### Step C: freeze posterior weights, remove labels (CPU, progress + ETA)

    python scripts/apply_frozen_cft_posteriors.py \
      --data "$ROOT/single-round/cft_train_features_n320.jsonl" \
      --model-dir "$ROOT/posterior_base_v2_pooled" \
      --expected-question-file "$ROOT/splits/cft_train.jsonl" \
      --forbid-question-file "$ROOT/splits/calibration_train.jsonl" \
      --forbid-question-file "$ROOT/splits/calibration_val.jsonl" \
      --forbid-question-file "$ROOT/splits/legacy_train.jsonl" \
      --forbid-question-file "$ROOT/splits/official_confirmation.jsonl" \
      --output "$ROOT/single-round/cft_train_posterior_weights_n320.jsonl"

The output is a **gold-blind training file**: the producer uses only an
explicit allowlist of inference-time features and drops all
gold_answers / EM / grounded_pair fields before writing. It records frozen
model file hashes, question-source identity, q_G/q_GV ESS, ESS-matching
power and full source provenance. It refuses overwrite or mismatched
question identities.

### Step D: matched one-seed LoRA feasibility run (A100)

    for MODE in posterior-g posterior-gv posterior-g-ess
    do
      python scripts/train_cft.py \
        --data "$ROOT/single-round/cft_train_posterior_weights_n320.jsonl" \
        --mode "$MODE" --model google/gemma-2-2b-it \
        --output-dir "$ROOT/single-round/checkpoints/$MODE" \
        --epochs 2 --updates-per-epoch 41 --seed 2026
    done

All three models use identical parsed source pairs, original sampled-order
generator/validator prompts and completions, optimizer update count and
random seed. Only their **training trajectory weights** differ.
These are **new** LoRA adapters trained from the same base model. Completion
sequence counts match among all three; gradient and weight effects differ.

### Step E: paired evaluation on previously inspected development questions

    for MODE in posterior-g posterior-gv posterior-g-ess
    do
      python scripts/run_triviaqa_gv.py \
        --model google/gemma-2-2b-it \
        --adapter "$ROOT/single-round/checkpoints/$MODE" \
        --device cuda \
        --question-file "$ROOT/splits/development.jsonl" \
        --limit 512 --generation-seed 43 \
        --order-seed 44 --validator-sample-seed 45 \
        --paired-order-diagnostic \
        --output "$ROOT/single-round/eval/\${MODE}_dev_n512.jsonl"
    done

Then compare matched question-level bootstrap outcome differences on
**all 512 development questions**, not just parsed ones:

    python scripts/audit_single_round_cft.py \
      --arm "posterior-g=$ROOT/single-round/eval/posterior-g_dev_n512.jsonl" \
      --arm "posterior-gv=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --arm "posterior-g-ess=$ROOT/single-round/eval/posterior-g-ess_dev_n512.jsonl" \
      --first posterior-g --second posterior-gv \
      --output "$ROOT/single-round/paired_dev_audit.json" \
      --bootstrap 5000 --seed 2026

**Primary downstream outcome:** difference in *all-question*
P(parsed AND generated pair grounded by EM alias), q_GV minus q_G.
**Secondary:** parsing, grounded given parsed, intended-correct-only EM
yield, validator consistency, and q_GV minus q_G-ESS.
Comparisons with hard/soft must use identical *new* training question
pool to avoid confounding by changes in source population.

One seed (2026) is only exploratory. Positive development-set evidence
should lead to a preplanned multi-training-seed comparison (at least
three seeds) before touching the final 1,024-question untouched holdout.
The previously evaluated 512 official confirmation questions are
not a pristine final downstream evaluation set.

### Important distinction

The frozen correctness posterior is learned using supervised labels, but
its predictions for new CFT training questions are not direct gold labels.
This is **supervised-calibrated weighted MLE**, not a label-free CFT method
and not mathematically exact EM. The posterior weights do not become
probabilities of semantic truth simply by being calibrated against EM.


### October 11, 2026: Stage-4 frozen training population obtained

User-side Colab run passed all 58 tests at commit 0531e0e. The new,
question-disjoint 320-question CFT population has **238 parsed pairs**,
including 105 exactly grounded (44.12% given parsed). Generation and
generator-only likelihood feature extraction have completed; no LoRA
training has yet been run.

The gold-blind posterior training file has been written at
\`$ROOT/single-round/cft_train_posterior_weights_n320.jsonl\`.
Its inference features and frozen model hashes are logged in the paired
\`.summary.json\`:

- q_G mean .4025176692; ESS fraction .7491802647
- q_GV mean .4123795544; ESS fraction .7298680315
- q_G ESS-matching exponent 1.0618032217; ESS fraction .7298680353
- 1,034 gold/EM-related field occurrences stripped
- Frozen q_G model SHA256
  458d4f5660e24b8d6d6741654ca6e76c0a3f5e603d8e82389225005f9e39e3b5
- Frozen q_GV model SHA256
  5a5199dd5790bc2b93156f85571165dccc250ff4800afc048e7b9ed4ad1d1a79

The three posterior arms have **identical 476 training sequences**
(238 generator, 238 validator) per epoch. With \`--epochs 2\` and
\`--updates-per-epoch 41\`, they perform **82 optimizer updates** each.
Unlike the published historical 512-question runs, these use the newly
frozen 320-question training population.

To protect GPU runs and reproducibility, \`train_cft.py\` now refuses
to overwrite any nonempty output checkpoint directory before loading
the base LM. For future training seeds, use a new directory.

Use the existing Step D training and Step E development evaluation
commands above. Keep official_final_holdout untouched, and report
the per-arm \`gvsmc_training_summary.json\`, especially the optimizer
updates and weighted sequence-NLL trajectory. A single training seed
is a feasibility/pilot experiment, not final inferential evidence.


### October 11, 2026: Stage-4 single-seed LoRA downstream pilot

All three posterior arms were trained in Colab from the exact same 238
parsed frozen CFT-training pairs (476 generator+validator sequences), 2
epochs, 82 optimizer updates, fixed seed 2026, LoRA r=8/alpha=16/dropout=.05,
learning rate 2e-5, 10,383,360 trainable parameters. All 58 tests passed.

Training NLL is reported for monitoring **the arm-specific weighted
objectives**, not as a common held-out metric:

| Arm | epoch1 weighted sequence NLL | epoch2 |
|---|---:|---:|
| posterior-g | 9.2897 | 2.4367 |
| posterior-gv | 9.1416 | 2.3982 |
| posterior-g-ess | 9.2756 | 2.4185 |

The previously explored frozen 512-question RankAlign development set,
with the SAME generation/validator inference settings across all arms,
yielded:

| Arm | parsed / 512 | grounded among parsed | grounded among all 512 | intended correct among all | greedy hard GV consistency |
|---|---:|---:|---:|---:|---:|
| posterior-g | 474 | 186/474 = .3924 | 186/512 = .3633 | .3848 | .8354 |
| posterior-gv | 476 | 193/476 = .4055 | 193/512 = .3770 | .3945 | .8508 |
| posterior-g-ess | 475 | 191/475 = .4021 | 191/512 = .3730 | .3926 | .8400 |

Preregistered primary downstream paired question-bootstrap:
q_GV-CFT minus q_G-CFT *all-question* parse-and-grounded EM yield
= **+7/512 = +.0136719**, percentile 95% CI
[-.0019531, +.03125]. CI includes zero, therefore the pilot does NOT
demonstrate statistically resolved downstream improvements for this one
training seed. q_GV-CFT minus ESS-matched q_G-CFT point difference is
**+2/512 = +.00390625**; run the direct paired bootstrap and report its
CI before interpreting it.

All arm-specific truth-enrichment ratios remain near 1.17-1.19: there
is no evidence here of recovered downstream filtering discrimination.
The best GV posterior training curve is **not** itself evidence of improved
generalization, since each arm changes the loss weights.

#### Cheap next comparison: GV versus ESS-matched generator CFT

The audit script was extended to print per-question **lost / gained /
unchanged** grounded-pair transitions, in addition to its existing paired
bootstrap. Run the direct ESS-matched comparison on saved evaluations:

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_single_round_cft.py \
      --arm "posterior-g-ess=$ROOT/single-round/eval/posterior-g-ess_dev_n512.jsonl" \
      --arm "posterior-gv=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --first posterior-g-ess --second posterior-gv \
      --output "$ROOT/single-round/paired_gv_minus_ess_dev_audit.json" \
      --bootstrap 5000 --seed 2026

This is cheap CPU computation, not a new evaluation set. The outcome
remains exploratory even if the CI excludes zero.

#### Next matched original-CFT controls

Before drawing any method-level conclusion, train at least ONE hard-CFT
and ONE soft-CFT baseline on this exact new 320-question frozen population.
Use the **gold-blind** posterior-weight file: it preserves the hard/soft
original potentials, generator/validator prompts and completions. Training
needs identical seed, scheduler, epochs, and **41 optimizer steps/epoch**.

    for MODE in hard soft; do
      CKPT="$ROOT/single-round/checkpoints/${MODE}-matched-n320-seed2026"
      if [ -e "$CKPT" ]; then
        echo "Refusing to reuse existing checkpoint path: $CKPT" >&2
        exit 1
      fi
      python scripts/train_cft.py \
        --data "$ROOT/single-round/cft_train_posterior_weights_n320.jsonl" \
        --mode "$MODE" --model google/gemma-2-2b-it \
        --output-dir "$CKPT" \
        --epochs 2 --updates-per-epoch 41 --seed 2026
    done

    for MODE in hard soft; do
      CKPT="$ROOT/single-round/checkpoints/${MODE}-matched-n320-seed2026"
      python scripts/run_triviaqa_gv.py \
        --model google/gemma-2-2b-it --adapter "$CKPT" --device cuda \
        --question-file "$ROOT/splits/development.jsonl" --limit 512 \
        --generation-seed 43 --order-seed 44 --validator-sample-seed 45 \
        --paired-order-diagnostic \
        --output "$ROOT/single-round/eval/${MODE}-matched-n320-seed2026_dev_n512.jsonl"
    done

Evaluate each adapter on the exact SAME previously explored development
questions/seeds with run_triviaqa_gv.py. Hard filtering retains fewer
generator+validator sequences than the 476 used by posterior and soft
training; match optimizer updates but report different sequence/token
budgets. Do not compare to prior hard/soft checkpoints trained on
different question cohorts as if these were controlled ablations.

An earlier run was stopped by a preexisting nonempty checkpoint folder
`$ROOT/single-round/checkpoints/hard`, correctly prevented by the
nonoverwrite preflight. This folder's provenance is unknown until its
`gvsmc_training_summary.json` is inspected. The distinct
`*-matched-n320-seed2026` directories above avoid clobbering historical
models and ensure the evaluation command points to precisely those newly
trained adapters. Training-data and evaluation files are already frozen;
do not regenerate them or use the protected 1024-question final holdout.

The next large-sample robustness check is multiple **independent training
seeds**, keeping training pairs and evaluation questions fixed. Question
bootstrap CIs conditional on one seed do not quantify variance across
initialization/data-order seeds. Prioritize q_GV, q_G-ESS and soft-CFT
control arms for more seeds only after examining these first controls.

**Do not access or evaluate official_final_holdout yet**; it is the
unexpended 1,024-question final evaluation resource for the eventual
predeclared model/arm selection, and this stage is development only.

Key interpretation: A prospective inference advantage
q_GV(T|generator, validator) > q_G(T|generator) was established on the
locked 394-parsed-question confirmation set. That does **not logically
imply** downstream improved fine-tuning. This single-round single-seed
fine-tuning study shows favorable but small *point estimates*, with the
main paired CI crossing zero. The distinction should remain explicit.


### October 11, 2026: five matched one-seed arms completed

All three posterior runs and both original hard/soft CFT control arms have
been evaluated on the same 512 previously inspected development questions
under the same greedy generation and randomized validator settings. The
run based on commit f2f8d9c passed 59 Colab tests, before adding the new
five-arm audit utility and tests.

All five arms use the same frozen new 320-question source file and seed
2026, with 82 optimizer updates over two epochs, and matched LoRA
hyperparameters. Hard CFT selected 198 trajectories / 396
generator+validator sequences, whereas soft and all three posterior
arms used all 238 parsed trajectories / 476 sequences. **Equal updates
do not imply equal examples, token budget, or FLOPs.**

| CFT training potential | parsed of 512 | grounded of 512 | grounded fraction | hard GV agreement |
|---|---:|---:|---:|---:|
| hard | 478 | 191 | .373046875 | .807531 |
| soft | 474 | 184 | .359375 | .818565 |
| posterior-g | 474 | 186 | .36328125 | .835443 |
| posterior-g-ess | 475 | 191 | .373046875 | .840000 |
| posterior-gv | 476 | 193 | .376953125 | .850840 |

The best observed grounded yield belongs to posterior-GV CFT, but it
exceeds hard and ESS-matched generator CFT by only 2/512 = .00390625.
Relative to generator CFT it gains 7/512; the previously computed
paired question bootstrap 95% CI on that primary contrast was
[-.001953125, +.03125], which includes zero. The GV minus ESS matched
contrast was +2/512 with CI [-.01171875, +.01953125]; the paired
transition counts were 10 gained and 8 lost groundings.

The preexisting checkpoint directory
\`$ROOT/single-round/checkpoints/hard\` was discovered to contain
the same metadata, training data, seed, and reported weighted-NLL
trajectory as the new \`hard-matched-n320-seed2026\` checkpoint.
It was thus an apparent duplicate training run. Preserve both; no
further reruns required. File-byte equality was NOT checked.

Training code check: loss = (weights * per-sequence-NLL).mean() /
mean_sequence_weight, before normalizing across optimizer-accumulation
groups. Thus absolute scaling of weights is normalized away; relative
weights still change the objective. Hard-vs-soft selection changes
the number of sequences and token budget.

#### Next step: five-arm paired CPU audit, no GPU runs

One new script performs all target-versus-baseline question-aligned
grounded-yield contrasts from existing saved JSONL outputs and reports
95% question-bootstrap CIs, gain/loss transitions, parse rate, all-question
intended-correct EM and hard consistency. Bootstrap draws share their
sampled questions across all arms. Primary contrast: GV posterior
versus G posterior all-question grounded yield; other contrasts and
endpoints are *exploratory*, unadjusted for multiplicity.

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_five_arm_cft.py \
      --arm "hard=$ROOT/single-round/eval/hard-matched-n320-seed2026_dev_n512.jsonl" \
      --arm "soft=$ROOT/single-round/eval/soft-matched-n320-seed2026_dev_n512.jsonl" \
      --arm "posterior-g=$ROOT/single-round/eval/posterior-g_dev_n512.jsonl" \
      --arm "posterior-g-ess=$ROOT/single-round/eval/posterior-g-ess_dev_n512.jsonl" \
      --arm "posterior-gv=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --target posterior-gv \
      --output "$ROOT/single-round/five_arm_dev_audit.json" \
      --bootstrap 5000 --seed 2026

Do not regenerate any existing evaluation or train additional adapters
before interpreting this audit. Follow it with independent training
seeds only for the prespecified comparison of most interest, as budget
allows, and preserve the **official_final_holdout n=1024** until final
model-choice and evaluation protocols are frozen.

The research distinction remains: adding V gave a prospectively
confirmed gain in *predicting factual validity* from frozen GV
measurements (394 parsed official confirmation questions), but its
additional downstream training value beyond hard CFT or ESS-matched
generator posterior is unresolved. Under the current original GV
generator target, a single full generator completion **contains both the
intended-correct and intended-misleading answer**; even grounded-pair
weighting learns that *paired-answer instruction*, not pure
single-answer factual QA. This is a plausible bottleneck worth
targeted ablation rather than proof of a causal mechanism.


### October 11, 2026: completed five-arm paired audit and exact-test correction

Colab tests: **63 passed** at commit b08593b, before the additions
below. Same original 512-question development set, single training
seed 2026, all five arms on the same 320 training questions.

The five-arm question-bootstrap results (5,000 resamples), GV posterior
minus comparator, in percentage points of **all-question parsed-and-
grounded EM yield**:

| comparator | GV delta (pp) | unadjusted percentile bootstrap 95% CI (pp) | gained / lost grounded |
|---|---:|---:|---:|
| hard | +0.3906 | [-1.56,+2.54] | 15 / 13 |
| soft | +1.7578 | [+0.20,+3.52] | 14 / 5 |
| generator-only posterior | +1.3672 | [-0.20,+3.12] | 13 / 6 |
| generator posterior ESS-matched | +0.3906 | [-1.17,+1.95] | 10 / 8 |

The soft contrast has a percentile-bootstrap CI that excludes zero.
**But** an EXACT two-sided conditional McNemar test on the 14 gained
and 5 lost pairs yields p = 0.063568115234375, not below 0.05. The
two tests are not equivalent in this small discrete discordance regime.
Therefore report the soft contrast as *suggestive*, not definitively
positive. In addition, it is a secondary comparison with unadjusted
uncertainty on a repeatedly analyzed development set.

The primary predeclared downstream contrast remains GV posterior
minus generator posterior (+7/512, CI includes zero). The strongest
hard and generator ESS-matched baselines are within only 2 grounded
pairs of GV posterior. No downstream method superiority is
established by this one-seed experiment.

The script \`scripts/audit_five_arm_cft.py\` now reports the exact
two-sided conditional McNemar p-value as an additional audit, without
changing the previously saved bootstrap results:

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_five_arm_cft.py \
      --arm "hard=$ROOT/single-round/eval/hard-matched-n320-seed2026_dev_n512.jsonl" \
      --arm "soft=$ROOT/single-round/eval/soft-matched-n320-seed2026_dev_n512.jsonl" \
      --arm "posterior-g=$ROOT/single-round/eval/posterior-g_dev_n512.jsonl" \
      --arm "posterior-g-ess=$ROOT/single-round/eval/posterior-g-ess_dev_n512.jsonl" \
      --arm "posterior-gv=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --target posterior-gv \
      --output "$ROOT/single-round/five_arm_dev_audit_exact.json" \
      --bootstrap 5000 --seed 2026

New distinct output path prevents overwriting the original
five_arm_dev_audit.json.

#### Multi-seed robustness design — next GPU runs

Prioritize three original-CFT trajectory-weight modes:
\`posterior-g\` (predeclared primary), \`posterior-gv\`, and
\`posterior-g-ess\` (controls for relative selection concentration).
Repeat **the same 238 parsed train trajectories** for seeds 2027 and
2028, keeping the *data, decoding/evaluation question set, optimizer
budget, and LoRA hyperparameters* constant. Only train initialization,
dropout/data order and seed-dependent choices change.

One new seed requires three LoRA runs (~35 min training on prior
hardware) plus three 512-question evaluations (~50 min).
Do not regenerate the data, posterior weights, or gold aliases.
Do not touch the official 1024-question final_holdout.

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    for SEED in 2027 2028; do
      for MODE in posterior-g posterior-gv posterior-g-ess; do
        CKPT="$ROOT/single-round/checkpoints/seed$SEED/$MODE"
        OUT="$ROOT/single-round/eval/seed$SEED/$MODE"_"dev_n512.jsonl"
        if [ -e "$CKPT" ] || [ -e "$OUT" ]; then
          echo "Refusing to overwrite preexisting data: $CKPT or $OUT" >&2
          exit 1
        fi
        python scripts/train_cft.py \
          --data "$ROOT/single-round/cft_train_posterior_weights_n320.jsonl" \
          --mode "$MODE" --model google/gemma-2-2b-it \
          --output-dir "$CKPT" --epochs 2 \
          --updates-per-epoch 41 --seed "$SEED"
        python scripts/run_triviaqa_gv.py \
          --model google/gemma-2-2b-it --adapter "$CKPT" --device cuda \
          --question-file "$ROOT/splits/development.jsonl" --limit 512 \
          --generation-seed 43 --order-seed 44 \
          --validator-sample-seed 45 --paired-order-diagnostic \
          --output "$OUT"
      done
    done

The cross-seed audit utility receives repeated \`--run SEED:ARM=PATH\`
arguments. Supply the already saved 2026 posterior evaluation JSONLs
plus all three new arm paths for seeds 2027 and 2028:

    python scripts/audit_multiseed_cft.py \
      --run "2026:posterior-g=$ROOT/single-round/eval/posterior-g_dev_n512.jsonl" \
      --run "2026:posterior-gv=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --run "2026:posterior-g-ess=$ROOT/single-round/eval/posterior-g-ess_dev_n512.jsonl" \
      --run "2027:posterior-g=$ROOT/single-round/eval/seed2027/posterior-g_dev_n512.jsonl" \
      --run "2027:posterior-gv=$ROOT/single-round/eval/seed2027/posterior-gv_dev_n512.jsonl" \
      --run "2027:posterior-g-ess=$ROOT/single-round/eval/seed2027/posterior-g-ess_dev_n512.jsonl" \
      --run "2028:posterior-g=$ROOT/single-round/eval/seed2028/posterior-g_dev_n512.jsonl" \
      --run "2028:posterior-gv=$ROOT/single-round/eval/seed2028/posterior-gv_dev_n512.jsonl" \
      --run "2028:posterior-g-ess=$ROOT/single-round/eval/seed2028/posterior-g-ess_dev_n512.jsonl" \
      --output "$ROOT/single-round/multiseed_posterior_cft_audit.json"

This reports per-seed effects, their mean and **sample standard
deviation across training seeds**, counts of favorable/unfavorable seeds,
and per-seed grounding transitions. It deliberately avoids claiming
significance from seed*question pseudo-replication. **Even three seeds
provide limited information about between-seed variability**, and
comparison on repeatedly examined development questions remains
exploratory. Evaluate hard/soft repeated seeds later only if required
by the final method-level claim, matched to a declared compute budget.

#### Structural follow-up, only after multi-seed audit

The original generator completion contains intended-correct AND intended-
misleading answers. A posterior-weighted generator loss therefore
supervises *both* answers even for an exactly grounded pair.
The existing \`train_cft.py --components generator|validator|both\`
enables a subsequent factorial component ablation on the **same
training trajectories**, including a careful separate factual-QA
outcome. Do not conflate improved pair validity with direct QA accuracy,
and do not treat a change in validator behavior as proof of better
generator knowledge.

The prospectively confirmed **posterior prediction** improvement and
the as-yet-unresolved **downstream CFT training** improvement are
distinct contributions. The final official holdout remains untouched.


### October 11, 2026: 2027 training-seed posterior CFT replication

The user completed the second independent LoRA initialization/shuffle seed
(2027), using exactly the original 320 question CFT population / 238
parsed pairs, same frozen q_G/q_GV/g-ESS weights, two epochs and 82
optimizer updates for each of three posterior modes. All three were
evaluated on the same 512 previously used RankAlign development questions
with fixed greedy generation and validator order seeds (43, 44, 45).
The 2027 run completed; these are **not independent evaluation questions**.

| Method | Seed 2026 grounded/all 512 | Seed 2027 grounded/all 512 |
|---|---:|---:|
| posterior-g | 186 (.36328125) | 189 (.369140625) |
| posterior-g-ess | 191 (.373046875) | 188 (.3671875) |
| posterior-gv | 193 (.376953125) | 192 (.375) |

2027 parsed totals: g 471, gv 472, g-ESS 479.
2027 hard GV consistency conditional on parsing: g .855626,
gv .834746, g-ESS .847599; **the GV-consistency ranking reverses**
relative to seed 2026. This is further evidence that GV agreement and
factual-pair correctness are different metrics.

Two-seed descriptive grounded-pair yield differences:
GV minus G = [+7/512, +3/512], mean +5/512 = +0.9765625 pp;
GV minus G-ESS = [+2/512, +4/512], mean +3/512 = +0.5859375 pp.
Both positive in each seed but small. This does not demonstrate a
statistically reliable CFT advantage over either comparator; the
development set is repeatedly inspected.

Observed 2027 timing on the user's upgraded GPU: per-arm training
[7:22, 7:10, 7:22] and GV evaluation [12:26, 13:07, 12:42],
about 60 minutes of total progress-bar runtime for three pairs.
The hardware model was not logged; do not label this a measured
T4-versus-A100 hardware benchmark.

#### Cheap two-seed audit from saved JSONL

The cross-training-seed audit works with two seeds and warns that
seed variability with n=2 is unreliable. Run without generating new
responses or modifying checkpoints:

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_multiseed_cft.py \
      --run "2026:posterior-g=$ROOT/single-round/eval/posterior-g_dev_n512.jsonl" \
      --run "2026:posterior-gv=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --run "2026:posterior-g-ess=$ROOT/single-round/eval/posterior-g-ess_dev_n512.jsonl" \
      --run "2027:posterior-g=$ROOT/single-round/eval/seed2027/posterior-g_dev_n512.jsonl" \
      --run "2027:posterior-gv=$ROOT/single-round/eval/seed2027/posterior-gv_dev_n512.jsonl" \
      --run "2027:posterior-g-ess=$ROOT/single-round/eval/seed2027/posterior-g-ess_dev_n512.jsonl" \
      --output "$ROOT/single-round/multiseed_posterior_cft_2026_2027.json"

Remaining third seed: 2028, same weights, architecture, questions,
evaluation decoder seeds, and training budget, with isolated
\`seed2028\` checkpoint and evaluation output paths. The final official
1024-question holdout remains untouched. Before final evaluation,
predeclare both arm selection and the multi-seed reporting rule.


### October 11, 2026: third seed completed; downstream robustness claim NOT supported

The user completed the third planned matched CFT LoRA training and
development evaluation seed 2028, preserving the original 320
question CFT training file, frozen posterior weights, 238 parsed
trajectories/476 sequences, two epochs and 82 updates, architecture
and decoding settings. GPU logs confirm the training and evaluation
workloads completed for posterior-g, posterior-gv, and
posterior-g-ess.

The three-seed **grounded-pair exact-match yield** on identical
512-question development sets:

| Training method | seed 2026 | seed 2027 | seed 2028 | average over seeds |
|---|---:|---:|---:|---:|
| posterior-g | 186/512 | 189/512 | 195/512 | 37.1094% |
| posterior-g-ess | 191/512 | 188/512 | 192/512 | 37.1745% |
| posterior-gv | 193/512 | 192/512 | 190/512 | 37.4349% |

2028 alone: parsed questions g 484, gv 483, ESS-g 478;
greedy hard-GV agreement conditional on parsing respectively
.845041, .836439, .841004. Thus posterior-gv no longer has
the highest grounding or consistency in seed 2028.

Seed-level differences in grounded/all-512 (percentage points):

| Contrast | 2026 | 2027 | 2028 | mean | sample SD |
|---|---:|---:|---:|---:|---:|
| GV - G | +1.3672 | +0.5859 | -0.9766 | +0.3255 | 1.1934 |
| GV - G-ESS | +0.3906 | +0.7813 | -0.3906 | +0.2604 | 0.5967 |

The downstream difference changes sign, while its mean is smaller
than its across-training-seed SD. Three seeds are too few for reliable
population inference; **no robust downstream GV superiority
is established**. Do not pool the three observations of each fixed
development question as if there were 1536 independent questions.
These are three *training* replicates on the SAME development set;
three-seed SD describes instability, not domain or population
generalization. The earlier positive prospective q_GV-vs-q_G
**inference/posterior NLL** result remains valid and is not
contradicted by this fine-tuning result.

The matched LoRA 2028 runtime was about 7:05-7:12 per training
arm and 12:23-12:31 per evaluation arm on the newer GPU. These
logs alone do not establish the exact device model.

#### Finish the seed audit (CPU-only, no regeneration or additional training)

    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_multiseed_cft.py \
      --run "2026:posterior-g=$ROOT/single-round/eval/posterior-g_dev_n512.jsonl" \
      --run "2026:posterior-gv=$ROOT/single-round/eval/posterior-gv_dev_n512.jsonl" \
      --run "2026:posterior-g-ess=$ROOT/single-round/eval/posterior-g-ess_dev_n512.jsonl" \
      --run "2027:posterior-g=$ROOT/single-round/eval/seed2027/posterior-g_dev_n512.jsonl" \
      --run "2027:posterior-gv=$ROOT/single-round/eval/seed2027/posterior-gv_dev_n512.jsonl" \
      --run "2027:posterior-g-ess=$ROOT/single-round/eval/seed2027/posterior-g-ess_dev_n512.jsonl" \
      --run "2028:posterior-g=$ROOT/single-round/eval/seed2028/posterior-g_dev_n512.jsonl" \
      --run "2028:posterior-gv=$ROOT/single-round/eval/seed2028/posterior-gv_dev_n512.jsonl" \
      --run "2028:posterior-g-ess=$ROOT/single-round/eval/seed2028/posterior-g-ess_dev_n512.jsonl" \
      --output "$ROOT/single-round/multiseed_posterior_cft_2026_2028.json"

Do not overwrite any earlier two-seed report or read the 1024-question
official_final_holdout. The three-seed audit is descriptive and
works from saved JSONL only.

#### Research decision after Stage 4

The protocol's confirmed contribution is the prospectively
evaluated **incremental predictive information** from a sequential
generator-to-validator measurement under an EM-proxy factual label.
The attempted implication from improved frozen q_GV prediction to
a reliable improvement through weighted paired-answer CFT **failed to
be established** in three LoRA seeds. Rather than continue identical
training, design a mechanistic ablation:

1. Generator-only versus validator-only versus both-component
   fine-tuning with matched posterior weighting *and matched training
   budget within each component comparison*. The existing trainer
   implements \`--components generator|validator|both\` but has not
   been re-evaluated in this factorial setting.
2. Evaluate direct factual QA separately from *paired-generator*
   EM correctness and GV validator agreement. The current generator
   target explicitly includes both intended-correct and misleading
   answers; higher pair-level grounding weights do not guarantee
   stronger ordinary QA.
3. Predeclare causal interpretations conservatively: component
   ablations of one shared LM reveal training-objective effects,
   not independently manipulated generator/validator modules.

Treat final official 1024-question holdout as *sealed*. Do not use it
simply to rescue weak development-set results or choose the best
seed. Any final holdout deployment needs a frozen selection and
reporting policy specified before running the evaluation.


### October 11, 2026: three-seed CFT audit executed and verified

Colab tests **69 passed**. \`scripts/audit_multiseed_cft.py\` read all
nine saved posterior CFT evaluation files, checked all **512 question
identities and ordering** across arms and training seeds, and produced
\`$ROOT/single-round/multiseed_posterior_cft_2026_2028.json\`.

Reported paired all-question grounded-pair EM differences:

| Contrast | seed 2026 gained/lost | seed 2027 gained/lost | seed 2028 gained/lost | mean +/- sample seed SD |
|---|---|---|---|---|
| GV - G | 13/6 | 11/8 | 2/7 | +0.33 +/- 1.19 pp |
| GV - G ESS matched | 10/8 | 12/8 | 5/7 | +0.26 +/- 0.60 pp |

Both contrasts favor GV in 2 of 3 seeds, and both reverse sign
for seed 2028. Differences are **across fixed, reused development
questions**, not nine independent samples of a question population.
With only three training seeds, sample seed SD has high uncertainty.
Device configuration also changed between earlier and later runs,
so precise hardware-controlled seed attribution should not be claimed.

**Stage-4 decision: completed.** Do not conduct additional
identical CFT seed runs merely to pursue a positive p-value, and
do not test the sealed official 1,024-question holdout to rescue
the development result. Preserved finding: frozen q_GV inferred
EM-proxy grounded-pair validity more accurately than q_G in the
prospective confirmation test. Not supported: an unambiguous
downstream advantage of q_GV-weighted CFT over equally-concentrated
q_G weights.

#### Proposed Stage-5 mechanistic ablation (NOT yet launched)

Predefine a **separate research question**: whether the diluted
downstream benefit is associated with optimizing both the paired-
answer generator completion and the A/B validator on a single
shared LM.

Use the frozen 320-question new training set and *development*
evaluation only. Compare \`--components generator\`,
\`--components validator\`, and \`--components both\` for the same
posterior method, holding training seed and optimizer updates
fixed and explicitly logging **different numbers of examples and
completion tokens**. Compare q_G with q_GV component-wise
if compute allows; prioritize actual factual direct-answer QA
instead of solely task-format-specific paired-answer EM. This
requires separate direct-QA evaluation implementation/prompt
freeze before training, with exact-match/alias grading.

Given both G and V share underlying parameters, training-
component ablation does not causally isolate modules. A stronger
diagnostic would evaluate the same generated answer pair with
**fixed alternate validator adapters**, and vice versa,
to separate answer-generation changes from validator-judgment
changes. This cross-role evaluation requires explicit model-
loading/swap support and is not currently guaranteed by
\`run_triviaqa_gv.py\` (which uses one \`--adapter\` for both).

Predeclare design, seeds and the decision rule before any new
experiment. Do not read \`official_final_holdout.jsonl\`
unless an independent final protocol is locked.
