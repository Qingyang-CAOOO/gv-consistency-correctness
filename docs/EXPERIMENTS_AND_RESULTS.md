# Final experiments and results

This document is the **concise, completed-experiment index**. It does not
supersede the detailed records in [`protocol_v2_truth_audit.md`](protocol_v2_truth_audit.md),
[`stage5_iterative_cft_v1.md`](stage5_iterative_cft_v1.md), or
[`stage5_false_retention_posthoc.md`](stage5_false_retention_posthoc.md).
Those older notes include chronological work-in-progress entries: read the
*latest completed results*, not earlier planned statuses.

## Research questions

1. Do supervised-calibrated factual-correctness weights distinguish EM-ungrounded
   trajectories that hard or soft generator–validator consistency favors?
2. Does the selection distinction yield reliably better factual learning in
   matched consistency fine-tuning (CFT), rather than only increased agreement?

## Fixed model and task

- Model: `google/gemma-2-2b-it`, shared for generation and validation.
- Task: closed-book TriviaQA; generate one intended-correct and one
  intended-misleading answer. Validate a randomized A/B presentation.
- Factual proxy: parsed, intended-correct answer matches a gold alias,
  and intended-misleading answer does not. This is **not** semantic truth.
- Five CFT weights: hard greedy GV, sampled-order soft GV, generator-only
  calibrated posterior `qG`, ESS-matched generator-only control, and
  generator+both-order-validator calibrated posterior `qGV`.
- Posterior models: fitted on designated base-model calibration training
  examples with gold supervision, frozen before confirmation and CFT. They
  are **not** learned by EM, and `qG-ESS` is not a calibrated posterior.
- CFT: same 320 training questions, 512 repeatedly inspected development
  questions, 2 epochs, 82 optimizer steps, LoRA rank 8, alpha 16,
  dropout 0.05. Same question cohort and evaluation protocol, not identical
  selected-example count or FLOPs.
- Seed robustness: three LoRA training seeds for `qG`, `qG-ESS` and `qGV`
  **in the single-round stage only**.
- Iterative dynamics: all five arms for three rounds at training seed
  **2026**; each round uses its own previous checkpoint to generate/score
  new data and re-trains a fresh LoRA from the common base model on just
  that round's weighted examples. The posterior estimators remain frozen.

## Experiment A: frozen correctness inference

Independent 512-question official *confirmation* split, 394 successfully
parsed question-pairs, 177 grounded. This original prospective comparison
was specified before looking at confirmation outcomes; the subsequent
false-retention subgroup analysis described below was **post-hoc**.

| Metric, parsed confirmation pairs | qG | qGV |
|---|---:|---:|
| Negative log-likelihood | 0.597160 | **0.563113** |
| Brier score | 0.201548 | **0.188140** |
| AUROC | 0.753235 | **0.786560** |
| Selection-weighted grounded precision | 0.583160 | **0.607329** |

qGV minus qG paired question-bootstrap NLL difference:
**-0.034047**, 95% CI [-0.057660, -0.011543].
These are correctness-*prediction* gains, not downstream training effects.

## Experiment B: post-hoc GV false-retention audit

Among 394 parsed confirmation pairs, greedy-hard GV accepted 326; **164
of those 326 (50.31%)** failed the EM grounding proxy.

| Selection weight | Weighted grounded precision | Mass on EM-ungrounded pairs |
|---|---:|---:|
| hard GV | 49.69% | 50.31% |
| soft GV, one order | 49.85% | 50.15% |
| soft GV, both-order mean | 51.27% | 48.73% |
| qG | 58.32% | 41.68% |
| ESS-matched qG | 59.02% | 40.98% |
| qGV | **60.73%** | **39.27%** |

qGV-minus-hard grounded-precision difference: **+11.04 percentage points**,
paired retrospective CI [+8.33,+13.71] pp. qGV-minus-ESS-matched qG:
**+1.72 pp**, CI [+0.45,+3.03] pp.

Within the **hard-accepted** subset, mean weight on grounded minus
ungrounded pairs is **0** for hard GV, **+0.0195** for soft GV,
**+0.2179** for qG and **+0.2418** for qGV. This directly tests whether
a correctness estimator can resolve false retention by hard agreement.
The subgroup tests were not pre-registered and the already inspected
confirmation set was reused; their intervals are exploratory.

## Experiment C: one-round CFT and seed variation

All methods use the same fixed CFT training questions and development
evaluation cohort. Seed-2026 first-round grounded yields:
hard **191/512**, soft **184/512**, qG **186/512**,
qG-ESS **191/512**, qGV **193/512**.

Three independent optimizer seeds were run for the posterior arms:
qG grounded counts **[186,189,195]**,
qG-ESS **[191,188,192]**,
qGV **[193,192,190]**.
The qGV-qG advantage changes sign across seeds
(**[+7,+3,-5] /512**); its mean is only
**+0.33 percentage points**. This is not evidence of robust
posterior-weighted CFT superiority.

## Experiment D: three feedback rounds

Grounded-pair **end-to-end yield** on the *same* 512 development
questions, one training seed (2026):

| Method | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| Hard GV CFT | 191 (37.30%) | 214 (41.80%) | 217 (42.38%) |
| Soft GV CFT | 184 (35.94%) | 212 (41.41%) | **221 (43.16%)** |
| qG CFT | 186 (36.33%) | 202 (39.45%) | 214 (41.80%) |
| qG-ESS CFT | 191 (37.30%) | 205 (40.04%) | 215 (41.99%) |
| qGV CFT | 193 (37.70%) | 210 (41.02%) | 213 (41.60%) |

Round-3 qGV-minus-hard = **-0.78 pp**, retrospective paired 95% CI
[-2.73,+1.17] pp; versus soft = **-1.56 pp**, CI
[-3.52,+0.39] pp; versus qG = **-0.20 pp**, CI
[-1.76,+1.18] pp. All include zero. Seed-2026 alone is
**not** a training-seed replication. Improvements in parse compliance
contribute to end-to-end factual yield; do not conflate this with
improved factual quality conditional on parsing.

Exploratory Stage-5 training-cohort audits show qGV's
grounded-minus-ungrounded weight separation within hard-retained
examples remains positive in all rounds, but does **not** establish
that inference quality translates into better generation after CFT.

## Interpretation and limitations

**Main finding:** correctness-oriented weighting better distinguishes
EM-grounded from EM-ungrounded examples that agreement-based methods
would retain, but the posterior's extra selection signal does not yield
a *reliably superior* downstream CFT learner in the investigated setup.

The saved data do **not** establish why that transfer fails. In particular:
- No crossed-generator/validator-checkpoint evaluation isolates the
  validator's co-adaptation or self-confirmation mechanism.
- qGV/qG were calibrated on base-model data but applied without
  refitting to changing adapter-generated distributions.
- The original method involves supervised-calibrated posterior weighting,
  not label-free Bayesian inference, exact EM, or full SMC.
- Developer choices and CIs involve repeatedly analyzed data and small
  numbers of training seeds.
- The reserved **1,024-question final holdout was never opened**.

The repository contains the analysis code and narrative, but not
the original saved trajectories, fitted posterior models, or fine-tuned
checkpoints. For exact commands, provenance conditions and artifact
requirements consult [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
