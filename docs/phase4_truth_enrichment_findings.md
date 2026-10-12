# Truth-enrichment audit: iterative GV CFT, TriviaQA development N=512

Observed user-side Colab run on frozen iterative-cft artifacts, October 10, 2026.
All factual measures are conservative TriviaQA exact-match proxies on parseable generated pairs.
Development-set evidence, not an untouched external holdout.

| branch | iter | parse | grounded given parsed | hard selected grounded precision | soft selected grounded precision | LR hard | LR soft | ESS soft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | .7461 | .4058 | .4777 | .4742 | 1.3393 | 1.3210 | .7764 |
| hard | 1 | .9688 | .3972 | .4543 | .4516 | 1.2636 | 1.2501 | .8115 |
| hard | 2 | .9707 | .4165 | .4733 | .4725 | 1.2589 | 1.2548 | .8397 |
| hard | 3 | .9863 | .4238 | .4596 | .4579 | 1.1567 | 1.1487 | .8929 |
| soft | 1 | .9219 | .3962 | .4517 | .4503 | 1.2555 | 1.2482 | .8238 |
| soft | 2 | .9668 | .4020 | .4437 | .4428 | 1.1862 | 1.1820 | .8718 |
| soft | 3 | .9883 | .4111 | .4447 | .4429 | 1.1473 | 1.1390 | .8875 |

Definitions: T=1 if designated correct answer matches a gold alias and designated misleading answer does not.
D=1 if generator response is successfully parsed.
LR hard = E[C | T=1,D=1] / E[C | T=0,D=1].
LR soft = E[G | T=1,D=1] / E[G | T=0,D=1].
Selected precision = sum weight*T / sum weight among parsed pairs.
Hard C is the greedy validator's agreement, not a sampled Bernoulli label.
ESS reflects weight uniformity, not answer diversity.

Observed: greater consistency and parse compliance coexist with weaker discrimination between
grounded and nongrounded generated pairs. The ratio stays above 1 in every case.
Hard selection precision lift over prevalence declines from +.0719 base
to +.0358 at hard iteration3 and +.0336 at soft iteration3.

Interpretation: consistent with self-confirmation, not a causal demonstration.
Generated pair distributions change across checkpoints; labels are EM proxies.
Rerun the CPU-only audit after commit 1e182ef to collect paired question-bootstrap
intervals for baseline-to-iteration changes in enrichment ratio and selection lift.
A subsequent crossed generator-checkpoint/validator-checkpoint experiment is
needed to disentangle generator distribution shift and co-adaptation.


## Paired question-bootstrap findings (October 10, 2026)

All 38 Colab tests passed before the new within-checkpoint order comparison
was added. The base-to-iteration-3 differences are based on 5,000 bootstrap
resamples of the 512 aligned *questions*, not independently resampled responses.

| Base minus iter3 metric | Hard branch (95% CI) | Soft branch (95% CI) |
|---|---|---|
| Hard-potential enrichment ratio | +0.18257 [0.06502, 0.31697] | +0.19194 [0.06904, 0.33116] |
| Soft-potential enrichment ratio | +0.17235 [0.06343, 0.29823] | +0.18197 [0.06628, 0.31222] |
| Hard precision lift over prevalence | +0.03603 [0.01273, 0.06016] | +0.03828 [0.01390, 0.06364] |
| Soft precision lift over prevalence | +0.03433 [0.01271, 0.05727] | +0.03664 [0.01363, 0.06040] |

The CIs exclude zero for all eight comparisons (without multiplicity
correction). This supports the within-development-set deterioration of
truth-discriminative selection metrics under repeated CFT. It does **not**
establish that fine-tuning caused the change through validator coadaptation,
because generated answer pairs also change by checkpoint.

Factual pair-state counts on parsed questions:

| checkpoint | desired 10 | reversed 01 | both true 11 | both EM false 00 | parsed |
|---|---:|---:|---:|---:|---:|
| base | 155 | 21 | 6 | 200 | 382 |
| hard iter3 | 214 | 25 | 6 | 260 | 505 |
| soft iter3 | 208 | 25 | 9 | 264 | 506 |

The end-to-end desired pair yield (all 512 questions as denominator) rises
from 30.3% (base) to 41.8% (hard iteration3) and 40.6% (soft iteration3).
The approximately 52% both-false rate applies only to the parsed subset
and uses conservative exact-match labels, not semantic adjudication.

## Immediate next analysis

The existing evaluation outputs already contain A/B-paired validator
probabilities because the iterative runner enabled paired-order diagnostics.
Use the updated CPU audit to estimate **within-checkpoint, paired** differences
between the original sampled-order soft potential and the explicitly
order-marginalized potential, including confidence intervals for grounding
precision, enrichment ratio and ESS. This is possible *before* any new GPU
training or rescoring of training trajectories.

The next GPU task, only after that analysis, is to augment the frozen
historical training trajectories with both validator orders, then proceed to
calibrated correctness inference on the newly frozen question-disjoint split.

Caveat: as both grounded and nongrounded acceptance approach 1, the
enrichment ratio necessarily approaches 1; report both conditional
acceptance rates and their **difference** alongside the ratio. These
diagnostics alone cannot distinguish saturation from semantic loss of
discrimination. Counterfactual crossed generator/validator checkpoints
are a later mechanism test.


## Order-marginalization findings (October 10, 2026)

On all seven frozen N=512 development checkpoints, question-bootstrap 95%
intervals for the difference in truth-enrichment ratio between
order-marginalized and sampled-order soft potential include zero. The same
is true of grounded-pair selection precision. Key examples:

| checkpoint | delta grounded precision | 95% CI | delta enrichment ratio | 95% CI |
|---|---:|---|---:|---|
| base | -.00666 | [-.02078, .00648] | -.03482 | [-.11304, .03348] |
| hard iter3 | +.00310 | [-.00424, .01047] | +.01442 | [-.02001, .04842] |
| soft iter3 | +.00428 | [-.00311, .01157] | +.01993 | [-.01477, .05325] |

This does not demonstrate equivalence or rule out small gains in training:
within-population factual selection and the downstream fine-tuning outcome
are different endpoints. The principled one-round order-marginalized control
remains available, but full iterative training is not warranted by these
selection metrics alone.

The frozen historical base-training data have now been scored in both
orders: 512 questions, 401 parsed, 401 rescored, mean absolute A/B gap
.214862. This is meaningful positional sensitivity but **does not
establish a factual-selection improvement** after marginalizing order.

The next no-GPU analysis is:

    python scripts/audit_frozen_gv_population.py \
      --data "/content/drive/MyDrive/gv-smc-artifacts/protocol-v2/base_train_both_orders_n512.jsonl" \
      --bootstrap 5000

Then move to collecting calibration-train and calibration-validation paired
GV data for correctness-posterior fitting, on the disjoint frozen question
sets; avoid further expensive iterated order-marginalized CFT for now.

Progress reporting: scripts/audit_iterative_truth_enrichment.py and
scripts/audit_frozen_gv_population.py now show overall bootstrap-draw
progress with elapsed time and ETA; scripts/train_cft.py shows optimizer
updates and ETA; generator inference and both-order scoring have per-sample
progress. In Colab these durations are estimates from observed iteration
speed and exclude initial model loading and artifact saving.
