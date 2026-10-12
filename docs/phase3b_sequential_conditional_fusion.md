# Phase 3B: sequential conditional GV fusion

Phase 3A established on held-out TriviaQA test data that an order-marginalized pairwise validator
adds predictive information beyond generator prior odds. Phase 3B asks whether several semantically
equivalent validator measurements add *new* information sequentially.

For a candidate pair (i,j), define

    s_G,ij = log pi_0(z_i) - log pi_0(z_j),
    s_Vt,ij = logit P_Vt(z_i beats z_j).

Fit nested symmetric logistic models on TriviaQA train:

    q_0 = P(T_i > T_j | G),
    q_1 = P(T_i > T_j | G,V_1),
    ...
    q_t = P(T_i > T_j | G,V_1,...,V_t).

The sequential evidence from validator measurement t is

    delta_t,ij = logit q_t - logit q_{t-1}.

Each delta_t matrix is projected to candidate increments by the complete-graph least-squares map

    r_t,i = (1/K) sum_j delta_t,ij,

and the candidate weights update as

    pi_t(z_i) proportional to pi_{t-1}(z_i) exp(r_t,i).

This construction does not multiply raw validator probabilities and does not assume conditional
independence of the validator measurements. A validator template contributes only to the extent that
it improves prediction after conditioning on G and all earlier validator measurements.

Four pairwise validator prompt templates are evaluated, each with A/B order marginalized. Template 0
matches the Phase-3A prompt and can be reused from the existing pairwise files, so only templates
1--3 require new model calls.

Primary held-out diagnostics are the per-step improvements in question-averaged pairwise NLL and
Brier score, plus per-step changes in gold mass, top-1 EM, ESS, and entropy. Question-cluster
bootstrap intervals are reported for each incremental step and for the final cumulative change.

## Frozen N=512 result

Train split: 322 usable candidate pools; 128 pairwise-evaluable questions; 282 gold/non-gold
pairs. Test split: 347 usable pools; 162 gold-covered questions; 156 pairwise-evaluable questions.

Order-marginalized A/B sensitivity remains substantial across all four templates. Mean train order gaps were

    [0.38185, 0.29748, 0.30738, 0.30134]

and test order gaps were

    [0.35981, 0.27281, 0.27855, 0.25445].

### Train grouped-CV nested models

| stage | measurements | best CV NLL | incremental CV improvement |
|---|---|---:|---:|
| 0 | G | 0.58167 | -- |
| 1 | G,V1 | 0.47815 | +0.10352 |
| 2 | G,V1,V2 | 0.47959 | -0.00143 |
| 3 | G,V1,V2,V3 | 0.48129 | -0.00170 |
| 4 | G,V1,V2,V3,V4 | 0.48271 | -0.00142 |

Thus train grouped cross-validation selects the one-validator model. Additional paraphrased validators
do not improve out-of-fold log loss after conditioning on V1.

### Held-out sequential diagnostics

| step | gold-mass increment | 95% CI | pair-NLL improvement | 95% CI |
|---|---:|---:|---:|---:|
| V1 | +0.07727 | [0.04714, 0.10655] | +0.08440 | [0.02196, 0.14486] |
| V2 | +0.00224 | [0.00038, 0.00416] | +0.00354 | [-0.00005, 0.00705] |
| V3 | -0.00120 | [-0.00525, 0.00248] | -0.00615 | [-0.01441, 0.00162] |
| V4 | +0.00237 | [-0.00264, 0.00727] | -0.00140 | [-0.01352, 0.00992] |

The Brier improvement for V2 is +0.00157 with 95% CI [0.00012, 0.00300], while V3 and V4
are unresolved/non-positive. Top-1 increments after V1 are also unresolved and small.

The full four-step sequence ends with gold-mass delta +0.08067
(95% CI [0.04968, 0.11105]) and pair-NLL improvement +0.08039
(95% CI [0.01495, 0.14385]), but these cumulative gains are dominated by V1. The final model is
slightly worse in held-out pair NLL than the stage-1 model (0.47425 versus 0.47024).

### Interpretation and stopping rule

The evidence supports strong diminishing conditional information from repeated paraphrased validation.
V1 provides the dominant new signal. V2 may contain a very small residual signal by gold-mass/Brier
criteria, but its train-CV NLL is worse than stage 1 and its held-out NLL interval includes zero.
V3 and V4 add no resolved predictive value and can be harmful.

For a model-selection pipeline that is fixed before test evaluation, grouped train CV would stop at
one validator measurement. Therefore the test-set V2 signal should be treated as exploratory rather
than used to choose a longer sequence.

This phase does not show that *all* sequential measurements are redundant. It shows that repeated
semantically equivalent pairwise-validator paraphrases of the same model are mostly redundant after
the first measurement. A subsequent sequential study should change the measurement operator or
information set, rather than merely paraphrasing the same validator question.
