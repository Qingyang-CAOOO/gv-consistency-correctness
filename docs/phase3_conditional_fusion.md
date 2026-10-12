# Phase 3A: conditional generator-validator evidence fusion

## Motivation

The N=512 Phase-2 result shows that the order-marginalized pairwise validator is truth-informative and generator-coupled, while raw validator multiplication is not a calibrated Bayesian update. Phase 3A therefore estimates how much predictive information validation adds conditional on the generator prior.

## Pairwise calibration model

For a candidate pair (i,j), define generator and validator log-odds

    s_G,ij = log pi_0(z_i) - log pi_0(z_j),
    s_V,ij = logit P_V(z_i beats z_j),

where the validator probability has already marginalized A/B order.

Fit two symmetric logistic models on TriviaQA train gold/non-gold pairs:

    q_0(i>j) = sigmoid(a_0 s_G,ij),
    q_1(i>j) = sigmoid(a_G s_G,ij + b_V s_V,ij).

The training examples are duplicated in both orientations and the models have no intercept, so q(j>i)=1-q(i>j). Each covered question receives total weight one. Regularization C is selected by grouped cross-validation with all pairs from a question kept in the same fold.

The conditional validator contribution is

    delta_ij = logit q_1(i>j) - logit q_0(i>j).

A positive held-out improvement in pairwise log loss or Brier score is direct evidence that V adds predictive information beyond G.

## Candidate-level fusion

The pairwise incremental log-odds are projected to candidate increments on the complete graph:

    r_i = (1/K) sum_j delta_ij.

This is the least-squares solution, under a zero-sum gauge, to r_i-r_j approximately delta_ij. The original generator prior is then updated only by the incremental validator evidence:

    pi_fused(z_i) proportional to pi_0(z_i) exp(r_i).

This construction preserves pi_0 exactly when the fitted validator contribution is zero.

## Important scope

This is supervised calibration because TriviaQA train labels are used to fit the evidence model. It is not the unlabeled CFT setting of the original GV paper. It is also not claimed to be an exact Bayesian posterior: the fitted model is discriminative and the multi-candidate distribution is induced by a pairwise projection.

## Held-out test criteria

On the frozen N=512 TriviaQA test candidate pools, evaluate:

- baseline versus full pairwise question-averaged log loss;
- baseline versus full pairwise Brier score;
- fused versus generator-prior gold mass;
- fused versus generator-prior top-1 EM;
- question-cluster bootstrap confidence intervals for all improvements.

Only after held-out conditional calibration succeeds should repeated sequential fusion or SMC resampling be introduced.

## Frozen Phase 3A result

Training used the TriviaQA train split with 512 requested questions, 322 usable candidate pools,
134 gold-covered pools, and 128 pools containing at least one gold and one non-gold candidate.
The pairwise calibration fit therefore used 128 pairwise-evaluable questions and 282 gold/non-gold
pairs (564 symmetrically oriented examples).

The fitted models were

    q_0(i>j) = sigmoid(0.83261 s_G,ij),

    q_1(i>j) = sigmoid(0.75989 s_G,ij + 0.24521 s_V,ij).

Grouped 5-fold CV selected C=1 for both models. Train-CV log loss decreased from 0.58167
(generator only) to 0.47815 (generator + validator), an improvement of 0.10352.

Held-out evaluation used the frozen N=512 test candidate pools: 347 usable questions,
162 gold-covered questions, and 156 pairwise-evaluable questions.

| held-out quantity | estimate | 95% question-cluster bootstrap CI |
|---|---:|---:|
| pairwise NLL improvement, G - (G+V) | +0.08440 | [0.02196, 0.14486] |
| pairwise Brier improvement, G - (G+V) | +0.03101 | [0.00531, 0.05610] |
| covered gold-mass delta | +0.07727 | [0.04714, 0.10655] |
| gold-mass delta, prior top correct | +0.04478 | [0.00887, 0.07889] |
| gold-mass delta, prior top wrong | +0.13106 | [0.08120, 0.18074] |
| top-1 EM delta | +0.01729 | [-0.01729, 0.05187] |

Pairwise classification accuracy improved from 0.64818 to 0.78524.
Covered gold mass increased from 0.50154 to 0.57881. Top-1 EM increased from
0.29107 to 0.30836, but that MAP improvement is not statistically resolved.

The calibrated fusion is also less aggressive than the raw pairwise potential update:
on the same test pools, raw pairwise weighting had ESS fraction 0.63820 and normalized entropy
0.64548, whereas calibrated fusion has ESS fraction 0.66360 and normalized entropy 0.73101
(the generator prior is 0.78627 and 0.85090 respectively).

### Conclusion

Phase 3A is frozen as a successful held-out conditional-prediction result. The validator provides
incremental predictive information beyond generator prior odds under this supervised calibration
protocol. This is operational evidence of conditional predictive value; it is not a direct estimate
of conditional mutual information and does not make the fused candidate distribution an exact
Bayesian posterior.

The next phase should test *sequential conditional increments*: for validator measurements
V_1,...,V_T, fit nested models q_t(T | G,V_1,...,V_t) and update with only

    delta_t = logit q_t - logit q_{t-1}.

This conditions each new validator measurement on all previous measurements and therefore avoids
the independence assumption implicit in multiplying raw validator scores.
