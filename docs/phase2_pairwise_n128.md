# Phase 2: N=128 pairwise validator with question-cluster bootstrap

## Setup

- model: google/gemma-2-2b-it
- test questions requested: 128
- usable candidate pools: 85
- gold-covered usable questions: 42
- generator samples per question: 8
- mean unique candidates: 4.2235
- pairwise validator: paper-style QA comparison
- A/B order marginalized
- uncertainty: 10,000 question-cluster bootstrap replicates, seed 2026

## Point estimates

| metric | estimate |
|---|---:|
| initial top-1 EM | 0.28235 |
| pairwise top-1 EM | 0.27059 |
| initial gold mass | 0.23529 |
| pairwise gold mass | 0.23476 |
| covered initial gold mass | 0.47619 |
| covered pairwise gold mass | 0.47511 |
| mean covered delta gold mass | -0.00108 |
| initial ESS fraction | 0.77083 |
| pairwise ESS fraction | 0.64152 |
| mean pairwise order gap | 0.36168 |

Direct pairwise truth discrimination:

- pair-weighted mean P(gold beats non-gold): 0.62647
- question-weighted mean P(gold beats non-gold): 0.60049
- fraction of cross-pairs above 0.5: 0.645
- fraction of covered questions with mean cross-pair probability above 0.5: 0.600

Generator-validator concordance:

- mean P(validator picks generator-prior-favored candidate): 0.58187
- correlation between prior weight difference and validator preference difference: 0.17472
- P(gold wins | generator favors gold): 0.61978
- P(gold wins | generator favors wrong): 0.52441

## Question-cluster bootstrap

| quantity | estimate | 95% CI |
|---|---:|---:|
| pair_truth_prob | 0.62647 | [0.50765, 0.74165] |
| question_truth_prob | 0.60049 | [0.49462, 0.70816] |
| gold_mass_delta | -0.00108 | [-0.06848, 0.06009] |
| top1_em_delta | -0.01176 | [-0.08235, 0.05882] |
| delta gold mass, prior top correct | -0.02609 | [-0.13099, 0.06686] |
| delta gold mass, prior top wrong | 0.03227 | [-0.03442, 0.10137] |
| wrong-minus-correct delta | 0.05836 | [-0.05815, 0.18518] |
| GV concordance | 0.58187 | [0.52489, 0.63954] |
| P(gold wins | G favors gold) | 0.61978 | [0.50123, 0.74852] |
| P(gold wins | G favors wrong) | 0.52441 | [0.35275, 0.74505] |
| V follows G | G favors wrong | 0.47559 | [0.25495, 0.64725] |
| follow-gap, G-gold minus G-wrong | 0.14418 | [-0.06301, 0.39341] |

## Interpretation

The bootstrap weakens the stronger interpretation suggested by the raw N=128 point estimates.

1. Pair-weighted truth discrimination is above chance with a bootstrap interval just excluding 0.5, but the question-weighted estimand includes 0.5. Thus evidence that the validator is truth-informative is suggestive, not yet robust at the question-population level.
2. Overall generator-validator concordance is more stable: its 95% interval lies above 0.5.
3. There is no resolved improvement in top-1 EM or gold mass after validation.
4. The apparent complementarity pattern -- validation helping when the generator top is wrong and hurting when it is correct -- is not statistically resolved; all relevant contrast intervals include zero.
5. The correct next step is replication / increased question count under the same frozen design, not additional SMC machinery.

These results do not justify treating the validator potential as a calibrated factual likelihood.