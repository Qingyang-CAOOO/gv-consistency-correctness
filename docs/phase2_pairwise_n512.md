# Phase 2: N=512 pairwise validator with question-cluster bootstrap

## Setup

- model: google/gemma-2-2b-it
- test questions requested: 512
- usable candidate pools: 347
- gold-covered usable questions: 162
- generator samples per question: 8
- mean unique candidates: 4.1931
- pairwise validator: paper-style QA comparison
- A/B order marginalized
- uncertainty: 10,000 question-cluster bootstrap replicates, seed 2026

## Main point estimates

| metric | generator prior | after pairwise validation |
|---|---:|---:|
| top-1 EM | 0.29107 | 0.30259 |
| gold mass | 0.23415 | 0.26697 |
| gold mass, covered only | 0.50154 | 0.57184 |
| ESS fraction | 0.78627 | 0.63820 |
| normalized entropy | 0.85090 | 0.64548 |

- mean covered delta gold mass: +0.07030
- fraction of covered questions with positive delta gold mass: 0.7222
- mean pairwise A/B order gap: 0.35981

## Truth discrimination

- pair-weighted mean P(gold beats non-gold): 0.68686
- question-weighted mean P(gold beats non-gold): 0.69268
- fraction of gold/non-gold pairs above 0.5: 0.703
- fraction of covered questions with question-level truth preference above 0.5: 0.750

Question-cluster bootstrap:

| quantity | estimate | 95% CI |
|---|---:|---:|
| pair truth probability | 0.68686 | [0.63289, 0.73879] |
| question truth probability | 0.69268 | [0.64190, 0.74217] |
| covered gold-mass delta | +0.07030 | [0.02940, 0.11008] |
| top-1 EM delta | +0.01153 | [-0.02306, 0.04611] |

Thus the validator has resolved truth-discriminative signal at the question level, and the pairwise reweighting increases gold mass on average. The top-1 EM gain is not statistically resolved.

## Generator-validator coupling

- mean P(validator picks generator-prior-favored candidate): 0.56873
- 95% CI: [0.54035, 0.59810]
- correlation between prior-weight difference and validator preference difference: 0.21069

Conditioned on the generator's pairwise preference:

| condition | P(validator chooses gold) | 95% CI |
|---|---:|---:|
| generator favors gold | 0.68408 | [0.62101, 0.74610] |
| generator favors wrong | 0.67008 | [0.57631, 0.76086] |

When the generator favors the wrong candidate, the validator follows that wrong preference with probability only 0.32992, 95% CI [0.23914, 0.42369].

The difference in validator-following-generator probability between generator-correct and generator-wrong pairwise preferences is 0.35416, 95% CI [0.23917, 0.46584]. This is strong evidence that GV agreement is correctness-dependent rather than a uniform tendency for the validator to echo the generator.

## Truth-conditioned reweighting

| condition | mean delta gold mass | 95% CI |
|---|---:|---:|
| prior top correct | +0.03159 | [-0.02172, 0.07998] |
| prior top wrong | +0.13439 | [0.07417, 0.19795] |
| wrong-minus-correct | +0.10280 | [0.02434, 0.18580] |

The validator-driven update helps significantly when the generator's top hypothesis is wrong, and the improvement is significantly larger than when the generator top is already correct.

## Interpretation

The N=512 replication resolves the ambiguity of the N=128 pilot. The pairwise validator is both generator-coupled and independently truth-informative. It should therefore not be modeled as either an independent truth oracle or a pure echo of the generator.

A useful inference view is to treat the generator prior and validator comparisons as dependent measurements of the latent answer hypothesis. The current pairwise score is still not a calibrated likelihood: candidate comparisons within a question are dependent, A/B order sensitivity remains large before marginalization, and the same LM produces both measurements.

The next methodological step should therefore be calibrated / tempered fusion of generator and validator evidence before introducing repeated sequential products or resampling.