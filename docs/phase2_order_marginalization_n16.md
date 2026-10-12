# Phase 2 pilot: order marginalization on TriviaQA (N=16)

## Setup

- model: google/gemma-2-2b-it
- split: test
- questions: 16
- generator samples per question: 8
- sequential validator rounds: 4
- order handling: marginalized
- usable questions: 10/16
- mean unique candidates: 4.3
- gold-candidate coverage among usable questions: 0.6

For each candidate z and validator template q, the A/B nuisance variable is integrated out:

    G_tilde_q(z) = 0.5 * G_q(z, R=A) + 0.5 * G_q(z, R=B).

This removes the high-variance per-particle effect of a single randomized label placement.

## Sequential results

Step t=0 is the generator prior.

| t | top-1 EM | gold mass | gold mass | covered | ESS fraction | normalized entropy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.30 | 0.2250 | 0.3750 | 0.7715 | 0.8460 |
| 1 | 0.30 | 0.2298 | 0.3829 | 0.7659 | 0.8310 |
| 2 | 0.30 | 0.1885 | 0.3141 | 0.6856 | 0.7396 |
| 3 | 0.20 | 0.1810 | 0.3017 | 0.6787 | 0.7195 |
| 4 | 0.20 | 0.1735 | 0.2892 | 0.6490 | 0.6778 |

Order marginalization largely removes the catastrophic weight collapse seen with one sampled A/B assignment. The first template is nearly neutral/slightly beneficial, but later templates still reduce gold mass while ESS and entropy decline.

## Independent one-step template audit

Each validator template was also applied once to the original generator prior, removing contamination from earlier sequential updates.

| template | mean delta gold mass | median delta | fraction positive | mean log evidence gold/non |
|---:|---:|---:|---:|---:|
| 1 | +0.007919 | +0.000004 | 0.500 | +0.088560 |
| 2 | -0.068853 | -0.001699 | 0.333 | -0.666358 |
| 3 | -0.021451 | -0.001305 | 0.333 | -0.104218 |
| 4 | -0.069737 | -0.010491 | 0.333 | -0.585456 |

The earlier apparent advantage of template 3 under sampled A/B ordering disappears after marginalization. Thus it was largely induced by label-position allocation rather than a robust semantic truth signal.

## Interpretation

1. Randomized A/B order is sufficient to neutralize constant-position shortcuts in aggregate GV evaluation, but a single realized order is too noisy for low-particle sequential weighting.
2. Marginalizing R is therefore the appropriate potential for per-particle sequential inference.
3. After removing order bias, the remaining validator signal is weak and, for three of four templates in this small pilot, mildly anti-truth.
4. Because only six usable questions contain a gold-equivalent candidate, these signs are diagnostic rather than statistically conclusive.
5. The next experiment should increase the number of covered questions before introducing resampling or more complicated SMC machinery.

These measures remain Feynman--Kac consistency measures, not calibrated factual posteriors.