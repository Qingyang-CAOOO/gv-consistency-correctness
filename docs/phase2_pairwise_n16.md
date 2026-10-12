# Phase 2 pilot: paper-style pairwise validator on frozen TriviaQA candidate pools

## Setup

- model: google/gemma-2-2b-it
- source candidate pools: phase2_base_n16_m8_t4_order_marginalized.jsonl
- usable questions: 10
- gold-covered questions: 6
- pairwise validator: original QA-style A/B comparison
- A/B order: marginalized over both positions

## Main result

| metric | generator prior | after pairwise validation |
|---|---:|---:|
| top-1 EM | 0.3000 | 0.2000 |
| gold mass | 0.2250 | 0.1754 |
| gold mass, covered only | 0.3750 | 0.2923 |
| ESS fraction | 0.7715 | 0.6193 |
| normalized entropy | 0.8460 | 0.6454 |

Mean covered-question delta gold mass is -0.08274, and only 1/3 of covered questions gain gold mass.

## Direct truth discrimination

- mean P(gold beats non-gold): 0.3289
- median P(gold beats non-gold): 0.3495
- fraction of gold/non-gold cross-pairs above 0.5: 0.300
- mean per-question cross-pair probability: 0.3935
- fraction of covered questions with mean cross-pair probability above 0.5: 0.167

In this small pilot, the pairwise validator is therefore anti-truth on the externally labeled answer pairs rather than merely uninformative.

## Generator-validator alignment

- mean pairwise potential of generator-top candidate: 0.5903
- mean pairwise potential of other candidates: 0.4726
- posterior top remains equal to generator-prior top on 70% of questions
- mean change in generator-top posterior mass: -0.0269

The validator therefore aligns with the generator's ranking in a relative sense: the generator-top answer receives a substantially higher validator potential and usually remains top-ranked. This should not be described as unconditional mass amplification, because the normalized mass of the prior top decreases slightly on average.

## Truth-conditioned effect

- mean delta gold mass when prior top is correct: -0.0418
- mean delta gold mass when prior top is wrong: -0.1237

Validation hurts both groups in this pilot, but the degradation is substantially larger when the generator's top answer is already wrong.

## Interpretation

The pilot supports separating internal consistency from external factual correctness. A useful latent-variable picture is

    T -> B_theta -> {Y_G, Y_V},

where T is external truth, B_theta is the model's latent answer preference/belief state, and generator and validator responses are two noisy readouts of B_theta. GV consistency measures agreement between readouts of B_theta; it is not by itself a likelihood for T.

Because only six questions contain a gold-equivalent candidate, this is diagnostic evidence rather than a stable population estimate. The next step is to scale the same frozen-pool, order-marginalized pairwise analysis before adding resampling or richer SMC machinery.