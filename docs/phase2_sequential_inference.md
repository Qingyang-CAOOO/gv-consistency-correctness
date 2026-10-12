# Phase 2: sequential latent-hypothesis inference

## Goal

Move from using GV consistency as a fine-tuning weight to using it as sequential evidence over a finite answer-hypothesis space.

For a TriviaQA question x, define a static latent hypothesis

    Z in {z_1, ..., z_K},

where each z_k is a distinct normalized answer candidate sampled from the frozen generator.

Multiple generator samples that collapse to the same normalized answer define the initial generator measure

    pi_0(z_k) = count(z_k) / sum_j count(z_j).

This is an empirical approximation to the generator's answer distribution. It is not claimed to be the factual posterior.

## Sequential measurement

At round t, each candidate is queried with a randomized-label correctness validator. The label position corresponding to 'candidate is correct' is randomized between A and B. For candidate z_k, define

    G_t(z_k) = P_theta(Y_{V,t} = R_{t,k} | x, z_k, q_t),

where q_t is the round-specific validator template and R_{t,k} is the randomized target label.

The static-state sequential importance update is

    pi_t(z_k) proportional to pi_{t-1}(z_k) G_t(z_k),

or equivalently

    pi_t(z_k) proportional to pi_0(z_k) product_{s=1}^t G_s(z_k).

These pi_t are Feynman--Kac consistency measures. They should not yet be called calibrated Bayesian factual posteriors: the validator templates are correlated self-measurements, and no conditional-independence/calibration model has been established.

## Why SIS before full SMC

The state is static and the initial candidate support is finite. Resampling without a mutation/rejuvenation kernel would only duplicate existing answers and cause particle impoverishment. Therefore Phase 2 first runs sequential importance sampling (SIS), tracking ESS and entropy but not resampling.

Phase 3 will introduce a resample-move step only after defining a valid mutation/revision proposal for answer hypotheses.

## Measurements and diagnostics

For each question, record:

- gold-candidate coverage of the sampled hypothesis pool;
- top-1 exact match at t=0 and after each validator round;
- posterior/consistency mass assigned to gold-equivalent candidates;
- ESS fraction and normalized entropy after each round;
- maximum particle weight and the complete per-round weight vector.

Ground truth is used only for evaluation, never for the sequential weight update.

## Pilot

Start with Gemma-2-2B-IT, test split, 16 questions, 8 generator samples per question, and four distinct validator templates. If the sequential potentials are useful, gold mass and/or top-1 EM should increase before ESS collapses. If ESS collapses without accuracy gains, repeated self-validation is sharpening confidence without adding truth-relevant information.
