# Stage 5 supplementary audit: false retention by GV consistency

**Scope:** post-hoc, CPU-only analysis of saved data; no change to the
frozen Stage-5 experimental implementation or claim. The immutable
snapshot remains at commit
\`48d2d220aaa1d20bd374187dfea71007166301e7\`
(branch \`frozen/stage5-2026-10-11\`). The supplementary script is in
the active \`exp/triviaqa-soft-gv\` branch.

## Question from the original proposal

> Do correctness-informed posterior weights distinguish trajectories
> that hard GV consistency would retain incorrectly, and downweight
> the ungrounded mass carried by soft GV consistency?

Let \`D\` denote successful paired-answer parsing and \`T\` the
conservative EM grounding proxy

    T = 1{intended-correct EM=1 and intended-misleading EM=0}.

For each parsed generated pair, \`C=1\` if the single randomly
ordered greedy A/B validator chose the intended-correct position.
The hard-false-retention subgroup is

    F = {D=1, C=1, T=0}.

Its denominator is the hard-retained subgroup \`{D=1,C=1}\`,
not the set of all 512 questions or all parsed trajectories.
"Incorrect" here means ungrounded according to EM proxy, not
verified semantic falsehood.

For soft GV, **there is no binary retained subset**; it assigns a
positive continuous probability to every parsed pair. We measure
its total weight allocated to \`T=0\` and, separately, describe the
top quartile of sampled-order soft GV scores (ties included) as
a diagnostic subgroup, **not** a new CFT filtering decision.

## Saved sources and audit

The *primary observational population* is the 512 original
question-disjoint official confirmation questions. Of these, 394
were parsed and 177 were EM-grounded. Frozen predictions
\`frozen_confirmation_predictions.jsonl\` already contain
\`q_g\`, \`q_gv\`, the EM truth proxy and order-marginalized soft
consistency for **only parsed cases**. Join them by normalized
question ID to saved \`official_confirmation_features_n512.jsonl\`,
which also contains single-sampled-order hard and soft consistency.
The old official confirmation was inspected already. **All new
subgroup tests are explicitly retrospective**, not a second
independent confirmation or a predeclared primary analysis.

The **secondary dynamics population** is the fixed 320 CFT
training questions, five arms, rounds 1–3. R1 source is shared
among arms; later rounds have arm-specific generated trajectories.
For each population, join the raw EM-labeled generator/validator
records to their separately saved **gold-blind** frozen-posterior
weight records; check question alignment, parsed flags, candidate
text, no label leakage and constant frozen calibrator hashes.
These are **in-sample** concerning the trained adapter and cannot
support independent generalization claims.

Metrics:
- Number and fraction of EM-ungrounded *hard-retained* pairs.
- Within the hard-retained subgroup, difference in mean weight
  between grounded and ungrounded pairs for each selection method.
  Direct contrasts \`qGV-qG\` and \`qGV-soft sampled\` with question
  bootstrap percentile intervals.
- Posterior-weighted grounded precision inside the hard-retained
  subset, versus hard's own grounded precision.
- All-parsed weight-normalized fraction allocated to ungrounded
  trajectories (one minus weighted grounded precision) for
  hard GV, sampled soft GV, order-marginalized soft GV, qG,
  qG-ESS-matched, and qGV, with paired question-bootstrap
  differences \`qGV-baseline\`. The qG-ESS matching exponent is
  chosen **without labels** on the full confirmation population,
  then held fixed across bootstrap resamples; it is not calibrated.
- ESS and top-soft-quartile composition for context. Bootstrap
  resamples the fixed 394 parsed confirmation questions; intervals
  are unadjusted and condition on the trained model and parseability.

Run in **CPU** Colab with Drive mounted and repository bootstrapped:

    %%bash
    set -euo pipefail
    cd /content/gv-smc
    git pull --ff-only
    python -m pytest -q
    ROOT="/content/drive/MyDrive/gv-smc-artifacts/protocol-v2"
    python scripts/audit_false_retention.py \
      --root "$ROOT" --bootstrap 3000 --seed 2026 \
      --output "$ROOT/iterative-posterior-v1/seed2026/false_retention_posthoc.json"

The script refuses to overwrite an earlier result.
It does not open \`splits/official_final_holdout.jsonl\`,
load an LM, retrain a posterior, or use CUDA. Repeated
reads of the existing confirmation predictions are not
additional independent replications.

**Interpretation rule:** A positive conditional qGV truth
gap would mean the frozen posterior *distinguishes grounded
from ungrounded records within examples hard GV endorses*.
If its paired CI excludes zero in a non-predeclared subgroup,
treat that as exploratory only. It would not by itself prove
qGV-weighted **training** is better, nor isolate learning
dynamics causally. Stage 4 and Stage 5 already found no robust
downstream qGV-vs-qG fine-tuning advantage.


## October 11, 2026: final saved-data audit completed

**User-reported execution:** 80 pytest tests passed. The CPU-only
\`audit_false_retention.py\` completed and wrote

    $ROOT/iterative-posterior-v1/seed2026/false_retention_posthoc.json

Official confirmation cohort: 512 questions, 394 parsed, 177 exactly
grounded among parsed. The original holdout confirmation was already
analyzed in Stage 4; this new subgroup analysis is POST-HOC and
exploratory, not an independently pre-registered inference test.
The official 1,024-question *final* holdout remains sealed.

### Question 1: Does posterior inference recognize GV false retention?

A single-order greedy hard GV accepts 326/394 parsed question-pairs,
of which 162 are EM-grounded and **164 are not**. False retention
\`P(T=0 | C=1,D=1)=164/326=50.31%\`. Of all 217 parsed
EM-ungrounded pairs, hard GV retains 164. "Ungrounded" is a strict
TriviaQA alias exact-match proxy, not semantic falsehood.

| Weighting | Precision, weighted parsed pairs | Fraction of weight on ungrounded | ESS fraction |
|---|---:|---:|---:|
| hard GV | .4969 | .5031 | .8274 |
| soft GV single order | .4985 | .5015 | .8357 |
| order-marginalized soft GV | .5127 | .4873 | .8618 |
| q_G | .5832 | .4168 | .7302 |
| q_G ESS-matched | .5902 | .4098 | .7097 |
| q_GV | **.6073** | **.3927** | .7097 |

q_GV-minus-comparator weighted grounded precision (paired question
bootstrap 3,000 draws, fixed frozen official confirmation
predictions, no familywise correction, retrospective intervals):

| Comparator | Observed diff, pp | 95% percentile CI, pp |
|---|---:|---|
| hard GV | +11.04 | [+8.33,+13.71] |
| soft GV sampled | +10.88 | [+8.20,+13.52] |
| soft GV order-marginalized | +9.46 | [+6.99,+11.89] |
| q_G | +2.42 | [+1.15,+3.70] |
| ESS-matched q_G | +1.72 | [+0.45,+3.03] |

**Direct conditional diagnosis: within hard-accepted (\`C=1\`),
truth-related weight separation**, mean weight at T=1 minus T=0:

| Weight | Difference |
|---|---:|
| hard GV | 0 by definition |
| soft GV sampled | +.0195 |
| soft GV both-order marginalized | +.0981 |
| q_G | +.2179 |
| q_GV | +.2418 |

The incremental gap q_GV − q_G is +.0239, 95% paired
bootstrap CI [+ .0066,+.0412]. Versus sampled-order soft GV,
the extra separation gap is +.2223 [+.1758,+.2688].
Reweighting ONLY within hard-retained examples with q_GV gives
a *hypothetical* weighted grounded precision increase of
+.1312 (CI [.1053,.1571]) from hard retention's .4969
to approximately **.6281**. This is not a new checkpoint
or observed downstream CFT outcome.

For soft GV there is NO canonical binary retention decision.
As an exploratory high-score subgroup only, the upper
sampled-order soft-weight quartile (99 examples including ties)
contains 18 EM-ungrounded (18.18%). It cannot be compared
as equal-budget selection to q_GV's full-population weighted
precision without matching mass/ESS and data exposure.

### Secondary feedback diagnostics: 320 CFT TRAIN questions

Hard-false-retained trajectory counts among hard-accepted pairs
by arm and round:

| Arm | R1 | R2 | R3 |
|---|---|---|---|
| hard | 102/198 | 133/251 | 150/279 |
| soft | 102/198 | 131/253 | 140/279 |
| posterior-g | 102/198 | 128/247 | 142/268 |
| posterior-g-ess | 102/198 | 134/255 | 139/276 |
| posterior-gv | 102/198 | 132/253 | 143/275 |

The q_GV mean-weight grounded-minus-ungrounded gap *within
hard-retained pairs* persists in all five evolving branches,
including the GV-weighted branch:
R1 +.185, R2 +.161, R3 +.172.
However, the training populations themselves change; the
conditioned subgroups are selected based on the generator
and validator jointly; no counterfactual generator-vs-validator
mechanism can be isolated from this observational comparison.

### Integrated conclusion for the research report

**RQ part 1, conditional false-retention distinction: YES,
descriptively**, the truth-directed posterior separates
EM-grounded and EM-ungrounded cases *within* hard GV agreement
and reallocates less mass to ungrounded cases than hard/soft
consistency potentials (with qG-only and ESS-aware controls).
This conclusion is for one model and benchmark; new subgroup
confidence intervals are retrospective.

**RQ part 2, learning dynamics: no reliable posterior-weighted
downstream factual superiority was established.** Stage 4
three-seed single-round comparisons change sign; Stage 5
single-seed three-round pilot finds all arms increase grounded
yield, but q_GV round-3 = 213/512, versus soft 221/512,
hard 217/512, q_G 214/512 and q_G ESS 215/512. All round-3
q_GV-versus-baseline paired CIs cover 0.

Therefore: *correctness-informed GV measurements measurably
distinguish some consistency-endorsed errors, but their
additional selection information does not automatically
translate into better factual learning under the implemented
paired-answer CFT objective.* Do not claim a universal
impossibility, a proven self-confirmation cause, or that
selection precision improvements themselves are training
effects.

No experimental code or immutable freeze-branch edits
are needed; this is an appended supplementary analysis in
the active development branch only.
