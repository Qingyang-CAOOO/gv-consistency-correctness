# TriviaQA CFT training-data generation: Gemma-2-2B-IT, N=512

Frozen source:
- model: `google/gemma-2-2b-it`
- split: RankAlign TriviaQA train
- requested questions: 512
- data/generation/order/validator-sample seeds: 142/143/144/145
- generator: greedy
- validator: one balanced randomized A/B order
- paired-order diagnostic: disabled for training-data generation

## Result

- parsed trajectories: 401/512 = 0.783203
- parse modes:
  - delimiter line: 326
  - labeled lines: 75
  - incomplete delimiter pair: 65
  - ambiguous multiple delimiters: 33
  - unrecognized format: 13
- intended-correct exact match: 0.453865
- intended-misleading exact match: 0.069825
- grounded pair fraction: 0.446384
- hard GV consistency: 0.810474
- mean soft GV consistency: 0.808153
- sampled-hard Monte Carlo mean: 0.808316
- sampled-hard MC minus soft mean: 0.000163
- soft ESS: 330.57 / 401 = 0.824352
- target A fraction, all: 0.500000
- target A fraction, parsed: 0.493766
- hard consistency, target A: 0.939394
- hard consistency, target B: 0.684729
- grounded pair count: 179
- validator hard accuracy on grounded pairs: 0.927374
- mean soft score on grounded pairs: 0.922270

## Hard/soft training mass

Hard CFT retains 325 of the 401 parsed trajectories.

Soft CFT retains all 401 parsed trajectories, but the total soft mass is

[
sum_i G_i = 401 \times 0.8081528544 \approx 324.07.
]

This is unusually well matched to the 325 hard-retained trajectories. It gives a useful comparison: the two methods have nearly equal total consistency mass, while hard CFT collapses the mass to a binary subset and soft CFT distributes it continuously across all parsed trajectories.

Among the 222 non-grounded parsed trajectories, 159 are still hard GV-consistent:

[
159/222 \approx 0.7162.
]

The corresponding mean soft score on the non-grounded subset is also approximately 0.7161.

## Training convention

Both CFT methods use the exact same frozen JSONL.

- Hard CFT: retain only (C_i=1) trajectories and train generator + validator examples with unit weight.
- Soft CFT: retain every parsed trajectory and weight both generator + validator examples by (G_i).
- Generator target: the raw generated response (y_G).
- Validator target: the randomized consistency target (R). On the hard-retained set this equals the realized consistent validator label; under soft weighting it is the target label in the marginal (C=1) change-of-measure objective.
- TriviaQA exact-match labels are never used for training.

The weighted trainer globally normalizes by the dataset mean weight rather than renormalizing each minibatch independently.
