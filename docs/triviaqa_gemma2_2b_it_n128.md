# TriviaQA generator-coupled GV validation: Gemma-2-2B-IT, N=128

Configuration is frozen in `configs/triviaqa_gemma2_2b_it_generated_gv.yaml`.

## Result

- questions: 128
- parsed generator pairs: 98/128 = 0.765625
- parse modes:
  - delimiter line: 72
  - labeled lines: 26
  - incomplete delimiter pair: 19
  - ambiguous multiple delimiters: 7
  - unrecognized format: 4
- intended-correct exact match: 0.397959
- intended-misleading exact match: 0.051020
- grounded pair fraction: 0.387755
- hard GV consistency: 0.826531
- mean soft GV consistency: 0.816661
- sampled-hard Monte Carlo mean: 0.816586
- sampled-hard MC minus soft mean: -0.0000756
- soft ESS: 81.20 / 98 = 0.828595
- target A fraction, all questions: 0.500000
- target A fraction, parsed questions: 0.489796
- prediction A fraction: 0.581633
- hard consistency, target A: 0.916667
- hard consistency, target B: 0.740000
- mean soft consistency, target A: 0.896325
- mean soft consistency, target B: 0.740183
- grounded pair count: 38
- validator hard accuracy on grounded pairs: 0.921053
- mean soft score on grounded pairs: 0.907775
- paired-order mean correct probability: 0.797195
- paired-order mean position gap A-B: 0.219164
- mean absolute position gap: 0.219767

## Derived comparison

There are 81 hard-consistent examples among the 98 parsed pairs. Of the 38 grounded pairs, 35 are hard-consistent. Therefore among the 60 non-grounded pairs, 46 are still hard-consistent:

[
46/60 \approx 0.7667.
]

The corresponding mean soft score on the non-grounded subset is approximately 0.7590.

This is the central diagnostic separation for the project: high GV agreement is present even when the generator's intended-correct candidate does not pass the conservative TriviaQA exact-match diagnostic.

## Decision

The measurement pipeline is frozen after this run. Parse failures remain observable generator/operator failures rather than being repaired beyond the documented conservative parser. Single randomized A/B order remains the primary GV measurement; paired-order evaluation remains a position-bias diagnostic.

The next phase generates a separate training split and compares hard consistency fine-tuning with soft weighted consistency fine-tuning on exactly the same generated data.
