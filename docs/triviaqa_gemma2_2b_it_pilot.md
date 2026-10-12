# TriviaQA generator-coupled GV pilot: Gemma-2-2B-IT

Pilot configuration:
- model: `google/gemma-2-2b-it`
- split: RankAlign TriviaQA test
- questions: 16
- generator: greedy, max 32 new tokens
- data/generation/order/validator-sample seeds: 42/43/44/45
- validator-order randomization: balanced
- paired-order diagnostic: enabled

## Pilot result

- parsed generator pairs: 14/16 = 0.875
- parse modes: 10 delimiter-line, 4 labeled-lines, 1 incomplete pair, 1 ambiguous multiple-delimiter output
- intended-correct exact match: 0.500
- intended-misleading exact match: 0.000
- grounded pair fraction: 0.500
- hard GV consistency: 0.7857
- mean soft GV consistency: 0.7935
- sampled-hard Monte Carlo mean: 0.7937
- sampled-hard MC minus soft mean: 0.000239
- soft ESS: 11.33 / 14 = 0.8093
- hard consistency when target is A: 1.000
- hard consistency when target is B: 0.5714
- paired-order mean correct probability: 0.7708
- paired-order mean position gap (A minus B): 0.2475
- mean absolute position gap: 0.2475
- grounded-pair count: 7
- validator hard accuracy on grounded pairs: 1.000
- mean soft score on grounded pairs: 0.8586

## Interpretation

This pilot supports keeping three measurements separate:
1. generator format compliance,
2. external factual correctness against TriviaQA,
3. generator-validator consistency.

The sampled-hard Monte Carlo average closely matches the soft potential, providing an empirical check of the conditional-expectation / Rao--Blackwell interpretation. The pilot also shows a substantial A/B position effect, so paired-order scoring remains a diagnostic while the primary GV score keeps the original single randomized order.
