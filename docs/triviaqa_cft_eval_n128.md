# TriviaQA held-out N=128: Base vs Hard CFT vs Soft CFT

All three runs use the same 128 test questions, the same deterministic data
order, greedy generator protocol, randomized A/B target sequence, and validator
settings. Hard and soft adapters were trained from the same Gemma-2-2B-IT base
checkpoint and the same frozen N=512 training trajectories.

## Reported parsed-trajectory metrics

| Metric | Base | Hard CFT | Soft CFT |
|---|---:|---:|---:|
| Parse success | 0.7734 | 0.9688 | 0.9375 |
| Intended-correct EM | 0.4141 | 0.4274 | 0.4333 |
| Grounded-pair fraction | 0.4141 | 0.3952 | 0.4083 |
| Hard GV consistency | 0.7980 | 0.8629 | 0.8250 |
| Mean soft GV consistency | 0.7939 | 0.8623 | 0.8204 |
| Soft ESS fraction | 0.8084 | 0.8699 | 0.8260 |
| Validator accuracy on grounded pairs | 0.9024 | 0.9184 | 0.8776 |
| Paired-order mean correct probability | 0.7805 | 0.8454 | 0.8121 |
| Mean absolute A/B position gap | 0.2168 | 0.1814 | 0.1685 |

## Preliminary interpretation

Hard CFT gives the largest increase in GV consistency: +6.49 percentage points
in hard consistency and +6.84 points in mean soft consistency relative to Base.
Soft CFT also improves both, but more modestly (+2.70 and +2.65 points).

Soft CFT gives the strongest reduction in order/position asymmetry. Its mean
absolute paired-order gap falls from 0.2168 to 0.1685, compared with 0.1814 for
Hard CFT. Its predicted-A fraction (0.55) is also closer to the balanced target
than Base (0.596) or Hard CFT (0.565).

Generator intended-correct EM on parsed examples changes only slightly:
0.4141 -> 0.4274 (Hard) and 0.4333 (Soft). Grounded-pair fraction conditional
on parsing does not improve: 0.4141 -> 0.3952 (Hard) and 0.4083 (Soft).

However, parse success changes substantially across models. Therefore the
parsed-only metrics are not directly comparable as if they were evaluated on
the same subset. Counting successful outputs over all 128 questions gives:

| End-to-end count/fraction | Base | Hard CFT | Soft CFT |
|---|---:|---:|---:|
| Parsed + intended-correct EM | 41/128 = 0.3203 | 53/128 = 0.4141 | 52/128 = 0.4062 |
| Parsed + grounded pair | 41/128 = 0.3203 | 49/128 = 0.3828 | 49/128 = 0.3828 |
| Parsed + hard-consistent | 79/128 = 0.6172 | 107/128 = 0.8359 | 99/128 = 0.7734 |

These end-to-end quantities include formatting/operator compliance, so they
should be reported separately from semantic metrics conditional on successful
parsing.

## Next analysis

Use `scripts/compare_gv_runs.py` on the three saved JSONL files. It reports
(1) end-to-end metrics over the same 128 questions, (2) semantic/GV metrics on
the common parsed intersection, and (3) paired bootstrap confidence intervals
for Base -> Hard, Base -> Soft, and Hard -> Soft differences.
