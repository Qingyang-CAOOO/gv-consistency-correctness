# TriviaQA sampled-hard CFT held-out N=128

Sampled-hard CFT was trained from the same frozen N=512 generator-coupled
training trajectories as the greedy-hard and soft CFT runs. The training JSONL
was transformed so that the hard selector used the stored sampled validator
draw, i.e. (C=1\{Y_V=R\}), rather than the greedy A/B argmax. The saved
training summary still reports mode `hard` because the existing hard trainer
was reused on that transformed JSONL.

## Held-out result

- parse success: 121/128 = 0.9453125
- intended-correct EM among parsed: 0.438017
- intended-misleading EM among parsed: 0.082645
- grounded-pair fraction among parsed: 0.396694
- greedy hard GV consistency: 0.867769
- mean soft GV consistency: 0.858858
- sampled hard consistency (one draw): 0.834711
- sampled-hard Monte Carlo mean: 0.858471
- sampled-hard MC minus soft mean: -0.000386
- soft ESS fraction: 0.870109
- prediction A fraction: 0.537190
- target-A hard consistency: 0.915254
- target-B hard consistency: 0.822581
- grounded-pair count: 48
- validator hard accuracy on grounded pairs: 0.937500
- mean soft score on grounded pairs: 0.927704
- paired-order mean correct probability: 0.844474
- paired-order mean A-B position gap: 0.112474
- paired-order mean absolute position gap: 0.128529

## Immediate comparison

Relative to greedy-hard CFT, sampled-hard CFT has nearly the same held-out
consistency (0.8678 vs 0.8629 hard; 0.8589 vs 0.8623 soft), but markedly lower
order asymmetry (absolute gap 0.1285 vs 0.1814) and a prediction-A fraction
closer to 0.5 (0.5372 vs 0.5645).

Relative to soft CFT, sampled-hard CFT has substantially higher held-out
consistency (0.8678 vs 0.8250 hard; 0.8589 vs 0.8204 soft) and lower absolute
order gap (0.1285 vs 0.1685). These are descriptive N=128 comparisons; paired
bootstrap uncertainty should be computed before treating them as findings.

The sampled-hard Monte Carlo mean again agrees closely with the exact soft
probability, as required by the Rao--Blackwell relationship for the stochastic
hard validator draw.
