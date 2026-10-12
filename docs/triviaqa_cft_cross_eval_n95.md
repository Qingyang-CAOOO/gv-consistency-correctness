# TriviaQA N=95 generator-validator cross evaluation

The analysis uses the 95 test questions for which Base, Hard CFT, and Soft CFT
all produced parseable generator pairs. Rows vary the generator trajectory and
columns vary the validator checkpoint.

## Hard GV consistency

| Generator \ Validator | Base V | Hard V | Soft V |
|---|---:|---:|---:|
| Base G | 0.8000 | 0.8105 | 0.8211 |
| Hard G | 0.8421 | 0.8526 | 0.8632 |
| Soft G | 0.8000 | 0.8105 | 0.8211 |

For the hard metric the aggregate matrix is almost exactly additive. Relative
to Base-G/Base-V:

- Hard generator effect: +0.0421 under every validator.
- Soft generator effect: approximately 0 under every validator.
- Hard validator effect: +0.0105 under every generator.
- Soft validator effect: +0.0211 under every generator.

Thus the matched Hard-Hard gain is

[
0.8526-0.8000 = 0.0526 \approx 0.0421+0.0105,
]

and the matched Soft-Soft gain is

[
0.8211-0.8000 = 0.0211 \approx 0+0.0211.
]

The best diagnostic cross-pair is Hard-G/Soft-V at 0.8632.

## Mean soft GV consistency

| Generator \ Validator | Base V | Hard V | Soft V |
|---|---:|---:|---:|
| Base G | 0.7962 | 0.8064 | 0.8156 |
| Hard G | 0.8375 | 0.8514 | 0.8606 |
| Soft G | 0.7985 | 0.8097 | 0.8154 |

The same pattern holds continuously: Hard CFT produces the strongest generator
row, while Soft CFT produces the strongest validator column. Pair-specific
interaction is small at the aggregate level. Using Base as the reference, the
2x2 interaction contrast is approximately +0.0037 for the Hard-Hard pair and
-0.0025 for the Soft-Soft pair.

## Interpretation

This does **not** mean the GV sequence is uncoupled: every validator query still
conditions structurally on the generator output. Rather, this experiment finds
little evidence that the *matched adapted checkpoints* have learned an extra
pair-specific compatibility beyond separable generator and validator effects.

The result suggests a functional specialization:

- Hard CFT appears to improve the generator side substantially.
- Soft CFT appears to improve the validator side more consistently.
- The Hard-generator / Soft-validator diagnostic combination exceeds either
  matched pair on this common intersection.

This motivates testing an asymmetric same-model objective in which generator
and validator terms use different consistency weights, while retaining the same
underlying LM parameters. Such an objective is a new method and should be
evaluated separately rather than conflated with the current Soft-GV baseline.

The aggregate interaction contrasts above are descriptive; per-question paired
uncertainty for the interaction itself has not yet been estimated.
