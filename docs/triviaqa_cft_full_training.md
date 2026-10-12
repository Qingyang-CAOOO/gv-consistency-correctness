# Hard vs soft CFT: full LoRA training

Both runs use the same frozen generator-coupled TriviaQA training JSONL, the
same base model `google/gemma-2-2b-it`, the same LoRA configuration, two
epochs, learning rate (2\times10^{-5}), seed 2026, and exactly 41 optimizer
updates per epoch (82 total).

## Hard CFT

- selected trajectories: 325
- training sequences: 650
- mean sequence weight: 1.0
- trainable parameters: 10,383,360 / 2,624,725,248 = 0.396%
- epoch mean weighted sequence NLL:
  - epoch 1: 9.06430
  - epoch 2: 2.53503

Adapter:
`/content/drive/MyDrive/gv-smc-artifacts/checkpoints/cft_hard_full`

## Soft CFT

- selected trajectories: 401
- training sequences: 802
- mean sequence weight: 0.8081529
- trainable parameters: 10,383,360 / 2,624,725,248 = 0.396%
- epoch mean weighted sequence NLL:
  - epoch 1: 9.16146
  - epoch 2: 2.52508

Adapter:
`/content/drive/MyDrive/gv-smc-artifacts/checkpoints/cft_soft_full`

## Interpretation

The training curves are very similar after matching optimizer-update budgets.
This is expected and is not itself evidence for either method: the objectives
are different weighted empirical measures, so their raw train NLLs are not a
direct performance comparison. The substantive comparison is end-to-end
evaluation on the held-out TriviaQA test protocol, where each adapted model
regenerates its own generator response and then validates that response.
