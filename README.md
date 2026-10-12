# Consistency versus Correctness: Learning Dynamics under Generator–Validator Conditioning
This project extends generator–validator (GV) consistency (introduced by [Li et al. (2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/bfcb583d225b1db8d3ca2331f6785774-Paper-Conference.pdf)) stochastically from hard binary filtering to probabilistic weighting, interprets the latent space via exploring the gap between GV-consistency conditioning and latent correctness posterior. 

## Research Questions

**Does conditioning on GV consistency produce different selection and learning dynamics from conditioning on inferred correctness?** Specifically:

1. Can correctness-posterior weights distinguish ungrounded examples that hard GV filtering accepts or soft GV weighting favors?
2. Does correctness-weighted fine-tuning improve factual grounding rather than merely generator–validator consistency?

## Methods
With
- dataset  ```TriviaQA``` and
- language model ```google/gemma-2-2b-it```, 

frozen correctness estimators, five matched CFT weighting rules, three single-round training seeds, and three iterative feedback rounds are evaluated.

1. The model generates an intended-correct and an intended-misleading answer; the same model validates the randomly ordered pair.
2. Hard GV uses the binary consistency indicator $C$; soft GV uses the normalized probability of choosing the intended-correct answer.
3. Compare these consistency weights with supervised-calibrated factual-validity estimates: $q_G$ (generator evidence only) and $q_{GV}$ (generator plus both-order validator evidence). 
4. Grounding is measured conservatively by exact match: the intended-correct answer must match a gold alias, and the misleading answer must not.
5. The controlled fine-tuning (CFT) comparison has **five arms**: `hard`, `soft`, `posterior-g`, `posterior-g-ess` (ESS-matched control), and `posterior-gv`. Each uses the same **320 training questions** and is evaluated on **512 development questions**. The posterior-weighted arms are replicated over three single-round training seeds; all five arms are followed for three feedback rounds in a seed-2026 pilot. Each feedback round generates and scores new pairs with the previous checkpoint, applies the *same frozen* posterior estimators, then trains a **fresh base-model LoRA adapter** on that round's weighted pairs (two epochs, 82 optimizer updates).

## Takeaways
Above experiments study whether GV-consistency and inferred factual correctness select different training examples, and whether correctness-informed selection leads to better learning. And the answer is asymmetric: 
- yes for selection, 
- not demonstrably for downstream fine-tuning.

Main results include: 
- **consistency does not imply factual grounding.** On 394 parsed confirmation pairs, hard GV retained 326, including **164 EM-ungrounded pairs** (50.31% false retention under this proxy).
- **Correctness inference improves selection.** The fraction of weight assigned to grounded pairs was **49.69%** with hard GV, **49.85%** with sampled-order soft GV, and **60.73%** with $q_{GV}$. Relative to ESS-matched $q_G$, the $q_{GV}$ improvement was **+1.72 percentage points** (retrospective paired 95% CI [0.45, 3.03] pp).
- **Better selection did not reliably improve learning.** At feedback round 3, grounded yields were **217/512** (hard), **221/512** (soft), **214/512** ($q_G$), **215/512** (ESS-matched $q_G$), and **213/512** ($q_{GV}$). Paired comparisons did not establish a robust advantage for posterior-GV training. The confirmation subgroup audit is retrospective; these results concern exact-match grounding, not semantic truth.

## Running Description

Use **Python 3.10+**. The model-scoring and fine-tuning experiments were run on a CUDA-enabled Google Colab environment. Clone the repository (GitHub authentication is required while it is private):

```bash
git clone https://github.com/Qingyang-CAOOO/gv-consistency-correctness-dev.git
cd gv-consistency-correctness-dev
python -m pip install -r requirements/colab.txt
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke_test.py
```

For a **small end-to-end GV measurement** (GPU and access to the Gemma model required), retrieve RankAlign's TriviaQA source files and run:

```bash
bash scripts/bootstrap_rankalign.sh
python scripts/run_triviaqa_gv.py \
  --model google/gemma-2-2b-it --device cuda \
  --split test --limit 16 --generation-mode greedy \
  --paired-order-diagnostic \
  --output outputs/gv_pilot_n16.jsonl
```

To **recompute the reported retrospective audits**, supply the original saved experiment artifacts (not included in Git), including frozen question splits, calibrated models, and per-example trajectories. For example:

```bash
ROOT=/path/to/protocol-v2
python scripts/audit_false_retention.py \
  --root "$ROOT" --bootstrap 3000 \
  --output outputs/false_retention_audit.json
```

The five-arm training runs additionally require the original LoRA checkpoints and artifact layout described in [`docs/protocol_v2_truth_audit.md`](docs/protocol_v2_truth_audit.md) and [`docs/stage5_iterative_cft_v1.md`](docs/stage5_iterative_cft_v1.md). The audit does **not** inspect the reserved final holdout. **This repository currently provides code and protocols, not a self-contained reproduction of all reported results.**
