# Upstream baseline

The companion baseline is RankAlign:

- Repository: https://github.com/juand-r/rankalign
- Paper: RankAlign: A Ranking View of the Generator-Validator Gap in Large Language Models

The public repository documents:

- data for Hypernymy, SWORDS, LAMBADA, and TriviaQA;
- `scripts/ranking_loss_ref.py` for RankAlign;
- `scripts/fine_tune_lora.py` for LoRA SFT;
- `scripts/dpo.py` for DPO;
- `scripts/consistency_ft.py` for the consistency-fine-tuning baseline.

Our repository should remain a companion layer rather than a copied fork. When using the bootstrap, record the exact upstream commit before experiments.
For deterministic reproduction, provide a verified historical SHA:

```bash
RANKALIGN_REF=<upstream-commit-sha> bash scripts/bootstrap_rankalign.sh
```

Without `RANKALIGN_REF`, the script clones upstream's current default head for a fresh exploratory run; this may not match the data revision used in the completed 2026 experiment. No authoritative historical upstream SHA is stored in this code-only snapshot; do not infer it from the current RankAlign head. The `external/rankalign/` directory is intentionally excluded from Git.
