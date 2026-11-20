# Relative near-miss loss experiment

The existing projected-input GINE branch and probability-mass loss were kept fixed. The experimental loss adds `0.25 * softplus(max_score(incorrect 3-word-overlap set) - max_score(correct set) + 0.5)` to each batch's probability-mass loss. This penalizes close or higher-scoring three-word-overlap errors relative to the best correct group, without treating every such error as equally bad. The masks contain 192, 96, and 32 incorrect near-miss candidates for s4, s3, and s2 respectively. All other architecture, features, split, optimizer, update budget, early stopping, and seed settings match the existing projected GINE runs.

Four paired seeds (0–3); each checkpoint selected on validation s4 Hit@1, then s4 MRR. No test evaluation was used.

| Validation metric | Probability mass only | Plus relative margin |
|---|---:|---:|
| s4 Hit@1 | 22.93% | 22.31% |
| s4 MRR | 0.3153 | 0.3176 |
| s4 Hit@4 | 36.98% | 37.60% |
| s3 Hit@1 | 23.45% | 23.81% |
| s2 Hit@1 | 28.65% | 29.27% |
| Purple s4 top1 | 0.83% | 0.83% |
| Purple s4 top4 | 1.86% | 1.45% |

s4 Hit@1 improved in one of four paired seeds; per-seed differences (percentage points) were +2.48, -0.83, -2.48, -1.65. The margin improved some later-state and aggregate ranking metrics but did not consistently fix first-choice ranking or Purple.

The same validation games were also played through the feedback harness (beam width 20, four mistakes, shuffle seed 0). These are repeated evaluations of the same 121 boards, not 484 independent boards.

| Seed | Baseline solves | Margin solves | Baseline first guess | Margin first guess | Baseline mean groups | Margin mean groups |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 5 | 18.18% | 15.70% | 0.603 | 0.752 |
| 1 | 6 | 8 | 19.83% | 21.49% | 0.752 | 0.744 |
| 2 | 2 | 3 | 17.36% | 15.70% | 0.512 | 0.512 |
| 3 | 2 | 2 | 19.83% | 20.66% | 0.620 | 0.669 |
| Mean | 3.75 | 4.50 | 18.80% | 18.39% | 0.622 | 0.669 |

The margin gained three total solves across four paired runs (two in seed 1, one in seed 2), but did not improve first-guess accuracy. It also produced more one-away guesses on average (210.25 vs 201 per run), so the evidence does not support the intended mechanism of suppressing near misses. Given the small and inconsistent gains, this one weight/margin setting should not replace the base objective. A future experiment could tune the weight on validation, but that would need a fresh held-out comparison before claiming a gain.

Raw training JSON: `experiments/graph-features/*gated-gine-projected-margin.json`. Per-game harness outcomes: `experiments/harness-margin-{seed}-{baseline,margin}-val.json` (seed 1 files omit the seed segment). Reproduce a run with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-gine-projected-margin --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --seed SEED --updates 3000 --min-updates-before-stop 1500 --patience 8 --lr 1e-4 --weight-decay 1e-6 --near-miss-weight 0.25 --near-miss-margin 0.5`.
