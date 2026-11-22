# Basic GINE hyperparameter tuning and fresh-seed reruns

The tuned model improves validation ranking and feedback solve rate across all four fresh seed pairs. Mean s4 Hit@1 rises from 22.31% to 34.09%, and mean validation solve rate rises from 14.67% to 25.83%. This is validation selection/analysis throughout, not an unbiased held-out test estimate.

## Training setup

Frozen Qwen embeddings; trainable input LayerNorm and 1024→256 Linear/ReLU projection initialized from the shared text-only checkpoint; trainable candidate-local GINE and pooled readout. All active non-Qwen layers train. Original 15 edge channels, same 968 training/121 validation puzzles, batch size 16, mixed s2/s3/s4 sampling, existing probability-mass objective, AdamW. No auxiliary loss, encoder freeze, scheduler, feature changes, or harness changes. The experimental checkpoint class retains unused inherited frozen attention modules; they do not affect scoring.

16 sequential Optuna trials on training seed 0, maximizing validation s4 Hit@1 only. Checkpoint ties within a trial use s4 MRR as before; MRR does not select hyperparameters or prune trials. Features/datasets are prepared once and reused. No harness evaluation inside trial training.

| Parameter | Search space | Selected |
|---|---|---:|
| GINE hidden width | 32, 64, 128 | 128 |
| Message-passing layers | 1–3 | 2 |
| Readout hidden width | 64, 128, 256 | 128 |
| Dropout | 0, 0.1, 0.2 | 0 |
| Learning rate | 3e-5–5e-4, log scale | 0.0003016684667937828 |
| AdamW weight decay | 1e-6–1e-2, log scale | 0.00009500890359387789 |

Dropout, when nonzero, is applied after ReLUs inside the active GINE message MLPs and readout. It is not an ineffective adjustment to unused attention layers. Input width remains 256 so every trial can start from the same learned projection. Selected trainable parameter count is 400,259 versus 286,307 for the baseline.

## Budget and pruning

Maximum 15,000 updates per trial, evaluation every 250. No ordinary early stopping before the full budget. Optuna MedianPruner uses six full startup trials, minimum four comparable values, no pruning before update 8,000, and 1,000-update pruning intervals. It receives best-so-far s4 Hit@1. Completed trials therefore have time to learn even when their best checkpoint appears early; poor later trials still receive at least 8,000 updates.

13 trials completed all 15,000 updates; three were pruned at exactly 8,000; none failed. Trial 11 selected the parameters above with 36.36% validation s4 Hit@1 on search seed 0. The baseline trial scored 25.62%. Sixteen trials are a modest search over this space, not evidence of a global optimum.

At 15,000 updates, each state-size loader gets approximately 5,000 updates: about 82 passes over s4 states, 20.7 over s3, and 13.8 over s2. This avoids the earlier short training budgets. Every final rerun also gets the full 15,000 updates; there is no pruning during reruns.

## Fresh-seed matched reruns

After selecting parameters once, reran the winner and original baseline on seeds 10–13, none used to optimize hyperparameters. Both conditions use the same initialization source, update budget, batch size, evaluation cadence and checkpoint selection. Baseline: hidden width 32, two layers, readout width 128, dropout 0, LR 1e-4, weight decay 1e-6. The same validation puzzles are reused, so fresh seeds test training robustness rather than create a new held-out evaluation.

| Metric | Matched baseline mean | Tuned mean |
|---|---:|---:|
| s4 Hit@1 | 22.31% | 34.09% |
| s4 MRR | 0.3082 | 0.4340 |
| s4 Hit@4 | 35.33% | 49.59% |
| s3 Hit@1 | 23.92% | 35.38% |
| s2 Hit@1 | 31.82% | 40.94% |

s4 Hit@1 standard deviation across four seeds: 2.78 percentage points baseline, 1.41 points tuned. s3/s2 are oracle reduced states, not states reached through feedback.

| Seed | Baseline s4 Hit@1 | Tuned s4 Hit@1 | Baseline solves | Tuned solves | Tuned selected update |
|---:|---:|---:|---:|---:|---:|
| 10 | 21.49% | 33.88% | 8 | 29 | 8250 |
| 11 | 20.66% | 35.54% | 18 | 32 | 1500 |
| 12 | 20.66% | 34.71% | 25 | 29 | 1750 |
| 13 | 26.45% | 32.23% | 20 | 35 | 8000 |
| Mean | 22.31% | 34.09% | 17.75 | 31.25 | — |

All tuned reruns actually trained through update 15,000; early selected checkpoints are restored only afterwards. Results use the entire rerun distribution rather than selecting the most favorable seed.

## Validation feedback games

Each checkpoint plays the same 121 validation puzzles with four allowed mistakes, beam width 20, full-board scoring, shuffle seed 0, and existing public-feedback pruning. Checkpoints were reconstructed by the normal loader from their saved configuration/feature metadata.

| Harness metric | Baseline | Tuned |
|---|---:|---:|
| Mean solves / 121 | 17.75 | 31.25 |
| Solve rate | 14.67% | 25.83% |
| First-guess accuracy | 26.65% | 36.16% |
| Mean recovered groups | 1.118 | 1.630 |

The solve-rate gain is 11.16 percentage points, and every seed pair improves. Across four repeated runs there are 484 opportunities per color:

| Color | Baseline recovered | Tuned recovered |
|---|---:|---:|
| Yellow | 192 (39.67%) | 251 (51.86%) |
| Green | 160 (33.06%) | 213 (44.01%) |
| Blue | 106 (21.90%) | 182 (37.60%) |
| Purple | 83 (17.15%) | 143 (29.55%) |

These are repeated measurements on the same puzzles, not 484 independent puzzles. Better Purple recovery does not establish that direct Purple scoring was solved: s4 individual-Purple-group Top4 remains low (2.27% baseline, 2.69% tuned); recovering other groups and receiving feedback can simplify the remaining problem.

Warm-cache mean game timings were about 0.34 s baseline and 0.30 s tuned. Cache warmth, number of guesses/states, and system variation make these unsuitable for attributing speed differences to architecture alone.

## Interpretation

Straightforward tuning produced a substantial validation gain without auxiliary objectives or complex schedules. The selected changes relative to baseline are wider GINE message states, a higher LR, and stronger weight decay; depth/readout width/dropout stayed unchanged. This study does not isolate which selected change caused the improvement. The earlier gap should not be attributed solely to frozen-representation benefits, vanishing gradients, or input projection size while leaving GINE width and optimization untuned.

The best single search score is more optimistic than the fresh-seed mean and should not be reported as the aggregate result. All analysis remains validation-only; existing test puzzles were untouched. New puzzles should supply a future final evaluation.

## Checkpoints and reproduction

All four tuned reruns are retained:

- [Seed 10](../weights/experiments/20261026T001045Z-tuning-final-tuned-seed-10.pt)
- [Seed 11](../weights/experiments/20261026T001500Z-tuning-final-tuned-seed-11.pt)
- [Seed 12](../weights/experiments/20261026T001915Z-tuning-final-tuned-seed-12.pt)
- [Seed 13](../weights/experiments/20261026T002325Z-tuning-final-tuned-seed-13.pt)

Runner: `scripts/tune_gine.py`. Reproduce with `nix develop -c uv run python scripts/tune_gine.py --trials 16 --updates 15000 --prune-after 8000 --seeds 10 11 12 13`. Study SQLite, trial CSV, exact best parameters, per-trial/per-rerun JSON and all eight harness histories are in `experiments/gine-tuning/`. The script reuses the completed study/reruns when invoked again with the same output directory. Detailed aggregate: [summary.json](gine-tuning/summary.json); learning curves: [rerun-curves.html](gine-tuning/rerun-curves.html). Regenerate aggregates with `nix develop -c uv run python scripts/summarize_gine_tuning.py`.

Checks cover active dropout, configurable width/depth/readout, input gradient flow, checkpoint round-trip, and no pruning before 8000 updates even for an obviously weak synthetic trial. Actual saved checkpoints also loaded successfully for all eight full validation harness replays.
