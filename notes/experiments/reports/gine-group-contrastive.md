# Per-correct-group contrastive loss on GINE

Validation only: 121 puzzles, three matched seeds (10/11/12), 15,000 updates each. Reuse completed probability-mass controls from `gine-curriculum/mixed`. The same validation puzzles are reused across seeds and previous experiments; 363 puzzle-seed appearances are not 363 independent puzzles. No notebooks or test-set evaluation.

## Controlled change

Replace `-log(sum_correct softmax(score))` with the mean, over each correct quartet g, of `softplus(logsumexp(incorrect scores) - score[g])`. Each correct group competes with all incorrect groups, excluding other correct groups from its denominator. Average groups within a state, then states within a batch. This encourages every correct group to outrank incorrect groups without imposing equal scores on correct groups. It remains candidate scoring, not joint differentiable partitioning.

Everything else is fixed: standalone candidate-local GINE, widths 128, two GINE layers, original 15 edges, no phonetics or letter-edit additions, frozen Qwen vectors, fully trainable active LayerNorm/projection/GINE/readout. Use the same text-only projection initialization, seeds, batch permutations and balanced state-size schedules as controls (5,000 updates each at s2/s3/s4). Batch 16, LR 0.0003016685, weight decay 0.0000950089, dropout 0. Evaluate every 250 updates, select s4 Hit@1 with MRR tie-break. No pruning, early stopping, auxiliary objectives, schedules or tuning. Hyperparameters were previously selected for the probability-mass objective; this is a fixed-setting objective comparison, not a tuned upper bound.

Harness: normal checkpoint loader, beam 20, original full-board group-score sums, four mistakes, public-feedback pruning, shuffle seed 0.

## Aggregate

| Metric | Probability mass | Per-group contrastive |
|---|---:|---:|
| s4 hit1 | 33.61% | 27.27% |
| s4 mrr | 0.4366 | 0.3628 |
| s4 hit4 | 51.24% | 42.42% |
| s3 hit1 | 36.23% | 28.37% |
| s3 mrr | 0.4697 | 0.4000 |
| s3 hit4 | 54.61% | 48.55% |
| s2 hit1 | 42.70% | 38.25% |
| s2 mrr | 0.5477 | 0.5135 |
| s2 hit4 | 64.23% | 62.03% |
| Harness solves | 86/363 (23.69%) | 64/363 (17.63%) |
| Harness first_correct | 126/363 (34.71%) | 105/363 (28.93%) |

## Coverage of all true groups

Full-board quartet ranks use the exact same selected checkpoints as the headline/harness metrics, never a separately selected checkpoint.

| Metric | Probability mass | Per-group contrastive |
|---|---:|---:|
| mean_rank | 577.95 | 402.96 |
| mrr | 0.117 | 0.102 |
| top4_recall | 13.43% | 11.36% |
| median_rank | 328.00 | 184.00 |
| mean_worst_group_rank | 1368.50 | 1000.29 |
| all_four_top4 | 0/363 | 0/363 |
| at_least_three_top4 | 0/363 | 0/363 |

| Color | Baseline Top4 recall | Per-group Top4 recall | Baseline recovered | Per-group recovered |
|---|---:|---:|---:|---:|
| yellow | 27.27% | 18.18% | 179/363 | 136/363 |
| green | 17.63% | 15.43% | 161/363 | 128/363 |
| blue | 5.23% | 6.61% | 122/363 | 93/363 |
| purple | 3.58% | 5.23% | 100/363 | 87/363 |

| Purple detail | Probability mass | Per-group |
|---|---:|---:|
| selected | 62/363 | 69/363 |
| forced | 38/363 | 18/363 |
| Targeted letter-edit Purple recovered | 5/24 | 3/24 |

## Per seed

| Seed | Objective | s4 Hit@1 | s3 Hit@1 | s2 Hit@1 | Solves | Purple | Selected update |
|---|---|---:|---:|---:|---:|---:|---:|
| 10 | probability-mass | 35.54% | 39.26% | 44.08% | 33/121 | 37/121 | 8250 |
| 10 | per-group | 27.27% | 29.34% | 39.39% | 22/121 | 31/121 | 1750 |
| 11 | probability-mass | 33.06% | 35.12% | 41.60% | 28/121 | 35/121 | 7250 |
| 11 | per-group | 28.10% | 29.34% | 40.08% | 27/121 | 35/121 | 1250 |
| 12 | probability-mass | 32.23% | 34.30% | 42.42% | 25/121 | 28/121 | 750 |
| 12 | per-group | 26.45% | 26.45% | 35.26% | 15/121 | 21/121 | 8500 |

## Paired changes

26 new solves and 48 lost solves; 34 new Purple recoveries and 47 lost Purple recoveries.

Full metrics and per-date Purple changes: `gine-group-contrastive/summary.json`. Mechanism labels are analysis-only; see `group-ranks.json` for all 2,904 true-quartet rank records. The existing feature families and default objective remain unchanged outside this opt-in experiment.

## Interpretation

This replacement objective hurts the current GINE system at the fixed settings. Full-board Hit@1 falls 6.34 percentage points and harness solve rate falls 6.06 points (22 fewer solves). Every seed loses both Hit@1 and solves, so the aggregate decline is not driven by just one seed. Smaller-state Hit@1 also declines.

There is a real change in the ranking distribution: mean true-group rank improves from 577.95 to 402.96, median from 328 to 184, and mean worst true-group rank from 1368.50 to 1000.29. However, true-group MRR and Top4 recall decline. Moving poorly ranked positives upward does not compensate for the loss of strong top choices.

Purple has a modest direct-retrieval gain: Top4 recall rises from 3.58% to 5.23%, and Purple selected before the forced last four rises from 62 to 69. Total Purple recovery nevertheless falls from 100 to 87 as forced final-group recoveries fall from 38 to 18. The eight curated letter-edit puzzles fall from 5/24 to 3/24 Purple recoveries. Those small direct gains do not justify the broader regressions.

All runs finish the full 15,000-update budget. Selected updates are 1,750, 1,250, and 8,500; later training continues to reduce training loss without beating those validation checkpoints. This is not evidence that an insufficient update budget caused the observed decline.

Keep the probability-mass baseline. The result applies to this full objective replacement with existing hyperparameters; it does not rule out a separately tuned objective or a weak all-group auxiliary term. Neither was tested here. Twelve targeted loss/training regression tests passed. All three new checkpoints loaded and ran through the normal validation harness, and the group-rank audit independently reproduced all six runs' full-board Hit@1.
