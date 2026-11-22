# Sparse candidate-local GINE

Hard edge pruning hurt this baseline. Keep complete connectivity: 97/363 solves versus 59/363 with moderate sparsity and 9/363 with strict sparsity. Both sparse variants lose harness solves in every seed.

Validation only: 121 puzzles, three matched initialization seeds (10/11/12). Six new runs, each with 15,000 updates; completed complete-graph controls reused. The same validation puzzles recur across seeds and prior experiments. 363 appearances are not 363 independent puzzles. No notebooks or test-set evaluation.

## Controlled change

At every GINE layer, remove the whole incoming message for excluded pairs, including its node contribution. Zeroing edge features alone would not do this. Preserve the usual self term, learned epsilon, sum aggregation, active projections and readout. Do not renormalize degrees or add a nearest-neighbor fallback.

Fit separate channel thresholds at the 90th and 99th percentiles of all off-diagonal unordered training-board pairs (one copy per full board). Keep an edge when at least one channel is positive and at least its threshold. Retain discrete ties, including binary evidence. Therefore these are channel quantiles, not target graph densities. No labels or validation features enter threshold fitting. Thresholds are fixed across candidates/states, saved in model config, and loaded by the normal harness/Streamlit checkpoint abstraction.

Architecture: candidate-local standalone GINE, hidden 128, two layers, readout 128, 15 original edge channels and frozen Qwen vectors; active LayerNorm, text projection, GINE and readout trainable. There is no global message-passing encoder in this baseline. The mask is deterministic but aggregation remains learned. Same initialization, batch permutations and balanced schedules (5,000 updates each on 8/12/16-word states). Batch 16, LR 0.0003016685, weight decay 0.0000950089, dropout zero; current probability-mass plus floor loss (weight 0.1, beta 0.1) unchanged. Evaluate every 250 updates; restore best s4 Hit@1 checkpoint with MRR tie-break. No early stopping, pruning, scheduling or extra tuning.

Harness: beam 20, full-board score sums, four mistakes, shuffle seed 0 and normal public-feedback pruning. Same selected checkpoint for ranking and harness.

## Connectivity diagnostics

| Split | Variant | All pairs retained | Within true groups | Across true groups | Isolated board nodes |
|---|---|---:|---:|---:|---:|
| train | sparse-q90 | 46.99% | 64.38% | 42.65% | 0.55% |
| train | sparse-q99 | 14.23% | 26.81% | 11.08% | 23.37% |
| val | sparse-q90 | 47.11% | 64.29% | 42.82% | 0.83% |
| val | sparse-q99 | 13.51% | 25.72% | 10.46% | 22.57% |

True-group labels above are used only to diagnose fitted masks; they do not determine connectivity. Isolated-node rates above use the full board; a node can still have no neighbors within a particular candidate quartet.

| Validation group color | q90 pair retention | q99 pair retention |
|---|---:|---:|
| purple | 56.47% | 22.73% |
| blue | 62.53% | 24.10% |
| green | 67.77% | 29.48% |
| yellow | 70.39% | 26.58% |

## Results

| Metric | Complete | Sparse q90 | Sparse q99 |
|---|---:|---:|---:|
| s4 hit1 | 36.36% | 33.33% | 22.31% |
| s4 mrr | 0.4601 | 0.4131 | 0.2951 |
| s4 hit4 | 53.17% | 45.45% | 33.61% |
| s3 hit1 | 38.36% | 32.85% | 21.56% |
| s3 mrr | 0.4837 | 0.4268 | 0.3031 |
| s3 hit4 | 55.10% | 49.31% | 35.67% |
| s2 hit1 | 42.47% | 37.37% | 25.34% |
| s2 mrr | 0.5526 | 0.5019 | 0.3740 |
| s2 hit4 | 65.70% | 60.15% | 44.67% |
| Harness solves | 97/363 (26.72%) | 59/363 (16.25%) | 9/363 (2.48%) |
| Harness first_correct | 144/363 (39.67%) | 127/363 (34.99%) | 68/363 (18.73%) |
| yellow recovered | 185/363 | 140/363 | 74/363 |
| green recovered | 162/363 | 123/363 | 65/363 |
| blue recovered | 136/363 | 97/363 | 35/363 |
| purple recovered | 106/363 | 73/363 | 29/363 |
| Purple selected | 61/363 | 47/363 | 27/363 |
| Purple forced | 45/363 | 26/363 | 2/363 |
| yellow raw Top4 recall | 28.37% | 20.94% | 12.95% |
| green raw Top4 recall | 17.91% | 16.53% | 12.67% |
| blue raw Top4 recall | 7.16% | 6.34% | 4.13% |
| purple raw Top4 recall | 3.86% | 4.96% | 4.41% |

## Per seed

| Seed | Variant | s4 Hit@1 | Solves | Purple selected / forced | Best update |
|---|---|---:|---:|---:|---:|
| 10 | complete | 38.84% | 33/121 | 21 / 15 | 4000 |
| 10 | sparse-q90 | 33.06% | 18/121 | 16 / 8 | 750 |
| 10 | sparse-q99 | 22.31% | 3/121 | 8 / 0 | 1500 |
| 11 | complete | 37.19% | 33/121 | 23 / 14 | 2500 |
| 11 | sparse-q90 | 33.06% | 19/121 | 16 / 8 | 1500 |
| 11 | sparse-q99 | 23.14% | 4/121 | 9 / 2 | 1250 |
| 12 | complete | 33.06% | 31/121 | 17 / 16 | 1750 |
| 12 | sparse-q90 | 33.88% | 22/121 | 15 / 10 | 1500 |
| 12 | sparse-q99 | 21.49% | 2/121 | 10 / 0 | 10750 |

## Paired changes

- sparse-q90: 12 new solves, 50 lost solves; 19 new Purple recoveries, 52 lost Purple recoveries.
- sparse-q99: 5 new solves, 93 lost solves; 11 new Purple recoveries, 88 lost Purple recoveries.

## Interpretation

The moderate mask retains 64.3% of validation true-group pairs but still retains 42.8% of cross-group pairs. The strict mask is more selective, but retains only 25.7% of true-group pairs. Purple is especially exposed: only 56.5% / 22.7% of its within-group edges survive. Existing hand-built features do not provide a reliable hard boundary for message passing.

Purple raw Top4 recall rises slightly (3.86% to 4.96% / 4.41%), but direct Purple selections fall from 61 to 47 / 27; total Purple recovery falls from 106 to 73 / 29. This is not a useful Purple tradeoff for the current harness. The loss of easy-group ranking and recovery dominates.

Keep sparse connectivity opt-in and retain the complete-graph baseline. This result concerns these two fixed hard masks under matched settings; it does not rule out learned or relation-specific graph structure. Removing neighbors also changes GINE message-sum magnitude; degree normalization, separate relation aggregation and soft message gates were not tested here. Missing edge evidence should not automatically be interpreted as an irrelevant connection.

Twelve targeted regression tests passed, covering exact complete-mask equivalence, whole-message removal, surviving self-term gradients, permutation invariance, threshold union/validation, checkpoint reload, and existing GINE/training/loss behavior. All six new checkpoints ran through the normal harness. The independent nine-checkpoint audit reproduced full-board Hit@1 on all 1,089 board/model evaluations.

## Practical limits

This implementation masks dense tensors, rather than using a sparse scatter kernel. It tests the inductive bias and does not promise reduced arithmetic or inference latency. Feature preparation cost is unchanged. Harness game timings include variable search work and are not a controlled model microbenchmark. Defaults remain complete connectivity.

Reproduce with `nix develop -c uv run python scripts/run_gine_sparse.py`, then `scripts/audit_gine_probability_floor.py --results experiments/gine-sparse` and `scripts/summarize_gine_sparse.py` under the same environment. Full thresholds, mask diagnostics, checkpoints, curves, and per-game histories are in `experiments/gine-sparse/`.
