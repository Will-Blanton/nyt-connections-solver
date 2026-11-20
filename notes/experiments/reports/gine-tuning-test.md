# Tuned GINE: fixed-checkpoint test evaluation

The tuning gain carries over to test: mean s4 Hit@1 improves by 7.02 percentage points and solve rate by 9.09 points versus the matched baseline. Every seed pair improves solve rate; ranking improves in three pairs and ties in one. Tuned test results are below tuned validation by 3.10 points Hit@1 and 3.51 points solve rate. Purple remains the least recovered color.

Evaluated the already-selected parameters and saved validation-selected weights without retraining, changing the harness, or selecting on test outcomes. Primary checkpoint: fresh tuned seed 11, chosen by validation s4 Hit@1 before test evaluation. All four previously fixed seeds are also reported for both baseline and tuned configurations.

Existing test split: 121 puzzles, disjoint from training and validation. This test split was inspected in earlier project experiments, so it is not a pristine final holdout; newly ingested puzzles remain appropriate for a future final evaluation.

Harness: beam 20, full-board scoring, four allowed mistakes, shuffle seed 0, existing feedback pruning. s3/s2 ranking metrics use oracle reduced states, not feedback trajectories. Seed averages repeat the same puzzles and are not independent test datasets.

| Metric | Baseline test mean | Tuned test mean | Tuned validation mean |
|---|---:|---:|---:|
| s4 hit1 | 23.97% | 30.99% | 34.09% |
| s4 mrr | 0.3479 | 0.4106 | 0.4340 |
| s4 hit4 | 42.77% | 50.21% | 49.59% |
| s3 hit1 | 28.00% | 34.66% | 35.38% |
| s3 mrr | 0.3883 | 0.4512 | 0.4613 |
| s3 hit4 | 47.16% | 53.41% | 54.24% |
| s2 hit1 | 34.57% | 40.87% | 40.94% |
| s2 mrr | 0.4697 | 0.5270 | 0.5361 |
| s2 hit4 | 55.99% | 61.43% | 64.39% |
| Harness solve_rate | 13.22% | 22.31% | 25.83% |
| Harness first_guess_accuracy | 28.51% | 39.26% | 36.16% |
| Harness mean_correct_groups | 1.157 | 1.481 | 1.630 |

| Seed | Baseline s4 Hit@1 | Tuned s4 Hit@1 | Baseline solves | Tuned solves |
|---:|---:|---:|---:|---:|
| 10 | 23.97% | 23.97% | 11 | 27 |
| 11 | 23.14% | 33.06% | 15 | 28 |
| 12 | 24.79% | 33.06% | 20 | 26 |
| 13 | 23.97% | 33.88% | 18 | 27 |

Primary checkpoint (seed 11): s4 Hit@1 33.06%, MRR 0.4460, 28/121 solves (23.14%).

| Color | Baseline recovered | Tuned recovered |
|---|---:|---:|
| yellow | 209/484 (43.18%) | 261/484 (53.93%) |
| green | 173/484 (35.74%) | 200/484 (41.32%) |
| blue | 100/484 (20.66%) | 139/484 (28.72%) |
| purple | 78/484 (16.12%) | 117/484 (24.17%) |

Detailed metrics and per-guess histories: [test/summary.json](gine-tuning/test/summary.json).
Reproduce: `nix develop -c uv run python scripts/evaluate_gine_tuning_test.py`. Existing completed harness replays are reused; no training runs.
