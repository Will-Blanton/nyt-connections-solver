# Feedback search beam-width comparison

Checkpoint selected from validation before test evaluation: `weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt`. Same 121 held-out test puzzles, deterministic board shuffle seed 0, four allowed mistakes, and checkpoint/features for every width. Each run used the same feedback-aware search policy; only beam width changed. The policy rescored after public feedback and pruned infeasible groups. Full per-guess traces with raw and post-pruning rank are in `harness-beam-{width}-test.json`.

| Beam width | Solved | First guess correct | Mean correct groups | First choice differs from raw top | Mean game time |
|---:|---:|---:|---:|---:|---:|
| 1 | 17/121 (14.05%) | 38/121 (31.40%) | 1.231 | 0/121 | 0.100 s |
| 5 | 17/121 (14.05%) | 40/121 (33.06%) | 1.149 | 28/121 | 0.132 s |
| 20 | 17/121 (14.05%) | 40/121 (33.06%) | 1.157 | 30/121 | 0.248 s |
| 100 | 17/121 (14.05%) | 40/121 (33.06%) | 1.157 | 30/121 | 0.782 s |

At width 20, the 30 non-top first choices had raw ranks: 21 at rank 2, two at rank 3, four at rank 4, one at rank 6, and two at rank 8. Five of those 30 were correct. Compared with width 1, changing the first group improved first-guess correctness on five boards and worsened it on three. The same number of games were solved at both widths, but they were not identical games: one win was gained and one lost. Width 5 vs 20 changed four first choices and no solved outcomes. Width 20 vs 100 changed no first choices or solved outcomes.

Across all guesses, width 20 chose the raw rank-1 candidate 198/599 times (33.1%), and the highest-scoring feedback-feasible candidate 417/599 times (69.6%). Raw rank-1 counts include later turns where feedback can prune the original leader. Width 1 always chooses the highest-scoring feasible candidate at each turn, yet only 243/603 guesses were raw rank 1 (40.3%). Thus feedback pruning matters, while deeper beam optimization rarely changes this checkpoint's decisions enough to improve full solves.

The bottleneck in this experiment is the ranking and/or the objective for choosing a guess, rather than beam truncation beyond a small width. A useful next controlled policy comparison would score guesses for the chance of an immediate correct group versus partition-level log probability, with the same model, puzzle order, mistake budget, and width. This is a hypothesis from these outcomes, not a measured improvement.

Reproduce with `nix develop -c uv run python scripts/evaluate_graph_feedback.py weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt --split test --beam-width WIDTH --output experiments/harness-beam-WIDTH-test.json` for widths 1, 5, 20, and 100. The first width-20 harness report was run before per-guess rank capture; this rerun has the same 17/121 solves. Game-time figures are wall-clock and can vary with cache warmth, so treat them as approximate.
