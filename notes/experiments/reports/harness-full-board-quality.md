# Original-board partition quality term

The feedback harness previously accumulated per-state normalized candidate log probabilities. A bad first group could leave a weak final group that receives no model score when forced. This experiment adds the original 16-word board log probability of each proposed group to the partition search score, including the final forced group. The final group's term is added as soon as a path leaves four words, before that path is truncated by the beam. The base scorer, model checkpoint, feasibility pruning, beam width 20, deterministic puzzle shuffle seed 0, and four-mistake budget are unchanged. Weight 0 reproduces the prior policy; weight 1 applies the new term once per group. No model training or calibration was changed.

Checkpoint: `weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt`, selected previously by validation s4 Hit@1 and MRR. Weight 1 was selected as the one simple candidate for this experiment. Validation was evaluated first; the same fixed setting was then evaluated once on test. The test set had been used for previous harness diagnostics, so it is not a pristine benchmark for iterative harness research.

| Split | Weight | Fully solved | First guess correct | Mean correct groups | One-away guesses | Mean game time |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0 | 7/121 (5.79%) | 28/121 (23.14%) | 0.843 | 214 | 0.238 s |
| Validation | 1 | 10/121 (8.26%) | 29/121 (23.97%) | 0.975 | 221 | 0.258 s |
| Test | 0 | 17/121 (14.05%) | 40/121 (33.06%) | 1.157 | 208 | 0.248 s |
| Test | 1 | 17/121 (14.05%) | 39/121 (32.23%) | 1.198 | 208 | 0.245 s |

On validation, the new score changed the first choice on 40 boards, gained four solved games and lost one. Correct groups recovered by color changed from Purple/Blue/Green/Yellow = 12/15/37/38 to 18/21/37/42. On test, it changed the first choice on 37 boards, gained two solved games and lost two. Correct groups by color changed from 20/22/48/50 to 21/24/46/54. Full solve rate did not improve on test, though mean correct groups rose slightly. Observed runtime differences are small compared with run-to-run cache and timing variation.

This supports the concern that per-state normalized path scores miss some partition quality information, but does not establish a solve-rate improvement. The term is optional (`full_board_weight=0` by default). It is a fixed harness calculation from the same model outputs; it does not use answer labels at inference. Further weight tuning on this test set would undermine the held-out comparison.

Raw per-game results: `experiments/harness-full-board-weight-{0,1}-val.json` and `experiments/harness-full-board-weight-1-test.json`; the matching test baseline is `experiments/harness-beam-20-test.json`. Reproduce with `nix develop -c uv run python scripts/evaluate_graph_feedback.py weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt --split SPLIT --beam-width 20 --full-board-weight WEIGHT --output OUTPUT.json`.
