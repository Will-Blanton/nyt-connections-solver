# Replace path probabilities with original-board partition quality

The existing search adds conditional, per-state normalized group log probabilities as it traverses 16-, 12-, and 8-word states; the forced last group has no model score. This experiment scores every proposed group with the same model on the original 16-word board and sums those four log probabilities for a complete partition. The last group's score is included as soon as a path leaves four words, before beam pruning. Feedback constraints, checkpoint, puzzle order/shuffle seed 0, four-mistake budget, and model features are identical. No training changes. The alternative is `score_mode=full_board`; `path` remains the default.

Checkpoint: `weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt` (previously selected by validation s4 Hit@1/MRR). Validation compared both modes at widths 20 and 100. Width 20 was then evaluated once on test. The test set has been used for earlier harness diagnostics, so this is a useful confirmation on the fixed setting but not a pristine benchmark for continued policy tuning.

| Split | Beam | Score mode | Fully solved | First guess correct | Mean correct groups | Mean game time |
|---|---:|---|---:|---:|---:|---:|
| Validation | 20 | Per-state path | 7/121 (5.79%) | 28/121 (23.14%) | 0.843 | 0.24 s |
| Validation | 20 | Original-board partition | 14/121 (11.57%) | 30/121 (24.79%) | 1.074 | 0.26 s |
| Validation | 100 | Per-state path | 7/121 (5.79%) | 28/121 (23.14%) | 0.843 | 0.82 s |
| Validation | 100 | Original-board partition | 14/121 (11.57%) | 30/121 (24.79%) | 1.058 | 0.86 s |
| Test | 20 | Per-state path | 17/121 (14.05%) | 40/121 (33.06%) | 1.157 | 0.25 s |
| Test | 20 | Original-board partition | 19/121 (15.70%) | 39/121 (32.23%) | 1.256 | 0.25 s |

At width 20, the replacement changed the first choice on 56 validation boards, gained nine solved games and lost two. On test it changed 56 first choices, gained five solves and lost three. It raised mean correct groups on both splits. One-away guesses rose on test (208 to 229), so its benefit is not simply fewer near misses.

Diagnostic: the true complete partition was present in the penultimate beam on 6/121 validation boards for the path score and 4/121 for original-board score at width 20; at width 100 the counts were 11 and 15. Those counts do not bound eventual solves, because the policy replans after each piece of feedback and can recover a partition that was absent initially. The score replacement improved decisions despite the very limited initial beam coverage. Width 100 improved beam coverage but did not improve solves over width 20 in this comparison.

This supports the concern that independently normalized residual-state probabilities make poor partition scores. It also suggests a cheap harness change can use the existing model more effectively, though the test solve gain is small and repeated test inspection limits confidence. Retain the path mode as a control until a fresh split or new puzzles confirm the effect.

Raw per-game traces with raw ranks and gold-partition beam survival: `experiments/harness-score-{path,full_board}-beam-{20,100}-val.json`, `experiments/harness-score-full_board-beam-20-test.json`; the matching earlier test control is `experiments/harness-beam-20-test.json`. Reproduce with `nix develop -c uv run python scripts/evaluate_graph_feedback.py weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt --split SPLIT --beam-width WIDTH --score-mode MODE --output OUTPUT.json`.
