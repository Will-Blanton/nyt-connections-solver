# Best validation checkpoint: held-out feedback search

Selected before viewing test outcomes: `weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt` (seed 2). Among saved graph experiments, it ties the best validation s4 Hit@1 (26.45%) and has the higher s4 MRR (0.3664). This is the projected-input GINE branch with the original 15 edge channels; its two new compound channels are masked. The frozen base still has global and candidate attention.

The local historical feedback environment shuffled each 16-word board (seed 0), allowed four mistakes, and gave correct/one-away/ordinary-miss feedback. Search beam width was 20. The model received no gold labels; evaluation used the checkpoint's untouched 121-date test split. The same policy rescored after pruning. This measures actual feedback-game trajectories rather than oracle s3/s2 states. Full per-guess histories are in `harness-best-test.json`.

| Measure | Test result |
|---|---:|
| Fully solved | 17/121 (14.05%) |
| First guess correct | 40/121 (33.06%) |
| Mean correct groups | 1.157/4 |
| Mean guesses | 4.95 |
| Mean mistakes | 3.79/4 |
| Mean search/game time | 0.383 s |

Games ended with 0, 1, 2, 3, or 4 correct groups in counts 47, 42, 15, 0, 17. Three is impossible at termination here because the last group is forced once three are correct. Among successful games, 11/17 used all three allowable mistakes, 4 used two, and 2 used one.

## Failure patterns

- Incorrect guesses dominated the budget: 251 ordinary misses plus 208 one-away guesses across 121 games. Of ordinary misses, 232 overlapped a true group by two words and 19 by one word. One-away feedback appeared in 104/121 games; only 15 of those were eventually solved. This is an observed association, not a causal effect of one-away feedback.
- After a one-away guess with another turn available, the next guess was correct 33/164 times (20.1%), one-away again 64 times, and an ordinary miss 67 times. The feedback pruning is being used, but the next choice often remains wrong.
- Starting with a correct group helped but was insufficient: 12/40 such games were solved. Starting with a one-away guess yielded 5/43 solves; starting with an ordinary miss yielded 0/38.
- Correct groups selected across all games: Yellow 50, Green 48, Blue 22, Purple 20. Each color appears once per board, so these are also recoveries out of 121 opportunities per color. Among losses, only 5 Blue and 3 Purple groups were recovered. The model tends to find easier groups first and struggles to finish the difficult ones before exhausting mistakes.

The test solve rate is encouraging relative to pure blind guessing, but there is a substantial gap between first-group accuracy and complete solves. The most direct next diagnostic is whether increasing the allowed mistake budget or beam width helps; that isolates search/retry limits from the remaining model ranking weakness. Keep this test split out of model selection after this evaluation.

Reproduce with `nix develop -c uv run python scripts/evaluate_graph_feedback.py weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt --split test --beam-width 20 --output experiments/harness-best-test.json`.
