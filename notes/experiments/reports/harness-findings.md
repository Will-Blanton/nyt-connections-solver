# Feedback harness findings

This note consolidates the search-harness experiments for the validation-selected graph checkpoint `weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt`. Unless stated otherwise, each evaluation used 121 historical puzzles, the same shuffled word order (seed 0), four allowed mistakes, exact public-feedback pruning, and no answer labels in the policy. The checkpoint is a frozen text scorer with a projected-input GINE correction; its original 15 edge channels are active and its two experimental closed-compound channels are masked.

Later architecture work found that a standalone GINE group scorer substantially outperformed the residual scorer on validation games using the same original-board harness score. See `standalone-gine.md`. The policy comparisons below remain scoped to the earlier fixed checkpoint; they should not be read as the standalone model's results.

## What the harness already does

After an ordinary incorrect guess, every proposed group sharing **three or four words** with it is rejected. A one-away response requires at least one true group to overlap that guess by exactly three words, without assuming which three. The harness also requires each retained path to admit a full partition consistent with all feedback. It rescales model scores after pruning and replans after each guess. Thus it cannot repeatedly submit the same failed group or a group with 3/4 overlap with an ordinary miss. These constraints are in `src/nyt_connections_solver/feedback_search.py`.

## Main measured results

| Comparison | Validation full solves | Test full solves | Finding |
|---|---:|---:|---|
| Original path score, beam 20 | 7/121 | 17/121 | Reference policy; test first guess correct 40/121. |
| Original path score, beam 100 | 7/121 | 17/121 | Wider beam did not improve solves or test first choices; slower. |
| Vote across beam partitions, beam 100 | 4/121 | Not run | More correct first guesses than baseline (30 vs 28 on validation), fewer complete solves. |
| Partition vote plus feedback information, beam 100 | 5/121 | Not run | Changed only two first guesses versus voting; no solve gain over baseline. |
| Original path plus original-board group scores, beam 20 | 10/121 | 17/121 | Helped validation, not test solve count. |
| **Replace path score with sum of four original-board group scores**, beam 20 | **14/121** | **19/121** | Best tested harness scoring rule; test gain was two puzzles. |
| Original-board sum, beam 100 | 14/121 | Not run | Same validation solves as width 20 despite broader beam coverage. |
| Original-board sum plus 0.25 × weakest-group score, beam 20 | 13/121 | Not run | Slightly worse than plain sum. |

The original path score sums probabilities normalized separately at 16-, 12-, and 8-word states; its final forced group contributes no model score. This makes path scores hard to compare across branches. The original-board rule uses the same model's four group scores from the initial 16-word board, including the group that would otherwise be forced. It changed the first guess on 56/121 boards on both validation and test. Validation mean correct groups rose from 0.843 to 1.074; test rose from 1.157 to 1.256. First-guess accuracy changed much less: 28 to 30 correct on validation, and 40 to 39 on test. Its benefit is therefore broader than simply improving the first move. Search time at beam 20 remained about 0.25 seconds per game in the recorded runs.

## Failure modes and limits

- The original test harness fully solved 17/121 puzzles, despite 40/121 correct first guesses. It averaged 1.16 correct groups and 3.79 mistakes. Games ended with 0, 1, 2, or 4 correct groups in counts 47, 42, 15, and 17. Three correct groups force the fourth. It recovered Yellow/Green/Blue/Purple groups 50/48/22/20 times across test games: Blue and Purple remain much harder.
- One-away feedback is frequent, but recovery is weak. In the original test run, 104/121 games had at least one one-away response and only 15 of those games were solved. After a one-away guess with another turn available, the next guess was correct 33/164 times. These figures come from saved per-guess histories, not the coarse mistake count.
- Beam width is not the main lever for this checkpoint. On test, widths 1, 5, 20, and 100 all solved 17/121 under the original path score. Width 20 picked something other than the raw top-ranked first group on 30/121 boards; width 100 made no different first choices or solved outcomes. Feedback pruning matters: at width 20, only 198/599 submitted guesses were raw rank 1, while 417/599 were rank 1 among feedback-feasible candidates.
- The true partition is rarely retained near the end of the **initial** beam. Under original-board scoring, it appeared on 4/121 validation boards at width 20 and 15/121 at width 100. At width 20 all four surviving true partitions ranked first. At width 100 four ranked first; the other eleven ranked 5–85. The wider beam still solved 14/121. Initial beam absence does not prevent later success because the policy replans after feedback. The diagnostic points to both candidate-path generation and model partition ranking, not just final reranking.
- Voting across the retained partitions is a weak approximation to a belief state: near-identical partitions remain numerous, and the path scores are not calibrated posterior probabilities. That experiment does not rule out a better belief-state policy. The simple weakest-group penalty likewise does not rule out a learned partition-quality model; it only failed at the one prespecified weight.

## Decision and evaluation boundary

For harness use with this checkpoint, `score_mode=full_board` and beam width 20 is the most promising tested setting. It is optional in code; `path` remains the default. The improvement was clear on validation and modest on test, so it should be confirmed on genuinely new puzzles before becoming a strong performance claim. **Do not use the existing 121-date test split for further harness selection:** it has been inspected repeatedly. New daily puzzles or a freshly ingested chronological holdout are the appropriate next evaluation data. Continue development analysis on validation.

The next substantive work should improve the model's group/partition ranking or construct a more diverse set of candidate partitions. More beam width, a hand-tuned weakest-group penalty, and simple partition voting have not shown reliable solve-rate gains here.

## Source reports and artifacts

- `harness-best-test.md`: original test outcomes and per-feedback failure modes.
- `harness-beam-widths.md`: widths 1, 5, 20, 100; raw versus feasible candidate rank.
- `harness-policy-framings.md`: partition voting and information-value policies on validation.
- `harness-full-board-quality.md`: additive original-board score.
- `harness-original-board-score.md`: replacement score on validation and test.
- `harness-weakest-group-and-beam-audit.md`: weakest-group penalty and true-partition beam ranks.
- Corresponding `experiments/harness-*.json` files hold complete per-game histories and diagnostics. The runner is `scripts/evaluate_graph_feedback.py`.
