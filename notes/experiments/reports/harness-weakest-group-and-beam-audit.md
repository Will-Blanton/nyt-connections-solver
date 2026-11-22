# Weakest-group penalty and true-partition beam audit

Validation only. Fixed checkpoint `weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt`, same 121 validation puzzles, shuffle seed 0, four mistakes. The baseline score sums all four groups' original-16-word log probabilities. The single prespecified experiment adds `0.25 * min(group log probability)` to that sum, penalizing a partition with one especially weak group. No model training or feature changes. Beam width 20 for the penalty comparison.

| Validation policy | Full solves | First guess correct | Mean correct groups | Purple groups found |
|---|---:|---:|---:|---:|
| Sum of four original-board scores | 14/121 (11.57%) | 30/121 (24.79%) | 1.074 | 18 |
| Sum plus weakest-group penalty | 13/121 (10.74%) | 29/121 (23.97%) | 1.033 | 18 |

The penalty changed first guesses on 9/121 boards, gained two solves and lost three. It did not improve this validation set. A low-scoring Purple group is a plausible cost of favoring only balanced partitions, though this run does not establish that as the causal mechanism. The sum remains the better tested score.

## Where the true partition goes

For each first decision, the diagnostic reconstructs the complete partitions represented by the beam immediately before its final forced group, and ranks them by the same search score. This inspects the model/search output after the fact; no gold answers enter the policy. Because the harness can replan after feedback, absence from the initial beam does not make eventual success impossible.

| Beam width | True partition present on initial board | Rank when present | Full solves |
|---:|---:|---|---:|
| 20 | 4/121 | All four rank 1 | 14/121 |
| 100 | 15/121 | Four rank 1; other eleven rank 5–85 (median of all fifteen: 43) | 14/121 |

At width 20, the true partition is rarely generated/retained among near-complete paths. Width 100 retains it more often, but usually scores it far below the winning path and does not increase solves. This points to both beam generation and model partition ranking as limitations. It does not support another simple final-score penalty as the immediate priority. Feedback still rescues some boards whose true partition is absent initially.

Raw game and rank diagnostics: `experiments/harness-weakest-0-val.json`, `experiments/harness-weakest-0.25-val.json`, and `experiments/harness-weakest-0-beam100-val.json`. Reproduce with `nix develop -c uv run python scripts/evaluate_graph_feedback.py weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt --split val --score-mode full_board --beam-width WIDTH --weakest-group-weight WEIGHT --output OUTPUT.json`.
