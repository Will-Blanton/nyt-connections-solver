# Model-guided feedback policy comparison

Selected checkpoint: `weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt`. Three policies were run on the same 121 validation games, shuffled with seed 0, four allowed mistakes, and beam width 100. The checkpoint, edge features, feasibility pruning, and model-scored candidate groups were fixed. The held-out test set was not used for this policy comparison.

The baseline submits the first group of the highest-scoring complete partition. The **marginal** policy uses the beam's penultimate paths, appends each forced remaining group, deduplicates exact complete partitions, softmax-weights their model path scores at temperature 2, then chooses the feasible group with the most partition weight. The **information** policy adds 0.25 times the normalized entropy of predicted feedback outcomes (correct, one-away, ordinary miss) to that immediate-correct probability. It drops the information term when one mistake remains. This is an approximate belief over beam-retained partitions, not an exact posterior over all valid solutions. It removes duplicate complete partitions but does not enforce diversity among near-identical partitions.

| Policy | Fully solved | First guess correct | Mean correct groups | Mean mistakes | One-away guesses | Mean game time |
|---|---:|---:|---:|---:|---:|---:|
| Best partition | 7/121 (5.79%) | 28/121 (23.14%) | 0.843 | 3.893 | 214 | 1.00 s |
| Partition marginal | 4/121 (3.31%) | 30/121 (24.79%) | 0.851 | 3.942 | 230 | 1.11 s |
| Marginal + feedback information | 5/121 (4.13%) | 30/121 (24.79%) | 0.860 | 3.934 | 232 | 1.12 s |

The marginal policy changed the first guess on 50/121 boards versus baseline, but lost four baseline wins and gained one. Information changed 51/121 first guesses versus baseline; it lost four baseline wins and gained two. Relative to marginal voting, information changed only two first guesses and gained one win. These are paired puzzle counts, not independent samples.

Even though both new policies slightly increased first-guess correctness and mean groups, neither improved complete solves. The sampled partition belief appears too narrow or poorly weighted to guide the four-mistake decision problem. A wider beam was already shown to leave the baseline test policy unchanged between widths 20 and 100. This experiment does not rule out a genuinely diverse or better-calibrated belief pool, nor a deeper feedback-contingent lookahead; it does reject these two simple framings as current replacements.

Per-game guesses, feedback, raw model rank, and post-pruning rank are saved in `harness-{partition,marginal,information}-val.json`. Reproduce with `nix develop -c uv run python scripts/evaluate_graph_feedback.py weights/experiments/20261023T172423Z-gated-gine-projected-compounds-control.pt --split val --beam-width 100 --policy POLICY --output experiments/harness-POLICY-val.json`.
