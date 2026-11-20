# Original-only learned-edge isolation

The large regression remains when the learned matcher receives only original words. Original-only and full transformations are close in aggregate (59 versus 57 solves), while both trail the existing GINE (96). Extra alternatives are not established as the main cause of the earlier performance drop.

Validation only: three new matched seeds (10/11/12), 30,000 updates each; reuse the completed full-transformation/text-node and original GINE controls. 121 puzzles per seed. Repeated appearances are correlated, and these puzzles have been used in previous experiments. No test-set evaluation or notebook changes.

## Controlled intervention

Keep text nodes, the 64-dimensional transformation encoder, 16-dimensional learned edge vector, pair MLP and max pooling unchanged. Both learned-edge arms have exactly 484,787 trainable parameters, matched initialization, batch permutations and balanced schedules. In the new arm, valid alternatives are masked to type=original only before pair pooling. The payload, padding, parameter layout and number of computed slots remain unchanged. Original alternatives are lowercased frozen Qwen vectors, identical to slot zero of the prior full-transformation arm.

This masks only the learned-edge branch. The original 15 hand-built edge channels, including existing substring similarity metrics, remain present in every arm. This is not a comparison against a system with all transformation information removed. The control column is the ordinary GINE without any added learned edges.

Full budget: 30,000 updates, batch 16, 10,000 updates per 8/12/16-word state size, evaluation every 500 updates, four CPU threads. Ordinary probability-mass loss, LR 0.0003016685, weight decay 0.0000950089, dropout zero, no scheduling, auxiliary/floor loss, pruning or early stopping. Restore best s4 Hit@1 with MRR tie-break. Harness: beam 20, full-board score sums, four mistakes and shuffle seed zero.

A unit test verifies identical initialization/counts, equivalence to a single original-pair calculation, no output/gradient dependence on excluded readings, and checkpoint reconstruction. The new flag defaults to false, preserving existing model behavior.

## Aggregate

| Metric | Existing GINE | Original-only learned edge | Full-transformation learned edge |
|---|---:|---:|---:|
| s4 hit1 | 34.16% | 28.65% | 29.20% |
| s4 mrr | 0.4467 | 0.3852 | 0.3835 |
| s4 hit4 | 52.62% | 46.28% | 45.73% |
| s3 hit1 | 36.09% | 30.85% | 30.99% |
| s3 mrr | 0.4758 | 0.4194 | 0.4168 |
| s3 hit4 | 56.40% | 51.10% | 50.14% |
| s2 hit1 | 42.15% | 37.19% | 38.20% |
| s2 mrr | 0.5516 | 0.4982 | 0.5078 |
| s2 hit4 | 66.35% | 59.23% | 60.24% |
| Harness solves | 96/363 (26.45%) | 59/363 (16.25%) | 57/363 (15.70%) |
| Harness first_correct | 146/363 (40.22%) | 105/363 (28.93%) | 103/363 (28.37%) |
| yellow recovered | 182/363 | 154/363 | 135/363 |
| green recovered | 160/363 | 131/363 | 127/363 |
| blue recovered | 136/363 | 97/363 | 97/363 |
| purple recovered | 106/363 | 75/363 | 70/363 |
| Purple selected | 59/363 | 52/363 | 46/363 |
| Purple forced | 47/363 | 23/363 | 24/363 |
| Concrete covered Purple selected | 0/12 | 0/12 | 0/12 |
| Concrete covered Purple forced | 2/12 | 1/12 | 1/12 |

## Per seed

| Seed | Arm | s4 Hit@1 | Solves | Purple direct / forced | Best update |
|---|---|---:|---:|---:|---:|
| 10 | control | 34.71% | 35/121 | 22 / 17 | 7500 |
| 10 | original-only | 23.97% | 20/121 | 18 / 9 | 500 |
| 10 | edges-text | 26.45% | 17/121 | 17 / 4 | 16500 |
| 11 | control | 34.71% | 33/121 | 22 / 16 | 5000 |
| 11 | original-only | 30.58% | 18/121 | 13 / 9 | 22000 |
| 11 | edges-text | 30.58% | 20/121 | 14 / 10 | 1000 |
| 12 | control | 33.06% | 28/121 | 15 / 14 | 2000 |
| 12 | original-only | 31.40% | 21/121 | 21 / 5 | 24000 |
| 12 | edges-text | 30.58% | 20/121 | 15 / 10 | 17000 |

## Paired changes

- edges-text_versus_original-only: 25 new solves, 27 lost; 28 new Purple recoveries, 33 lost.
- original-only_versus_control: 15 new solves, 52 lost; 21 new Purple recoveries, 52 lost.

## Concrete targeted cases

| Mechanism | Existing GINE | Original-only | Full transformations |
|---|---:|---:|---:|
| starting with rock bands | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced |
| ending in synonyms for "location" | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced |
| starting with slang for zero | 0 direct / 2 forced | 0 direct / 1 forced | 0 direct / 1 forced |
| starting with synonyms for "slush" | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced |

These four intended-mechanism examples were chosen in the previous coverage audit; all sixteen readings are available in the full payload. They are analysis-only labels.

## Conclusions

Adding full transformations changes 16-word Hit@1 from 28.65% to 29.20% (+0.55 percentage points), while solves change from 59 to 57 (-0.55 points). Per-seed solves are 20/18/21 original-only versus 17/20/20 full. Full transformations gain 25 paired solves and lose 27: the effect is not uniformly negative or positive. These small mixed aggregate differences do not establish equivalence, but do not show a useful consistent improvement either.

The dominant loss is already present in the original-only learned branch: 96 to 59 solves versus ordinary GINE, with losses in every seed. Thus the earlier full-transformation result should not be attributed solely to substring alternatives or their extraction. The learned matcher, added capacity and optimization interactions are stronger candidates for investigation. This experiment does not separately identify which one is responsible.

Full transformations do not show targeted Purple improvements here: overall Purple recovery changes 75 to 70 and direct selections 52 to 46 versus original-only. Neither learned-edge arm directly selects any of the four coverage-audited Purple groups. The one rock-band success in the earlier study belonged to the fused-node arm, which was deliberately not varied in this isolation.

All three new runs completed 30,000 updates and sixty validation observations. Two selected checkpoints occur after 15,000 updates (22,000 and 24,000), improving on their first-half Hit@1. Final training losses are approximately 0.05; insufficient ability to fit the training objective is not the observed problem. Dataset size remains a plausible generalization limit, not a demonstrated sole cause.

Keep the baseline and do not discount the entire transformation family. The next informative test is a smaller, structured matcher (shared projection/cosine with explicit operation/position evidence), rather than widening this pair MLP or merely extending its budget. No such follow-up was run as part of this requested isolation.

Seven targeted tests passed. The original-only mask preserves parameter counts and initialization, matches a single-original-pair calculation, excludes non-original outputs and gradients, and survives checkpoint reconstruction. The independent 1,089-board audit reproduces all nine selected full-board Hit@1 values.

## Interpretation limits

This isolates the presence of additional alternatives in this matcher. It does not distinguish useful transformed content from the effects of extra comparisons, positional variety, or elementwise-max selection. Same parameter count does not imply identical effective complexity. These are matched existing optimizer settings, not tuned upper bounds. Do not extrapolate a negative result to all transformation representations.

All nine checkpoints are audited on all 1,820 full-board candidates, independently reproducing selected Hit@1. Checkpoint paths, per-game guesses, curves and per-state metrics are retained under `experiments/gine-original-only/`. Defaults remain unchanged.

Run: `nix develop -c uv run python scripts/run_gine_original_only.py`. Audit: `nix develop -c uv run python scripts/audit_gine_probability_floor.py --results experiments/gine-original-only`. Summary: `nix develop -c uv run python scripts/summarize_gine_original_only.py`.
