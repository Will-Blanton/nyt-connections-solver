# Two-stage additive GINE

Validation only: 121 puzzles, three matched baseline seeds (10/11/12). The three seeds reuse the same puzzles and are not independent held-out samples. No notebooks or test-set evaluation.

## Setup

Reuse the completed original-15-edge mixed-state GINE controls from `gine-curriculum` (15,000 updates each). Freeze each entire selected baseline. Train a separate two-layer GINE correction with width 64 and readout width 64 (baseline widths 128/128). The correction has its own trainable LayerNorm and text projection, initialized from the same shared text-only reference as the controls; fresh GINE weights use seeds 110/111/112. Only the original Qwen embeddings and first-stage model are frozen. All existing 15 edge features are available to both stages; no phonetics, letter edits or hypergraph changes.

Final candidate logit = frozen baseline logit + 0.25 × correction logit. Zero-initialize the correction final output layer, preserving the baseline at update 0. Optimize the existing correct-choice probability-mass loss for 15,000 additional updates, batch 16, balanced mixed s2/s3/s4 states, fixed LR 0.0003016685 and weight decay 0.0000950089. Evaluate every 250 updates, select by s4 Hit@1 then MRR, including the unchanged baseline at update 0. No early stopping, auxiliary loss or LR scheduling. Assert frozen baseline tensors remain bit-for-bit unchanged.

Ensemble control: equally average logits from already trained baseline pairs (10,11), (11,12), (12,10). These require the same aggregate 30,000 training updates per pair as two stages, but use two full-width models, so capacity is not matched. Both approaches add model inference work. The ensemble pairs share components and are correlated. No ensemble weight tuning.

All checkpoints contain both components and load through the normal model registry/harness. Harness: beam 20, original full-board group-score sums, four mistakes, public-feedback pruning, shuffle seed 0.

## Aggregate

| Metric | Baseline | Residual stage | Ensemble |
|---|---:|---:|---:|
| s4 hit1 | 33.61% | 34.71% | 35.54% |
| s4 mrr | 0.4366 | 0.4441 | 0.4560 |
| s4 hit4 | 51.24% | 51.52% | 53.17% |
| s3 hit1 | 36.23% | 36.57% | 38.29% |
| s3 mrr | 0.4697 | 0.4726 | 0.4933 |
| s3 hit4 | 54.61% | 54.89% | 57.78% |
| s2 hit1 | 42.70% | 42.79% | 44.90% |
| s2 mrr | 0.5477 | 0.5487 | 0.5686 |
| s2 hit4 | 64.23% | 64.33% | 65.89% |
| Harness solves | 86/363 (23.69%) | 89/363 (24.52%) | 94/363 (25.90%) |
| Harness first_correct | 126/363 (34.71%) | 124/363 (34.16%) | 133/363 (36.64%) |
| yellow recovered | 179/363 | 179/363 | 189/363 |
| green recovered | 161/363 | 161/363 | 166/363 |
| blue recovered | 122/363 | 121/363 | 133/363 |
| purple recovered | 100/363 | 103/363 | 108/363 |
| Purple selected before final four | 62/363 | 57/363 | 67/363 |
| Targeted letter-edit Purple recovered | 5/24 | 5/24 | 3/24 |

The targeted eight letter-edit puzzles are an analysis subset only; their new features are intentionally disabled in this experiment.

## Per seed

| Seed | Variant | s4 Hit@1 | s3 Hit@1 | s2 Hit@1 | Solves | Purple recovered | Selected update within run |
|---|---|---:|---:|---:|---:|---:|---:|
| 10 | baseline | 35.54% | 39.26% | 44.08% | 33/121 | 37/121 | 8250 |
| 10 | residual | 36.36% | 39.05% | 44.08% | 33/121 | 37/121 | 1250 |
| 10 | ensemble | 35.54% | 39.46% | 46.28% | 29/121 | 35/121 | None |
| 11 | baseline | 33.06% | 35.12% | 41.60% | 28/121 | 35/121 | 7250 |
| 11 | residual | 33.06% | 35.12% | 41.60% | 28/121 | 35/121 | 0 |
| 11 | ensemble | 35.54% | 36.78% | 42.84% | 27/121 | 32/121 | None |
| 12 | baseline | 32.23% | 34.30% | 42.42% | 25/121 | 28/121 | 750 |
| 12 | residual | 34.71% | 35.54% | 42.70% | 28/121 | 31/121 | 500 |
| 12 | ensemble | 35.54% | 38.64% | 45.59% | 38/121 | 41/121 | None |

## Paired changes versus baseline

- residual: 7 new solves, 4 lost solves; 8 new Purple recoveries, 5 lost Purple recoveries.
- ensemble: 23 new solves, 15 lost solves; 24 new Purple recoveries, 16 lost Purple recoveries.

## Interpretation

The residual adds four net full-board correct top choices and three net harness solves across 363 puzzle-seed attempts. All three net harness solves come from seed 12 (25 to 28); seed 10 stays at 33, and seed 11 rejects its trained correction in favor of the unchanged baseline. Selected residual updates are 1,250, 0, and 500 despite completing all 15,000 updates each. Later training lowers training loss while validation performance declines.

Purple recovery rises from 100 to 103, but selections before the forced last four fall from 62 to 57. The aggregate gain therefore does not demonstrate stronger direct Purple recognition. The eight curated letter-edit puzzles have identical per-seed Purple recovery outcomes between baseline and residual, not just identical totals.

The simple ensemble has better aggregate ranking and solve rate than the residual, but its solve improvement is concentrated in one pair (seeds 12 and 10), and it uses more capacity. This is weak evidence for this particular correction-stage setup, not a rejection of all staged training. Keep the current baseline; no default-model change. A relation-aware or hypergraph branch would test complementary evidence rather than asking another GINE with the same inputs to specialize.

Validation: 27 model/checkpoint/diagnostic regression tests passed; all six additive checkpoints were evaluated by the normal harness loader. Frozen first-stage tensors are checked after every training run. Machine-readable metrics, histories and checkpoint paths are in `gine-two-stage/summary.json`.

Results are selected on this repeatedly used validation set. Improvements do not establish performance on unseen puzzles or automatic specialization by connection type. See `gine-two-stage/summary.json` for complete metrics and the saved per-guess harness histories for feedback outcomes.
