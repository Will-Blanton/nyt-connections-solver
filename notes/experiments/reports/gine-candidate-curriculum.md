# GINE candidate-size curriculum: pairs to triples to quads

Three matched seeds (10–12), validation only, 15,000 updates each, batch 16, evaluation every 250 updates. Same tuned standalone GINE, original 15 edge features, frozen Qwen embeddings and shared train-only text-projection initialization. Active LayerNorm, projection, GINE and readout train throughout. No phonetic additions, auxiliary loss, optimizer/LR changes, early stopping or tuning. Notebooks and test set untouched.

## Intervention

- Every training input is the full 16-word board. The board batch stream and initial model weights are identical across the matched arms.
- Control: 15,000 updates scoring four-word candidates.
- Curriculum: 3,000 updates scoring pairs, 3,000 scoring triples, then 9,000 scoring quads. The same GINE and readout weights transfer between stages; no reset.
- Labels: a candidate is positive exactly when all its words belong to the same official group. All candidates are enumerated. There are 24/120 positive pairs, 16/560 positive triples, and 4/1,820 positive quads.
- The original probability-mass loss is retained: maximize total softmax mass over positive candidates. Correct subsets still compete collectively against incorrect subsets, and the objective can prioritize easier source groups.
- GINE sum aggregation and mean/max readout are unchanged. Only the hard-coded four-word reshape was generalized. The four-word computation is numerically identical and checkpoint shapes/normal loading are unchanged.

Candidate size is distinct from board size. Validation always scores quads on 16/12/8-word canonical states (s4/s3/s2); it never evaluates pair/triple accuracy as the primary metric. Chance rates and loss baselines differ by candidate size, so training losses across stages are not directly comparable. Some officially negative pairs/triples can have other legitimate connections; labels follow only the historical puzzle solution.

All runs complete 15,000 updates. Checkpoint selection is validation full-board Hit@1, MRR tie-break, across all 60 evaluations, including before quad training. Curriculum receives 9,000 quad updates versus 15,000 for control; the matched budget is board examples and optimizer updates, not quad exposure or FLOPs. This is a curriculum within the existing warm-start setup, not text pretraining or training all projections from scratch.

## Results

| Metric | Quad-only | Pairs → triples → quads | Change |
|---|---:|---:|---:|
| s4 hit1 | 30.03% | 30.03% | +0.00 pp |
| s4 mrr | 0.3912 | 0.4037 | +0.0125 |
| s4 hit4 | 45.45% | 48.48% | +3.03 pp |
| s3 hit1 | 30.79% | 32.71% | +1.93 pp |
| s3 mrr | 0.4135 | 0.4336 | +0.0201 |
| s3 hit4 | 49.31% | 51.38% | +2.07 pp |
| s2 hit1 | 35.58% | 38.25% | +2.66 pp |
| s2 mrr | 0.4875 | 0.5074 | +0.0199 |
| s2 hit4 | 59.50% | 60.33% | +0.83 pp |
| Harness solve_rate | 18.46% | 21.21% | +2.75 pp |
| Harness first_guess_accuracy | 31.68% | 34.44% | +2.75 pp |
| Harness mean_correct_groups | 1.333 | 1.410 | +0.077 |

Harness: same 121 validation puzzles, beam 20, full-board scoring, four allowed mistakes, shuffle seed 0, same public-feedback pruning. Saved weights were loaded through the standard harness/checkpoint interface. Smaller-state ranking is oracle evaluation, not feedback trajectories.

| Seed | Variant | s4 Hit@1 | Solves /121 | Selected update |
|---:|---|---:|---:|---:|
| 10 | quad_only | 28.93% | 18 | 750 |
| 10 | pairs_triples_quads | 26.45% | 21 | 6250 |
| 11 | quad_only | 29.75% | 29 | 1000 |
| 11 | pairs_triples_quads | 30.58% | 28 | 6750 |
| 12 | quad_only | 31.40% | 20 | 2750 |
| 12 | pairs_triples_quads | 33.06% | 28 | 6500 |

| Color-specific group Top4 | Quad-only | Curriculum |
|---|---:|---:|
| yellow | 24.52% | 25.62% |
| green | 12.12% | 13.22% |
| blue | 7.71% | 8.26% |
| purple | 2.48% | 2.75% |

| Harness color recovery | Quad-only | Curriculum |
|---|---:|---:|
| yellow | 45.45% | 46.56% |
| green | 38.02% | 37.74% |
| blue | 28.93% | 31.40% |
| purple | 20.94% | 25.34% |

The true full partition is initially retained in the beam on 9.92% of control attempts versus 12.67% with the curriculum.

## Stage progression

Peaks use validation quad Hit@1 on full boards in each time window. Only the curriculum changes candidate size; control uses quads throughout.

| Seed | Arm | First 3,000 peak | Next 3,000 peak | Last 9,000 peak | Final update |
|---:|---|---:|---:|---:|---:|
| 10 | quad_only | 28.93% | 26.45% | 28.10% | 24.79% |
| 10 | pairs_triples_quads | 7.44% | 21.49% | 26.45% | 24.79% |
| 11 | quad_only | 29.75% | 27.27% | 25.62% | 23.14% |
| 11 | pairs_triples_quads | 10.74% | 25.62% | 30.58% | 22.31% |
| 12 | quad_only | 31.40% | 25.62% | 23.14% | 23.14% |
| 12 | pairs_triples_quads | 11.57% | 26.45% | 33.06% | 28.10% |

## Near-miss diagnostics

Raw full-board quad predictions before search. Near misses are incorrect quads with three words from an official group.

| Diagnostic | Quad-only | Curriculum |
|---|---:|---:|
| Top1 is a 3-of-4 negative, all boards | 35.81% | 36.64% |
| Best positive beats best near miss | 31.96% | 33.06% |
| Softmax mass on near misses | 35.77% | 37.72% |

## Interpretation

The curriculum leaves mean full-board Hit@1 exactly unchanged at 30.03%, but improves Hit@4 (45.45→48.48%) and MRR (0.3912→0.4037). Oracle s3/s2 Hit@1 improves modestly. Harness solve rate increases 18.46→21.21%, or 67→77 solves over 363 repeated-seed attempts. Paired solves are 18→21, 29→28, and 20→28; the largest gain comes from seed 12, so this is not uniform across seeds.

There is some global-search benefit despite no Hit@1 gain: initial true partition retention rises 9.92→12.67%, and purple recovery rises 20.94→25.34%. These diagnostics are consistent with better coverage beyond the top candidate, but do not establish a causal explanation. The raw 3-of-4 Top1 error rate does not improve (35.81→36.64%).

Pair-only training has weak direct transfer to quad ranking (peak 7.44–11.57% full-board Hit@1), followed by 21.49–26.45% during triple training. All selected curriculum checkpoints occur shortly after switching to quads: 6,250, 6,750 and 6,500 updates. Later quad training does not surpass those peaks. Low training loss only means the probability-mass objective is satisfied; it does not imply all correct pairs/triples were learned, since a single easy positive can dominate.

This is a modest positive result against this full-board quad-only control, not a new baseline. For context, the earlier fragment-state experiment reached 37.47% Hit@1 and 26.17% solves with different training-state sampling. That cross-experiment comparison is not the matched causal comparison here, and no combination of the two interventions was tested. Keep the durable notebook baseline unchanged. A shorter or interleaved warm-up remains an untested possibility; this run does not establish that candidate-size curriculum in general is ineffective.

Softmax mass depends on score scale; rank comparisons are less sensitive. All seeds reuse the same validation boards; repeated-seed attempts are not independent held-out puzzles. Repeated validation selection makes absolute scores optimistic. This tests one fixed candidate-size curriculum, without combining it with fragment training or the earlier board-size curriculum.

Raw data: [summary.json](gine-candidate-curriculum/summary.json). Curves: [learning-curves.html](gine-candidate-curriculum/learning-curves.html). Near-miss details: [near-miss-audit.json](gine-candidate-curriculum/near-miss-audit.json).

Reproduce: `nix develop -c uv run python scripts/run_gine_candidate_curriculum.py`, then `nix develop -c uv run python scripts/audit_fragment_predictions.py --results experiments/gine-candidate-curriculum`, then `nix develop -c uv run python scripts/summarize_gine_candidate_curriculum.py`.
