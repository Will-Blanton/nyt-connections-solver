# One correct group plus partial groups

Three matched seeds (10–12), validation only. Each arm gets 15,000 updates, batch 16, evaluation every 250. Same tuned standalone GINE, original 15 edge features and frozen Qwen input embeddings. Active LayerNorm, text projection, GINE and readout all train. Both arms warm-start the same train-only text projection; GINE/readout initialization is identical per seed. No phonetic additions, auxiliary loss, LR schedule, tuning or early stopping. Notebooks and test set untouched.

## State construction and control

- Both arms use the prior curriculum: 2,500 eight-word updates, 2,500 twelve-word updates, then 10,000 mixed updates in an 8/12/16-word ratio of 1/1/2. Total is exactly 5,000 updates at each size.
- Fragment arm, 8 words: one complete group plus fragments of 2/1/1 words from the three other groups. For 12 words: one complete group plus 2/3/3 fragments. There is exactly one official positive among all 70 or 495 four-word candidates.
- Control: two or three complete groups. Every complete group present is positive; the control is not forced to choose the designated target.
- Both arms enumerate 12 fixed examples per training board per small size, balanced across four designated targets and three distractor configurations. This is 11,616 examples at each small size. Same board/index sampling, target positions, word permutations and DataLoader seed/order across the pair. Fragment choices are deterministic per index/seed, not newly resampled each epoch.
- Full 16-word training boards are unchanged, with four positives, because one complete group plus three strictly incomplete groups can contain at most 13 words.
- Word order is shuffled and relevance computed from original group IDs for each sample. The model sees only frozen word vectors and sliced cached edges, not group IDs, target identity or labels as input.

The probability-mass loss is unchanged mathematically; it now accepts per-example positive masks. With one positive it is standard softmax cross-entropy. This tests removing alternative positive choices while diversifying distractors across all four source groups. Those two effects are not separately isolated here. All canonical validation states and harness puzzles remain unchanged.

Fresh controls were retrained because the paired small-state pools and permutations differ from the previous curriculum runner. These controls, rather than older reported numbers, are the causal comparison. Checkpoint selection is validation full-board Hit@1 with MRR tie-break. Every arm completes the full training budget.

## Results

| Metric | Complete-group control | One correct + fragments | Change |
|---|---:|---:|---:|
| s4 hit1 | 34.16% | 37.47% | +3.31 pp |
| s4 mrr | 0.4375 | 0.4790 | +0.0415 |
| s4 hit4 | 50.96% | 57.02% | +6.06 pp |
| s3 hit1 | 37.26% | 39.60% | +2.34 pp |
| s3 mrr | 0.4784 | 0.5099 | +0.0315 |
| s3 hit4 | 55.30% | 59.64% | +4.34 pp |
| s2 hit1 | 44.08% | 46.65% | +2.57 pp |
| s2 mrr | 0.5631 | 0.5904 | +0.0273 |
| s2 hit4 | 66.30% | 69.65% | +3.35 pp |
| Harness solve_rate | 26.17% | 26.17% | +0.00 pp |
| Harness first_guess_accuracy | 36.09% | 41.87% | +5.79 pp |
| Harness mean_correct_groups | 1.636 | 1.642 | +0.006 |

s2/s3 ranking evaluates oracle reduced states of 8/12 words, not synthetic fragment states. Harness: the same 121 validation puzzles, beam 20, full-board scoring, four allowed mistakes, shuffle seed 0, same public-feedback pruning. All saved weights reconstruct through the normal checkpoint loader.

| Seed | Arm | Full-board Hit@1 | Solves /121 | Selected update |
|---:|---|---:|---:|---:|
| 10 | complete_groups | 32.23% | 29 | 1500 |
| 10 | one_correct_fragments | 36.36% | 28 | 8750 |
| 11 | complete_groups | 36.36% | 30 | 2250 |
| 11 | one_correct_fragments | 37.19% | 32 | 9250 |
| 12 | complete_groups | 33.88% | 36 | 3000 |
| 12 | one_correct_fragments | 38.84% | 35 | 6500 |

| Specific true-group Top4 | Control | Fragments |
|---|---:|---:|
| yellow | 25.90% | 29.48% |
| green | 19.01% | 20.94% |
| blue | 7.44% | 9.09% |
| purple | 3.58% | 5.79% |

| Harness recovery of each color | Control | Fragments |
|---|---:|---:|
| yellow | 49.86% | 51.24% |
| green | 46.56% | 47.93% |
| blue | 38.29% | 36.09% |
| purple | 28.93% | 28.93% |

## Full-board near-miss diagnostic

Unconstrained raw model predictions on validation boards, before search/pruning. A near miss is an incorrect candidate overlapping an official true group in exactly three words. This is not game feedback or a claim that a specific triple is known to the harness.

| Diagnostic | Control | Fragments |
|---|---:|---:|
| Top1 is an incorrect 3-of-4 group (all boards) | 36.64% | 32.51% |
| 3-of-4 among incorrect Top1 choices | 55.65% | 51.98% |
| Best positive beats best 3-of-4 negative | 37.19% | 41.60% |
| Total softmax mass on 3-of-4 negatives | 35.12% | 32.20% |

Near-miss mass also depends on score scale/calibration; it is descriptive. Relative rank versus the best near miss is less sensitive to scale.

## Training progression

| Seed | Arm | s2-only peak s4 Hit@1 | s3-only peak | Mixed-stage peak | Final update |
|---:|---|---:|---:|---:|---:|
| 10 | complete_groups | 32.23% | 31.40% | 32.23% | 23.97% |
| 10 | one_correct_fragments | 29.75% | 36.36% | 36.36% | 33.88% |
| 11 | complete_groups | 36.36% | 35.54% | 33.06% | 30.58% |
| 11 | one_correct_fragments | 28.93% | 32.23% | 37.19% | 29.75% |
| 12 | complete_groups | 33.06% | 33.88% | 32.23% | 27.27% |
| 12 | one_correct_fragments | 28.10% | 34.71% | 38.84% | 30.58% |

## Interpretation

Fragment training improves raw full-board Hit@1 in all three paired seeds (32.23→36.36%, 36.36→37.19%, 33.88→38.84%). Mean Hit@1 increases 34.16→37.47%, MRR 0.4375→0.4790, and Hit@4 50.96→57.02%. Oracle reduced-state metrics improve too. The raw Top1 near-miss rate falls 36.64→32.51%; specific-group Top4 retrieval improves across all four colors. These results support the sampling intervention as a useful ranking experiment.

The harness does not gain in mean solve rate: both arms solve 95/363 repeated-seed attempts (26.17%). Paired solves are 29→28, 30→32, 36→35. First-guess accuracy improves 36.09→41.87%, but mean correct groups barely changes, blue recovery declines slightly, and purple recovery stays at 28.93%. Better initial choice and specific-group retrieval have not translated into stronger full partitions.

First guesses are correct on 131 control attempts versus 152 fragment attempts. Of those, 64 and 75 eventually solve (48.85% and 49.34%). After an incorrect first guess, 31/232 control attempts solve versus 20/211 fragment attempts (13.36% and 9.48%). These conditional populations differ, so this does not prove that fragment training causes worse recovery. The correct full partition is initially retained in the search beam on only 57/363 versus 59/363 attempts, showing little change in this global diagnostic.

Unlike the previous curriculum experiment, selected fragment checkpoints are all in the mixed stage (6,500–9,250 updates). Keep the ranking improvement as a promising candidate, but do not promote it as a solve-rate improvement. A next experiment should assess ranking/coverage of all true groups or partition quality, rather than assume another boost to any-correct Hit@1 will improve whole-puzzle solves. No additional experiment was selected or run from these results.

This is one fixed sampling scheme with three seeds on a repeatedly used validation set. Selected-checkpoint scores are optimistic, and seeds do not provide new held-out puzzles. No claim of test generalization follows. These synthetic states are not legal remaining boards after removing whole correct groups; they are a training intervention to alter competition among candidate sets.

Raw results, selected weights and full histories: [summary.json](gine-fragment-states/summary.json). Near-miss records: [near-miss-audit.json](gine-fragment-states/near-miss-audit.json). Curves: [learning-curves.html](gine-fragment-states/learning-curves.html). Per-seed example files show exact words, source positions and positive candidate indices for all four designated colors at each small size.

Reproduce: `nix develop -c uv run python scripts/run_gine_fragment_states.py`, then `nix develop -c uv run python scripts/audit_fragment_predictions.py`, then `nix develop -c uv run python scripts/summarize_gine_fragment_states.py`.
