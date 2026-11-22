# GINE state-size curriculum

Validation only: three matched seeds (10–12), 15,000 updates per arm, batch 16, evaluation every 250 updates. Same original 15 edge features and frozen Qwen text inputs as the durable GINE notebook baseline; no phonetic node/edge additions. Active input LayerNorm, text projection, two GINE layers, and readout all train. Tuned width/readout 128, dropout 0, fixed AdamW LR 0.0003016684667937828, weight decay 9.500890359387789e-05. No auxiliary loss, LR scheduling, early stopping, or tuning. Notebooks and test puzzles were not evaluated or modified. Both arms use the same train-only text projection initialization from the baseline reference checkpoint (not a frozen projection); GINE/readout are initialized identically per seed. This tests curriculum in the existing warm-start training setup, not training everything from random initialization.

## Controlled schedule

- Mixed control: shuffle one update each of s2, s3, and s4 in successive blocks.
- Curriculum: 2,500 updates of s2, then 2,500 of s3; final 10,000 updates shuffle blocks of s2, s3, s4, s4.
- Both arms consume exactly 5,000 batches of each state size, with identical per-size DataLoader seeds/order and identical per-seed starting weights. Only update ordering changes. s2/s3/s4 mean 8/12/16 remaining words.

Checkpoint selection is validation s4 Hit@1, MRR tie-break, across all 60 evaluations. Selected checkpoints need not be the final update. Every run still completes all 15,000 updates. s2/s3 validation is oracle reduced states; harness results use actual feedback trajectories. Harness: 121 validation puzzles, beam 20, full-board scoring, four allowed mistakes, shuffle seed 0, same pruning and standard checkpoint loader.

| Metric | Mixed control | Curriculum | Change |
|---|---:|---:|---:|
| s4 hit1 | 33.61% | 34.99% | +1.38 pp |
| s4 mrr | 0.4366 | 0.4429 | +0.0064 |
| s4 hit4 | 51.24% | 52.07% | +0.83 pp |
| s3 hit1 | 36.23% | 37.88% | +1.65 pp |
| s3 mrr | 0.4697 | 0.4844 | +0.0147 |
| s3 hit4 | 54.61% | 56.27% | +1.65 pp |
| s2 hit1 | 42.70% | 44.40% | +1.70 pp |
| s2 mrr | 0.5477 | 0.5635 | +0.0158 |
| s2 hit4 | 64.23% | 66.21% | +1.97 pp |
| Harness solve_rate | 23.69% | 25.34% | +1.65 pp |
| Harness first_guess_accuracy | 34.71% | 36.36% | +1.65 pp |
| Harness mean_correct_groups | 1.548 | 1.576 | +0.028 |

| Seed | Arm | s4 Hit@1 | Solves /121 | Selected update |
|---:|---|---:|---:|---:|
| 10 | mixed | 35.54% | 33 | 8250 |
| 10 | curriculum | 33.06% | 31 | 1750 |
| 11 | mixed | 33.06% | 28 | 7250 |
| 11 | curriculum | 34.71% | 32 | 2000 |
| 12 | mixed | 32.23% | 25 | 750 |
| 12 | curriculum | 37.19% | 29 | 4250 |

| Color | Mixed group Top4 | Curriculum group Top4 |
|---|---:|---:|
| yellow | 27.27% | 26.45% |
| green | 17.63% | 18.46% |
| blue | 5.23% | 7.16% |
| purple | 3.58% | 4.13% |

## Training progression

Peak validation s4 Hit@1 within each time window. For curriculum these are s2-only, s3-only, and mixed-size stages; the control is mixed throughout. The final column is the unselected final update, not the restored checkpoint.

| Seed | Arm | First 2,500 peak | Next 2,500 peak | Last 10,000 peak | Final update |
|---:|---|---:|---:|---:|---:|
| 10 | mixed | 31.40% | 34.71% | 35.54% | 24.79% |
| 10 | curriculum | 33.06% | 32.23% | 31.40% | 22.31% |
| 11 | mixed | 33.06% | 31.40% | 33.06% | 23.97% |
| 11 | curriculum | 34.71% | 33.88% | 31.40% | 27.27% |
| 12 | mixed | 32.23% | 28.10% | 28.93% | 26.45% |
| 12 | curriculum | 33.88% | 37.19% | 33.88% | 27.27% |

## Interpretation

The curriculum gives a modest aggregate improvement: full-board Hit@1 rises 33.61% to 34.99%, and harness solve rate rises 23.69% to 25.34%. Paired solves are 33→31, 28→32, and 25→29: two seeds improve, one worsens. The net six extra solves are over 363 repeated-seed game attempts, not six new unique puzzles. s2/s3 ranking improves slightly too. Specific-group Top4 improves for green, blue, and purple, with a small yellow decline; purple remains very weak.

All three selected curriculum checkpoints precede full-board training: updates 1,750 and 2,000 during s2-only training, and 4,250 during s3-only training. None of the later mixed/full-board stages surpasses its earlier peak. Thus the evidence supports useful transfer from smaller states, but does not show that this build-up schedule improves performance after reaching full boards. Final update scores are lower than selected scores in both arms; lower training loss does not translate into sustained validation improvement.

Keep this as a small positive experimental result, without promoting it as a reliable new baseline. The one-correct-group-plus-fragments proposal is a useful next isolated experiment: it changes the hard negatives and forces selection of the remaining positive rather than testing presentation order alone. No additional schedule search was performed on these results.

Interpret the paired seeds and learning curves together. This compares a single predeclared ordering schedule, not all possible curricula. It holds total updates and per-size exposure fixed, so it tests ordering rather than extra training on small states. Seeds reuse the same validation puzzles and are not independent held-out datasets. Checkpoint selection and repeated experiments on this validation set make absolute scores optimistic.

The proposed alternative—one complete correct group mixed with fragments of other groups—is not included. It needs example-specific relevance labels rather than the current canonical complete-partition labels. It could directly train rejection of near-miss groups while leaving exactly one positive; this would change the training-state distribution and available correct choices, not merely ordering. It is a separate follow-up, not evidence provided by this curriculum experiment.

Raw results, full histories, schedules, checkpoints and harness records: [summary.json](gine-curriculum/summary.json). Interactive learning curves: [learning-curves.html](gine-curriculum/learning-curves.html).

Reproduce: `nix develop -c uv run python scripts/run_gine_curriculum.py` then `nix develop -c uv run python scripts/summarize_gine_curriculum.py`.
