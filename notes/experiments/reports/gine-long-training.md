# Extend plain scratch GINE training

Tested training duration before changing batch size. Batch size remains 16 puzzles/states per optimizer update; enumerated candidate groups are correlated choices within each puzzle, not independent batch examples. This experiment does not determine whether a smaller batch would help.

Four seeds (0–3), scratch LayerNorm → Linear(1024,256)/ReLU → GINE node Linear(256,32) → two message-passing layers → pooled readout. Only Qwen embeddings are frozen in the active model. Original 15 edges, same 968 training/121 validation puzzles, probability-mass objective, AdamW LR 1e-4, weight decay 1e-6, evaluation every 100 updates. No auxiliary loss, LR schedule, or encoder freeze. Every run receives the full 10,000 updates, restoring the checkpoint selected by validation s4 Hit@1 then MRR. Existing test puzzles and notebooks were untouched.

The clean duration comparison selects within the first 3000 versus all 10000 updates **of the same trajectory**, rather than comparing different reruns. Only the best overall checkpoint is saved, so the shorter-budget checkpoint is represented by ranking history, not a separate harness replay.

| Validation metric | Best within 3000 | Best within 10000 |
|---|---:|---:|
| s4 Hit@1 | 21.49% | 23.76% |
| s4 MRR | 0.2962 | 0.3188 |
| s4 Hit@4 | 35.33% | 36.78% |
| s3 Hit@1 | 22.93% | 24.95% |
| s2 Hit@1 | 29.17% | 31.68% |

| Seed | Early selected update | Early s4 Hit@1 | Later selected update | Later s4 Hit@1 |
|---:|---:|---:|---:|---:|
| 0 | 2400 | 19.83% | 6400 | 23.97% |
| 1 | 2400 | 20.66% | 7100 | 22.31% |
| 2 | 2800 | 23.97% | 6900 | 24.79% |
| 3 | 2800 | 21.49% | 5900 | 23.97% |

Longer training finds a better checkpoint in every seed. Mean mixed-state training loss drops from 1.215 at update 3000 to 0.431 at update 10000. Validation does not improve monotonically: mean s4 Hit@1 at the final update is only 19.83%, below the selected checkpoints. Validation Hit@1 averaged over updates 2001–3000 is about 17.95%, versus 19.57% over 9001–10000, giving some evidence of a sustained improvement beyond additional checkpoint-selection opportunities. Still, selecting from 100 evaluations instead of 30 increases validation-selection optimism; this is not held-out generalization evidence.

At 10000 updates, each state-size loader gets about 3333 updates: approximately 54.6 passes over s4, 13.8 over s3, and 9.2 over s2. At 3000 updates those were 16.4, 4.1, and 2.8. There is no single epoch count because state sizes are interleaved equally despite different numbers of reduced states.

## Validation feedback harness

Selected longer-trained checkpoints, same 121 puzzles, four mistakes, beam width 20, full-board scoring, shuffle seed 0. Solves by seed: 14, 21, 14, 14. Mean is 15.75/121 (13.02%), first-guess accuracy 27.89%, mean recovered groups 1.079. Recovered groups across four repeated runs: Yellow 188, Green 145, Blue 106, Purple 83 (484 repeated opportunities per color).

For context, the earlier scratch-256 3000-update harness averaged 11.25/121 (9.30%), while the transferred frozen-projection standalone GINE averaged 19.5/121 (16.12%). These are different saved trajectories, not a paired harness estimate of the duration effect. The within-trajectory ranking comparison above is the stronger controlled result.

Conclusion: 3000 updates curtailed useful learning for this plain model. Use the longer plain-training baseline before adding schedules or losses. Batch size remains a separate question; compare future batch sizes at matched sample exposure, not simply equal updates (batch 8 would need roughly twice as many updates to match batch 16 exposure).

Artifacts: `experiments/long-training/*standalone-gine-scratch-projected.json`, corresponding checkpoints under `weights/experiments/`, `experiments/long-training/summary.json`, and interactive learning curves in [learning-curves.html](long-training/learning-curves.html). Harness histories: `experiments/harness-long-training-{seed}-val.json`. Regenerate summary/curves using `nix develop -c uv run python scripts/summarize_long_gine_training.py` (uses the existing Altair dependency).

Reproduce training with the usual reference checkpoint and `--variant standalone-gine-scratch-projected --seed SEED --batch-size 16 --updates 10000 --min-updates-before-stop 10000 --patience 8 --lr 1e-4 --weight-decay 1e-6 --output-dir experiments/long-training`.
