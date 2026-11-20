# Fade the contrastive auxiliary loss out early

One prespecified schedule: auxiliary weight starts at 0.05 on update 1, decreases linearly to zero at update 1500, and remains zero through update 3000. Temperature is 0.1. The second half trains on probability-mass ranking loss alone. Encoder learning rate is constant throughout: all active non-Qwen layers remain trainable. No encoder freeze, separate encoder LR, or cutoff/weight sweep is combined with this experiment.

The same scratch LayerNorm → Linear(1024,256)/ReLU → GINE node Linear(256,32) → two GINE layers → pooled readout is used, with frozen Qwen embeddings and the original 15 edges. Auxiliary loss is the existing puzzle-local supervised contrastive objective on the projected 256-dimensional words, three positives per word, other groups in the same state as negatives, self excluded. It adds no parameters or inference cost. Only training puzzles supply auxiliary pairs.

Four matched seeds (0–3), same 968 training/121 validation puzzles, batch size 16, mixed s2/s3/s4, AdamW LR 1e-4, weight decay 1e-6, fixed 3000-update budget, validation every 100, selected by s4 Hit@1 then MRR. Controls are the constant-encoder-LR, no-auxiliary runs from `experiments/encoder-annealing/`, which have the same fixed budget. Existing test puzzles and notebooks were untouched.

| Metric | No auxiliary loss | Auxiliary fades to zero |
|---|---:|---:|
| s4 Hit@1 | 19.83% | 19.63% |
| s4 MRR | 0.2789 | 0.2654 |
| s4 Hit@4 | 32.23% | 29.75% |
| s3 Hit@1 | 21.75% | 21.18% |
| s2 Hit@1 | 29.24% | 28.44% |

| Seed | Control s4 Hit@1 | Scheduled auxiliary s4 Hit@1 | Auxiliary selected update |
|---:|---:|---:|---:|
| 0 | 19.83% | 21.49% | 2200 |
| 1 | 18.18% | 19.83% | 2200 |
| 2 | 20.66% | 12.40% | 2800 |
| 3 | 20.66% | 24.79% | 2900 |

Three seeds improve, but seed 2 declines sharply. Mean Hit@1 is effectively unchanged, with lower MRR, Hit@4, and reduced-state Hit@1. This does not establish a reliable benefit and does not justify changing defaults. No more schedules were tried and no harness replays were run for this result. As with earlier runs, GPU numerical determinism is not enforced; conclusions concern this small repeated validation comparison.

Implementation: `--aux-anneal-steps` defaults to zero (constant auxiliary weight). The shared `linear_to_zero` schedule also supplies encoder LR scheduling. When effective auxiliary weight is zero, auxiliary computation stops; `history.aux_loss` is then zero because no contrastive loss is evaluated, not because learned contrastive error is zero. End-of-interval effective weight is recorded as `history.aux_weight`. The model remains fully trainable except Qwen.

Artifacts: `experiments/auxiliary-annealing/*.json`, corresponding checkpoints under `weights/experiments/`. Reproduce with the scratch-projected variant, seeds 0–3, fixed 3000 updates (`--min-updates-before-stop 3000`), LR 1e-4, weight decay 1e-6, and `--aux-weight 0.05 --aux-temperature 0.1 --aux-anneal-steps 1500 --output-dir experiments/auxiliary-annealing` using the usual text-only reference for split/feature metadata.
