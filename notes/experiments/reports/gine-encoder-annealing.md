# One simple encoder learning-rate schedule

Tested one prespecified schedule, with no cutoff/LR sweep and no contrastive objective. All non-Qwen layers start from scratch. The active scorer is trainable LayerNorm → Linear(1024,256)/ReLU → learned GINE node Linear(256,32) → two message-passing layers → pooled readout, with the original 15 edge channels. Qwen embeddings remain frozen.

The input LayerNorm and 1024→256 projection start at LR 1e-4 and linearly decay to zero at update 1500. From update 1500 through update 3000 they have zero learning rate. GINE and the readout retain LR 1e-4 throughout. Actual encoder weights are checked for exact equality between the start of update 1500 and training end; all four scheduled runs passed. Zero LR also suppresses AdamW weight decay updates.

Four matched seeds (0–3), 968 training and 121 validation puzzles, batch size 16, mixed s2/s3/s4 training, probability-mass objective, AdamW weight decay 1e-6, validation every 100 updates. Both the fresh controls and scheduled runs receive all 3000 updates, so early stopping cannot remove the stable-encoder phase. Best weights are selected by validation s4 Hit@1 then MRR. Auxiliary weight is zero. No test evaluations, notebook edits, or further schedule variants.

| Metric | Constant encoder LR control | Linear encoder LR → zero at 1500 |
|---|---:|---:|
| s4 Hit@1 | 19.83% | 19.21% |
| s4 MRR | 0.2789 | 0.2709 |
| s4 Hit@4 | 32.23% | 30.79% |
| s3 Hit@1 | 21.75% | 20.61% |
| s2 Hit@1 | 29.24% | 27.13% |

| Seed | Control s4 Hit@1 | Scheduled s4 Hit@1 | Control selected update | Scheduled selected update |
|---:|---:|---:|---:|---:|
| 0 | 19.83% | 19.83% | 2400 | 2100 |
| 1 | 18.18% | 20.66% | 2700 | 3000 |
| 2 | 20.66% | 16.53% | 1600 | 2700 |
| 3 | 20.66% | 19.83% | 3000 | 2700 |

The schedule wins one seed, ties one, and loses two. These results do not support adding this schedule to the default training setup. The mean gap is small; a four-seed validation result is not evidence that annealing always hurts. No harness replays were run for this ranking result. Fresh control performance differs from the earlier scratch-model experiments; compare within this matched run, rather than attributing differences from old controls to annealing. GPU numerical determinism is not enforced, and the budget here also differs from earlier early-stopped experiments.

Runner option: `--encoder-anneal-steps 1500`, default zero disables scheduling. Separate encoder and remaining-parameter groups use the same initial LR; no separate projection LR or auxiliary loss can be combined in this controlled experiment. LR is recorded in `training.history[*].encoder_lr`; successful freeze verification is recorded as `training.encoder_frozen_verified`.

Artifacts: `experiments/encoder-annealing/*.json`; distinguish conditions by `config.encoder_anneal_steps` (0/1500) and seed. Checkpoints use matching basenames under `weights/experiments/`. The first control run started before the additional freeze-verification metadata was added, so that inactive check is absent in its JSON; all scheduled runs contain it.

Reproduce with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant standalone-gine-scratch-projected --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --seed SEED --updates 3000 --min-updates-before-stop 3000 --patience 8 --lr 1e-4 --weight-decay 1e-6 --encoder-anneal-steps STOP --output-dir experiments/encoder-annealing`, with STOP 0 or 1500.
