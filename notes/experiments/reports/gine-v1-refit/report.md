# Train+validation refit: previous versus CV-selected regular GINE

Both setups initialize all active learned layers from scratch on the original train+validation pool (1,089 puzzles). Only original cached Qwen vectors stay frozen. Core 15 edge features only; coherence and direct edge-summary branch disabled. The previous setup omits its earlier 11 unused, zero-padded/masked edge columns; all 15 active channels and learned architecture are preserved. Three fixed seeds: 10, 11, 12.

No validation set or scheduler is used during refitting. The fixed final update is selected before test evaluation: median prior validation-selected checkpoint exposure, scaled by final/source training puzzle counts and rounded to a multiple of six. Every state size gets equal updates, batch 16. Constant AdamW learning rate and original floor loss (.1/.1).

| Setup | Source selected updates | Fixed refit updates |
|---|---|---:|
| previous | [8000, 7500, 2500] | 8436 |
| cv | [1500, 2500, 1000, 2500, 1500] | 1878 |

| Parameter | Previous | CV |
|---|---:|---:|
| hidden_width | 128 | 64 |
| gine_layers | 2 | 3 |
| readout_width | 128 | 64 |
| projection_width | legacy text MLP | 256 |
| dropout | 0.0 | 0.0 |
| lr | 0.0003016684667937828 | 0.0009802617130802425 |
| weight_decay | 9.500890359387789e-05 | 0.005032124249909019 |

All six checkpoints were saved and SHA-256 sealed before test feature preparation or evaluation. Test chooses only between the two frozen setups; it does not tune their individual hyperparameters, update budgets, or seeds. The original test split has been inspected in earlier experiments, so this comparison is not a pristine final holdout. Future live puzzles remain the prospective evaluation.

Harness: beam 20, original full-board scores, four mistakes, shuffle seed 0. Reduced-state ranking uses oracle remaining groups. Three seeds repeat the same 121 test puzzles; 363 appearances are correlated.

| Metric | Previous | CV |
|---|---:|---:|
| s4 hit1 | 0.3278 | 0.3251 |
| s4 mrr | 0.4367 | 0.4532 |
| s4 hit4 | 0.5344 | 0.5647 |
| s3 hit1 | 0.3616 | 0.3760 |
| s3 mrr | 0.4748 | 0.4993 |
| s3 hit4 | 0.5702 | 0.6081 |
| s2 hit1 | 0.4376 | 0.4550 |
| s2 mrr | 0.5572 | 0.5764 |
| s2 hit4 | 0.6625 | 0.6827 |
| solves | 75 | 75 |
| first_correct | 136 | 142 |
| mean_groups | 1.4104683195592287 | 1.4958677685950412 |

| Seed | Previous Hit@1 | CV Hit@1 | Previous solves | CV solves | CV gains / losses |
|---|---:|---:|---:|---:|---:|
| 10 | 29.75% | 33.06% | 26 | 22 | 9 / 13 |
| 11 | 32.23% | 33.06% | 31 | 28 | 11 / 14 |
| 12 | 36.36% | 31.40% | 18 | 25 | 17 / 10 |

| Color | Previous direct / forced | CV direct / forced |
|---|---:|---:|
| purple | 73 / 18 | 51 / 28 |
| blue | 62 / 33 | 76 / 31 |
| green | 122 / 19 | 143 / 12 |
| yellow | 180 / 5 | 198 / 4 |

Selected setup: **cv** by the predeclared solve-count, first-guess, Hit@1 tie-break rule. Deployment checkpoint uses predeclared seed 10, not the best test seed.

Checkpoint: `weights/experiments/20261107T055124Z-v1-refit-cv-seed-10.pt`

## Primary seed-10 checkpoint

Hit@1 33.06%; 22/121 solves; first-guess accuracy 35.54%.

Complete histories, independent group ranks, loss curves, configuration and budget provenance are retained. Checkpoints truthfully list all 1,089 fitted dates as training and no validation dates; the original split is separately preserved. Test was not used for training or scheduling. Notebooks are unchanged.


All six sealed checkpoints reconstruct through the production loader, retain their original SHA-256, and reproduce test Hit@1 on CPU. The refit-budget and fixed-training tests pass; `git diff --check` passes.
