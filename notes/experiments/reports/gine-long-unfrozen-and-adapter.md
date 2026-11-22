# Longer training with unfrozen input layers and residual adapter

Extended two previously tested configurations to 10000 updates, without schedules or auxiliary losses. Four seeds (0–3), batch size 16, AdamW LR 1e-4, weight decay 1e-6, evaluation every 100, full update budget, checkpoint selection by validation s4 Hit@1 then MRR. Same 968 training/121 validation puzzles and original 15 edge channels. Only Qwen embeddings are frozen in the active computation. No held-out test evaluation or notebook edits.

1. **Exact unfrozen original projection:** input LayerNorm and Linear(1024,256)/ReLU start from the text-only checkpoint's learned weights and are both trainable. GINE's learned node Linear(256,32), two message-passing layers and readout also train. This matches the earlier frozen-projection standalone architecture while allowing its input layers to update.
2. **Single-layer residual adapter:** fresh trainable LayerNorm, `z + ReLU(Linear(1024,1024)(z))`, learned GINE node Linear(1024,32), same message-passing structure and readout. All active non-Qwen parameters start from scratch.

The class retains unused frozen attention modules as checkpoint scaffolding; they do not affect outputs. The earlier long plain scratch experiment is shown for context. Duration comparisons below select early versus later checkpoints **within each trajectory**, so they are not comparisons against different numerical reruns.

| Model | s4 Hit@1 within 3000 | s4 Hit@1 within 10000 | s4 MRR early → later | s3 Hit@1 early → later | s2 Hit@1 early → later |
|---|---:|---:|---:|---:|---:|
| Original pretrained input projection, unfrozen | 21.69% | 24.59% | 0.2940 → 0.3183 | 23.19% → 24.95% | 29.34% → 31.44% |
| Single-layer residual adapter | 9.30% | 11.36% | 0.1441 → 0.1815 | 10.90% → 14.46% | 20.66% → 24.62% |
| Plain scratch projection, prior long experiment | 21.49% | 23.76% | 0.2962 → 0.3188 | 22.93% → 24.95% | 29.17% → 31.68% |

| Seed | Unfrozen projection early → later | Selected update | Residual adapter early → later | Selected update |
|---:|---:|---:|---:|---:|
| 0 | 23.14% → 26.45% | 6000 | 14.05% → 15.70% | 8300 |
| 1 | 23.97% → 25.62% | 7400 | 6.61% → 9.09% | 8600 |
| 2 | 19.83% → 24.79% | 8800 | 6.61% → 9.09% | 7200 |
| 3 | 19.83% → 21.49% | 5500 | 9.92% → 11.57% | 5900 |

All seeds improve with longer training in both configurations. The original unfrozen projection benefits more usefully than the residual adapter, whose ranking remains weak. The unfrozen projection's 24.59% mean still trails the earlier frozen-projection reference's 28.51%. Longer training explains part of the earlier difference, but does not close it. Selecting among 100 validation checkpoints versus 30 introduces additional validation-selection optimism; no claim of held-out generalization is made.

## Validation harness

The longer-trained exact-unfrozen-projection checkpoints were replayed on 121 validation puzzles with four mistakes, beam width 20, full-board scoring, shuffle seed 0. Solves by seed: 11, 17, 12, 14. Mean 13.5/121 (11.16%); first-guess accuracy 27.27%; mean recovered groups 1.012. Recovered groups across repeated runs: Yellow 183, Green 139, Blue 99, Purple 69 (484 repeated opportunities per color). No residual-adapter harness replay was run because ranking remained weak.

For context, the longer plain scratch model averaged 15.75/121 (13.02%) and the earlier frozen-projection reference averaged 19.5/121 (16.12%). These are different saved runs; a better mean s4 ranking result for the pretrained unfrozen projection did not produce a better harness solve rate than the long scratch model. The shorter within-trajectory checkpoints are not separately saved, so there is no paired short-versus-long harness estimate.

Artifacts: `experiments/long-unfrozen-projection/` and `experiments/long-adapter-training/` contain per-run JSON, `summary.json`, and `learning-curves.html`; checkpoints share the run basenames under `weights/experiments/`. Harness histories: `experiments/harness-long-unfrozen-{seed}-val.json`.

Use the existing runner with `--variant standalone-gine-trainable-projection --train-input-norm` for the original unfrozen setup, or `--variant standalone-gine-adapter-single` for the residual adapter. Shared options: usual text-only reference checkpoint, seeds 0–3, `--batch-size 16 --updates 10000 --min-updates-before-stop 10000 --patience 8 --lr 1e-4 --weight-decay 1e-6`, no auxiliary/scheduling flags, and the respective output directory. The text-only reference supplies split/features for both; its model weights load only for the unfrozen-projection variant.

Regenerate summaries/curves with `scripts/summarize_long_gine_training.py --directory DIRECTORY --variant VARIANT` under `nix develop -c uv run python`.
