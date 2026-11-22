# Standalone candidate GINE versus frozen-attention residual GINE

Four matched seeds (0–3), validation only. Both variants use the same frozen Qwen text embeddings, the same text projection initialized from the text-only checkpoint, the original 15 pairwise edge channels, the same 968/121 train/validation date split, batch/state order seeds, AdamW LR 1e-4, weight decay 1e-6, batch size 16, up to 3000 updates, and checkpoint selection by validation s4 Hit@1 then s4 MRR. Both use two candidate-local GINE layers with 32 hidden units and mean/max node pooling. The text projection is frozen in both. No held-out test puzzles were evaluated.

The residual model keeps the text-only attention scorer frozen and trains a gated GINE correction to its score. The standalone model gathers each four-word candidate immediately after the frozen text projection, runs GINE using pairwise edges, pools the four output nodes, and trains an independent 64→128→1 group readout. Its forward path does **not** run the global attention encoder, candidate attention encoder/decoder, or frozen base scorer. The implementation retains those unused modules to load the shared projection checkpoint; they do not affect predictions. Trainable parameter counts are 14,052 for residual and 21,859 for standalone. Thus this is standalone with respect to the group scorer, while still benefiting from a projection pretrained by the earlier text-only model.

## Ranking after checkpoint selection

Mean over four seeds, same selected checkpoint evaluated at every state size.

| State | Metric | Residual GINE | Standalone GINE |
|---|---|---:|---:|
| s4 | Hit@1 | 22.93% | 28.51% |
| s4 | MRR | 0.3153 | 0.3760 |
| s4 | Hit@4 | 36.98% | 43.39% |
| s3 | Hit@1 | 23.45% | 29.55% |
| s3 | MRR | 0.3329 | 0.4100 |
| s3 | Hit@4 | 40.08% | 49.17% |
| s2 | Hit@1 | 28.65% | 36.95% |
| s2 | MRR | 0.4194 | 0.5046 |
| s2 | Hit@4 | 50.93% | 61.85% |

Standalone improved s4 Hit@1 in all four paired seeds, by +9.92, +2.48, +6.61, and +3.31 percentage points.

## Validation feedback games

Each selected checkpoint played the same 121 validation puzzles with four allowed mistakes, shuffle seed 0, beam width 20, and `score_mode=full_board` (sum of all four original-board group scores). The harness and constraints are otherwise identical. Mean solve counts across seeds are repeated measurements on the **same** 121 puzzles, not independent puzzle samples.

| Seed | Residual solves | Standalone solves | Residual first-guess accuracy | Standalone first-guess accuracy | Residual mean groups | Standalone mean groups |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8/121 | 22/121 | 19.83% | 27.27% | 0.760 | 1.240 |
| 1 | 10/121 | 17/121 | 25.62% | 32.23% | 0.909 | 1.264 |
| 2 | 12/121 | 22/121 | 19.83% | 29.75% | 0.810 | 1.298 |
| 3 | 11/121 | 17/121 | 25.62% | 31.40% | 0.868 | 1.248 |
| Mean | 10.25/121 (8.47%) | 19.5/121 (16.12%) | 22.73% | 30.17% | 0.837 | 1.262 |

Across the four runs, residual recovered 141 Yellow, 131 Green, 73 Blue, and 60 Purple groups; standalone recovered 203, 177, 136, and 95 respectively. Each color has 484 opportunities across four repeated runs. Mean game time was about 0.38 s for residual and 0.33 s for standalone in these runs, though cache warmth and system variation limit this timing comparison.

This is a strong validation result for a simpler group-scoring path, but it does not isolate whether the gain comes from removing attention, using an independent readout instead of a gated correction, or increasing trainable capacity. The text projection is pretrained, so it is not a from-scratch GINE test. The next controlled architecture ablation should separate those factors before attributing the gain to GINE alone. Keep the old test split untouched; assess generalization on newly ingested puzzles.

Follow-up: [standalone-gine-trainable-projection.md](standalone-gine-trainable-projection.md) compares learning the pretrained 1024→256 projection at two learning rates. Both settings underperformed this frozen-projection baseline on validation.

Training JSON/checkpoints: `experiments/graph-features/*standalone-gine-projected.json`, `weights/experiments/*standalone-gine-projected.pt`. Per-game histories: `experiments/harness-standalone-{seed}-{residual,standalone}-val.json`. Reproduce training with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant standalone-gine-projected --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --seed SEED --updates 3000 --min-updates-before-stop 1500 --patience 8 --lr 1e-4 --weight-decay 1e-6`. Reproduce harness with `nix develop -c uv run python scripts/evaluate_graph_feedback.py CHECKPOINT --split val --beam-width 20 --score-mode full_board --output OUTPUT.json`.
