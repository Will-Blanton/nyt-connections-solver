# Standalone GINE: trainable text projection

The Qwen text embeddings are precomputed and frozen in all runs. In the previous standalone GINE model, the text-only checkpoint's 1024→256 projection was also frozen, while GINE's 256→32 node projection, edge projections, message updates, and group readout were trained. This ablation unfreezes only the 1024→256 projection (262,400 extra trainable parameters). The input layer norm and all unused attention modules remain frozen. The projection starts from the same text-only checkpoint in every variant.

A follow-up with fresh and random-frozen projections, matched initialization, and width controls is in [gine-projection-diagnostic.md](gine-projection-diagnostic.md).

Four matched seeds (0–3), the same 968 training/121 validation puzzles, original 15 edge channels, batch/state sampling, probability-mass loss, optimizer/weight decay, update budget, early stopping, and s4 Hit@1 then MRR checkpoint selection. GINE/readout LR is 1e-4 in all runs. The trainable projection uses either that same LR or a separate 1e-5 LR. No old test puzzles were evaluated.

## Validation ranking

Means across four seeds at each run's selected checkpoint. s3/s2 are oracle reduced states, not feedback-reached states.

| Projection | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
|---|---:|---:|---:|---:|---:|
| Frozen | 28.51% | 0.3760 | 43.39% | 29.55% | 36.95% |
| Trainable, LR 1e-4 | 20.04% | 0.2602 | 27.89% | 20.71% | 26.62% |
| Trainable, LR 1e-5 | 21.90% | 0.3162 | 37.81% | 24.64% | 32.71% |

The lower projection LR improves over the same-LR variant but trails frozen in every matched seed: s4 Hit@1 by seed is 28.10→23.14%, 28.93→22.31%, 28.10→18.18%, and 28.93→23.97%. The same-LR variant is worse still. Training loss continued to decrease, so the validation decline is compatible with overfitting or destructive drift of the pretrained projection; these data do not isolate which.

## Validation feedback harness

Frozen versus trainable at projection LR 1e-5. Both use the selected checkpoint, four allowed mistakes, beam width 20, `score_mode=full_board`, and the same 121 validation puzzles. Means across seeds are repeated measurements on the same puzzles.

| Seed | Frozen solves | Trainable solves | Frozen first-guess accuracy | Trainable first-guess accuracy |
|---:|---:|---:|---:|---:|
| 0 | 22 | 13 | 27.27% | 22.31% |
| 1 | 17 | 12 | 32.23% | 18.18% |
| 2 | 22 | 16 | 29.75% | 23.97% |
| 3 | 17 | 15 | 31.40% | 23.14% |
| Mean / 121 | 19.5 (16.12%) | 14.0 (11.57%) | 30.17% | 21.90% |

Mean correct groups per game: 1.262 frozen versus 1.031 trainable. Purple groups recovered across the four repeated runs: 95 versus 76 (out of 484 opportunities); Blue: 136 versus 106. This experiment does **not** support changing the default to a trainable text projection. It does show that the earlier frozen choice should be viewed as an empirical regularizer, not a conceptual requirement. A future test could use a still smaller projection LR or stronger regularization, but should be treated as a new tuning experiment rather than assumed beneficial.

Implementation: `StandaloneCandidateGINE(train_text_projection=True)` and runner variant `standalone-gine-trainable-projection`. The projection LR is persisted in checkpoint training metadata; the model config persists the trainable flag for reload. Training logs: `experiments/graph-features/standalone-trainable-{seed}.log` and `standalone-trainable-low-lr-{seed}.log`. Training summaries/checkpoints: `experiments/graph-features/*standalone-gine-trainable-projection.json`, `weights/experiments/*standalone-gine-trainable-projection.pt`. Per-game histories: `experiments/harness-standalone-trainable-low-lr-{seed}-val.json`.

Reproduce training with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant standalone-gine-trainable-projection --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --seed SEED --updates 3000 --min-updates-before-stop 1500 --patience 8 --lr 1e-4 --projection-lr 1e-5 --weight-decay 1e-6`. Reproduce harness with `nix develop -c uv run python scripts/evaluate_graph_feedback.py CHECKPOINT --split val --beam-width 20 --score-mode full_board --output OUTPUT.json`.
