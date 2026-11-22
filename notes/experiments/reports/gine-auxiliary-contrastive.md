# Puzzle-local contrastive auxiliary objective

Validation-only experiment with fully trainable scratch LayerNorm, 1024→256 Linear/ReLU projection, learned GINE 256→32 node projection, two message-passing layers, original 15 edge channels, and existing pooled readout. Qwen embeddings are frozen; no model weights are transferred from the text-only checkpoint. The reference supplies the split and feature metadata. The experimental class retains unused inherited attention modules; they are not part of prediction.

Training objective: probability-mass ranking loss plus `aux_weight * puzzle_contrastive_loss(projected_words)`. Contrastive supervision acts on normalized 256-dimensional pre-GINE word features, with temperature 0.1. For each anchor, average negative log probability over its three same-group positives, with all other words in that state in the denominator and self excluded. Negatives come only from other groups in that same state/puzzle; no cross-puzzle negatives or validation-derived pairs are used. State loader invariants preserve complete groups as contiguous four-word blocks, the same ordering already used for ranking relevance. The extra objective adds no model parameters or inference work. Auxiliary features are recomputed through the same deterministic input transformation during training.

Four matched seeds (0–3), same 968 training/121 validation date split, batch size 16, mixed s2/s3/s4 states, AdamW LR 1e-4, weight decay 1e-6, up to 3000 updates, evaluation every 100, minimum stopping budget 1500, patience 8, checkpoint selection by validation s4 Hit@1 then MRR. Reused the saved four-seed scratch-256 controls. First tried weight 0.05 on seeds 0 and 1; both were worse, so tested 0.01, then completed both weights on seeds 2 and 3. No old test evaluation or notebook edits.

## Ranking

Means over four selected checkpoints; s3/s2 use oracle reduced states.

| Auxiliary weight | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
|---|---:|---:|---:|---:|---:|
| 0 (control) | 22.52% | 0.2944 | 33.06% | 23.50% | 28.31% |
| 0.01 | 20.66% | 0.2901 | 34.50% | 22.42% | 28.44% |
| 0.05 | 18.18% | 0.2615 | 30.37% | 20.09% | 27.65% |

| Seed | Control s4 Hit@1 | Weight 0.01 | Weight 0.05 |
|---:|---:|---:|---:|
| 0 | 22.31% | 21.49% | 18.18% |
| 1 | 22.31% | 16.53% | 15.70% |
| 2 | 23.97% | 25.62% | 20.66% |
| 3 | 21.49% | 19.01% | 18.18% |

Per-color true-group Top4 at s4, averaged over seeds: control Yellow 16.32%, Green 9.71%, Blue 4.34%, Purple 3.10%; weight 0.01 gives 16.53%, 10.74%, 4.55%, 2.89%; weight 0.05 gives 13.02%, 9.09%, 4.96%, 3.93%. These are individual true-group retrieval rates, distinct from state Hit@4 (any correct group). No consistent semantic/color benefit emerges. The stronger loss trades away Yellow/Green retrieval despite a small Purple increase.

## Feedback harness

Weight 0.01 received harness evaluation because its Hit@4 improved slightly. Same 121 validation puzzles, four mistakes, beam width 20, full-board scoring, shuffle seed 0. Controls reuse `harness-scratch-256-{seed}-val.json`.

| Seed | Control solves | Auxiliary solves |
|---:|---:|---:|
| 0 | 14 | 12 |
| 1 | 14 | 9 |
| 2 | 8 | 13 |
| 3 | 9 | 13 |
| Mean / 121 | 11.25 (9.30%) | 11.75 (9.71%) |

First-guess accuracy declines from 24.17% to 20.66%; mean recovered groups declines from 0.913 to 0.905. The extra 0.5 solves on average is small, seed-dependent, and measured repeatedly on the same validation puzzles. It does not establish an overall benefit, especially alongside weaker ranking and first-guess performance. Keep auxiliary weight zero by default. This negative/mixed result does not rule out other temperatures, schedules, staged pretraining, or a different relationship objective, but those require separate experiments.

## Artifacts

Training summaries: `experiments/auxiliary-contrastive/*.json` (inspect `config.aux_weight` and seed). Checkpoints retain the same basename under `weights/experiments/`. Per-interval ranking loss and unweighted auxiliary loss are logged separately as `training.history[*].rank_loss` and `.aux_loss`. Harness histories: `experiments/harness-aux-001-{seed}-val.json`.

Reproduce with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant standalone-gine-scratch-projected --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --seed SEED --updates 3000 --min-updates-before-stop 1500 --patience 8 --lr 1e-4 --weight-decay 1e-6 --aux-weight WEIGHT --aux-temperature 0.1 --output-dir experiments/auxiliary-contrastive`.

Tests verify exclusion of self-positives, the uniform-similarity reference loss, lower loss for coherent true groups, independence between puzzles in a batch, and finite nonzero gradients reaching a learned input projection.
