# Five-fold tuning: ordinary GINE versus edge coherence

Status: **complete**. This note is regenerated after each trial and at completion.

## Protocol

1089 puzzles from the original training + validation pool. Five deterministic puzzle-date folds; all derived states stay in their puzzle fold. Historical test puzzles are excluded. Each puzzle is held out exactly once per configuration.

Equal budgets of 16 configurations per approach. Every completed fold receives 30,000 updates, batch 16, balanced s2/s3/s4 exposure, evaluation every 500 updates. No within-fold early stopping, pruning, LR scheduler, or extra losses. Fixed probability-mass plus floor loss (.1 weight, .1 beta). Six complete startup configurations per arm; median pruning only after at least three fully trained folds of subsequent configurations, compared at the same fold count.

All active learned parameters, including text projection and LayerNorm, initialize fresh for each fold; no learned checkpoint weights are loaded. Only the original Qwen embeddings are frozen. Models use the same original 15 edge channels and candidate-local GINE. The reference checkpoint supplies feature definitions, architecture metadata, and original split dates only. Folds use matched model seeds, schedules and per-size minibatch orders across approaches.

## Search space

| Parameter | Both approaches |
|---|---|
| GINE width | 32, 64, 128 |
| GINE depth | 1–3 |
| Text projection width | 128, 256 |
| Readout width | 64, 128, 256 |
| Dropout | 0, .1, .2, .3, .4 |
| Learning rate | 3e-5–1e-3, logarithmic |
| AdamW weight decay | 1e-6–1e-2, logarithmic |
| Coherence strength | Baseline: 0; amplified: .1–3, logarithmic |

Amplification is `edge_ij,k × (1 + strength × coherence_k)`, with coherence the weakest word’s average off-diagonal edge value to the other three words. No channels or edges are removed, and no new trainable parameters are added.

## Search progress

| Approach | Complete configurations | Pruned | Running | Best mean fold Hit@1 | Fold SD |
|---|---:|---:|---:|---:|---:|
| baseline | 12 | 4 | 0 | 40.04% | 2.47% |
| coherence | 9 | 7 | 0 | 39.48% | 2.31% |

### Current best: baseline

Trial 13; fold Hit@1: 36.70%, 43.12%, 40.83%, 38.53%, 41.01%.

```json
{
  "hidden_width": 64,
  "gine_layers": 3,
  "readout_width": 64,
  "projection_width": 256,
  "dropout": 0.0,
  "lr": 0.0009802617130802425,
  "weight_decay": 0.005032124249909019
}
```

### Current best: coherence

Trial 14; fold Hit@1: 37.16%, 41.74%, 41.28%, 40.37%, 36.87%.

```json
{
  "hidden_width": 128,
  "gine_layers": 3,
  "readout_width": 128,
  "projection_width": 256,
  "dropout": 0.0,
  "lr": 0.0009067614410863004,
  "weight_decay": 8.945694870525572e-05,
  "edge_coherence_scale": 0.17543030883511382
}
```

## Selected configurations: out-of-fold harness

Each puzzle below is scored by a model that did not train on that puzzle. Hyperparameters and checkpoint updates were selected using these same folds; these are selection-affected CV results, not an independent final test.

| Metric | Ordinary GINE | Coherence |
|---|---:|---:|
| s4 hit1 | 0.4004 | 0.3949 |
| s4 mrr | 0.5018 | 0.4972 |
| s4 hit4 | 0.5684 | 0.5813 |
| s3 hit1 | 0.4199 | 0.4125 |
| s3 mrr | 0.5280 | 0.5217 |
| s3 hit4 | 0.6104 | 0.6088 |
| s2 hit1 | 0.4715 | 0.4694 |
| s2 mrr | 0.5900 | 0.5866 |
| s2 hit4 | 0.6939 | 0.6837 |
| Harness solves | 257/1089 (23.60%) | 259/1089 (23.78%) |
| Harness first_correct | 438/1089 (40.22%) | 434/1089 (39.85%) |
| yellow recovered | 609 | 592 |
| green recovered | 500 | 508 |
| blue recovered | 373 | 370 |
| purple recovered | 289 | 290 |
| Purple selected | 182 | 181 |
| Purple forced | 107 | 109 |

### Fold results

| Fold | Approach | Puzzles | Hit@1 | Solves | Purple direct / forced | Selected update |
|---|---|---:|---:|---:|---:|---:|
| 0 | baseline | 218 | 36.70% | 55 | 44 / 22 | 1500 |
| 1 | baseline | 218 | 43.12% | 48 | 32 / 19 | 2500 |
| 2 | baseline | 218 | 40.83% | 50 | 34 / 21 | 1000 |
| 3 | baseline | 218 | 38.53% | 54 | 38 / 22 | 2500 |
| 4 | baseline | 217 | 41.01% | 50 | 34 / 23 | 1500 |
| 0 | coherence | 218 | 37.16% | 56 | 42 / 25 | 1500 |
| 1 | coherence | 218 | 41.74% | 46 | 29 / 20 | 4500 |
| 2 | coherence | 218 | 41.28% | 56 | 39 / 23 | 2500 |
| 3 | coherence | 218 | 40.37% | 52 | 39 / 21 | 3000 |
| 4 | coherence | 217 | 36.87% | 49 | 32 / 20 | 1000 |

Paired changes: {"solve_gain": 52, "solve_loss": 50, "purple_direct_gain": 35, "purple_direct_loss": 36}.


## Interpretation limits

Cross-validation reduces dependence on a single puzzle split, but folds overlap in their training data and the selected scores are optimistic because the folds also guide checkpoint and hyperparameter selection. This is not nested cross-validation. Do not treat five fold estimates as five independent experiments or compare these fresh-initialization scores directly with earlier warm-started single-split results. Reserve newly ingested chronological puzzles for a locked final comparison.

Harness settings are fixed: beam 20, original full-board quartet score sums, four mistakes and shuffle seed 0. No harness scores enter hyperparameter selection. Reduced-state s2/s3 metrics use oracle remaining groups. No notebooks changed.

Resume: `nix develop -c uv run python -u scripts/tune_gine_coherence_cv.py --background`. Completed folds are reused; an interrupted partial fold restarts with its original seed. Progress: `progress.json`; worker status: `status.json`; log: `run.log`. Final harness results, checkpoint audits and this report are generated automatically.
