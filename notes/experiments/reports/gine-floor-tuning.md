# GINE probability-floor hyperparameter tuning

**Outcome:** standalone Hit@1 improved, but the six-seed harness result was unchanged (180/726 current versus 179/726 tuned). Keep the current floor-loss baseline.

Validation-only bounded Optuna study followed by matched fresh-seed retrains. Notebooks and test-set evaluation untouched.

## Protocol

Sixteen trials at search seed 0; up to 15,000 updates each, batch 16, evaluate every 250 updates. First six trials finish the full budget. Subsequent trials can be pruned only from update 8,000, at 1,000-update intervals, below the 25th percentile of best-so-far s4 Hit@1 across at least four comparable completed trials. This proxy pruning can discard a model that would have played better; it is deliberately conservative. There is no harness run at each epoch/evaluation.

Checkpoint selection within a run: validation s4 Hit@1 with MRR tie-break. Hyperparameter selection: one validation harness replay of the selected checkpoint per completed trial, maximize solve rate; ties retain the earliest Optuna trial. Harness: beam 20, full_board scores, four mistakes, public-feedback pruning, shuffle 0.

Search GINE width 64/128/256, depth 1–3, readout width 64/128/256, dropout 0/0.05/0.1/0.2, LR 3e-5–6e-4, weight decay 1e-6–3e-3, and floor weight 0.01–0.3 (continuous ranges logarithmic). Fix beta=0.1 and the same 15 edge features. Frozen Qwen vectors; active norm, projection, GINE and readout fully trainable. Balanced mixed s2/s3/s4 states, identical seed-specific schedules, no LR schedule or extra objective. Architecture remains candidate-local GINE.

Retrain current-floor and tuned-floor from the same shared text-only reference on seeds 20,21,22, 15,000 updates each, no pruning or early stopping. Current-floor uses the previous tuned GINE plus floor weight 0.1. These are fresh initialization seeds, not fresh puzzles: all evaluations reuse the same 121 validation puzzles, which have been used repeatedly for selection. Treat tuning results as optimistic and seed reruns as a stability check, not independent generalization.

## Selected configuration

Trial 14, search-seed harness solve rate 25.62%.

```json
{
  "hidden_width": 128,
  "gine_layers": 2,
  "readout_width": 128,
  "dropout": 0.0,
  "lr": 0.000531106168624315,
  "weight_decay": 3.834300397867373e-05,
  "floor_weight": 0.012929770497078105
}
```

Trial states: {"COMPLETE": 15, "PRUNED": 2, "FAIL": 1}.

A pruning API interpretation was corrected during the study: Optuna keeps the specified percentile of top trials, so keeping 75% implements the intended bottom-quartile pruning. Trial 7 was incorrectly pruned and retried; trial 8 was interrupted to apply the fix and retried. Trial 6 would still prune under the corrected cutoff. No completed trial was discarded. Recovery metadata and both original and retry records are preserved; the invalid pruning and interrupted trial do not consume the sixteen-valid-trial budget. Boundary regression tests were strengthened and passed before resuming.

## Matched fresh-seed results

| Metric | Current floor loss | Tuned floor loss |
|---|---:|---:|
| s4 hit1 | 33.33% | 38.02% |
| s4 mrr | 0.4440 | 0.4689 |
| s4 hit4 | 52.62% | 52.07% |
| s3 hit1 | 35.95% | 39.12% |
| s3 mrr | 0.4800 | 0.4954 |
| s3 hit4 | 57.71% | 57.02% |
| s2 hit1 | 43.99% | 44.77% |
| s2 mrr | 0.5680 | 0.5721 |
| s2 hit4 | 68.23% | 68.09% |
| Harness solves | 83/363 (22.87%) | 92/363 (25.34%) |
| Harness first-guess accuracy | 36.64% | 37.74% |
| Mean harness seconds per game | 0.278 | 0.278 |
| Trainable parameters | 400,259 | 400,259 |
| yellow recovered | 177/363 | 191/363 |
| green recovered | 156/363 | 164/363 |
| blue recovered | 124/363 | 126/363 |
| purple recovered | 99/363 | 105/363 |
| Purple selected | 66/363 | 60/363 |
| Purple forced | 33/363 | 45/363 |

## Per seed

| Seed | Configuration | s4 Hit@1 | Solves | Purple recovered | Selected update |
|---|---|---:|---:|---:|---:|
| 20 | current-floor | 34.71% | 31/121 | 34/121 | 1500 |
| 20 | tuned-floor | 37.19% | 28/121 | 32/121 | 1500 |
| 21 | current-floor | 32.23% | 29/121 | 32/121 | 3500 |
| 21 | tuned-floor | 37.19% | 30/121 | 34/121 | 2250 |
| 22 | current-floor | 33.06% | 23/121 | 33/121 | 4250 |
| 22 | tuned-floor | 39.67% | 34/121 | 39/121 | 4750 |

Paired changes: 35 new solves, 26 lost solves; 39 new Purple recoveries, 33 lost Purple recoveries.

## Search trials

| Trial | Status | Updates | Selected Hit@1 | Harness solves |
|---|---|---:|---:|---:|
| 0 | COMPLETE | 15000 | 33.88% | 28 |
| 1 | COMPLETE | 15000 | 30.58% | 15 |
| 2 | COMPLETE | 15000 | 35.54% | 19 |
| 3 | COMPLETE | 15000 | 26.45% | 15 |
| 4 | COMPLETE | 15000 | 22.31% | 7 |
| 5 | COMPLETE | 15000 | 26.45% | 17 |
| 6 | PRUNED | 8000 | — | None |
| 7 | INVALID_PRUNING | 8000 | — | None |
| 8 | FAIL | 6000 | — | None |
| 9 | COMPLETE (retry 7) | 15000 | 30.58% | 23 |
| 10 | COMPLETE (retry 8) | 15000 | 37.19% | 28 |
| 11 | COMPLETE | 15000 | 32.23% | 22 |
| 12 | COMPLETE | 15000 | 30.58% | 28 |
| 13 | COMPLETE | 15000 | 37.19% | 27 |
| 14 | COMPLETE | 15000 | 38.02% | 31 |
| 15 | COMPLETE | 15000 | 33.88% | 19 |
| 16 | COMPLETE | 15000 | 32.23% | 26 |
| 17 | COMPLETE | 15000 | 33.88% | 30 |

Full trial parameters and checkpoint paths: `gine-floor-tuning/summary.json`; Optuna state: `study.sqlite3`; learning curves: `fresh-seed-curves.html`. The existing default objective and notebooks remain unchanged.

## Original-seed replication and six-seed comparison

Because the initial fresh-seed gain was concentrated in seed 22, also train the already selected configuration on original seeds 10/11/12. Reuse their completed current-floor controls. Three additional full 15,000-update runs; no new search or parameter selection. This permits a direct comparison with the earlier 26.72% result, which used those original seeds. All evaluations still reuse the same 121 validation puzzles; six seeds are not six independently held-out datasets.

| Seed | Current solves | Tuned solves | Current Hit@1 | Tuned Hit@1 |
|---|---:|---:|---:|---:|
| 10 | 33/121 | 28/121 | 38.84% | 34.71% |
| 11 | 33/121 | 31/121 | 37.19% | 34.71% |
| 12 | 31/121 | 28/121 | 33.06% | 38.02% |
| 20 | 31/121 | 28/121 | 34.71% | 37.19% |
| 21 | 29/121 | 30/121 | 32.23% | 37.19% |
| 22 | 23/121 | 34/121 | 33.06% | 39.67% |

### Original seeds 10–12

| Metric | Current floor loss | Tuned floor loss |
|---|---:|---:|
| Harness solves | 97/363 (26.72%) | 87/363 (23.97%) |
| s4 Hit@1 | 36.36% | 35.81% |
| s3 Hit@1 | 38.36% | 38.02% |
| s2 Hit@1 | 42.47% | 45.00% |
| yellow recovered | 185 | 182 |
| green recovered | 162 | 155 |
| blue recovered | 136 | 127 |
| purple recovered | 106 | 96 |
| Purple before final four | 61 | 52 |

### All six matched seeds

| Metric | Current floor loss | Tuned floor loss |
|---|---:|---:|
| Harness solves | 180/726 (24.79%) | 179/726 (24.66%) |
| s4 Hit@1 | 34.85% | 36.91% |
| s3 Hit@1 | 37.16% | 38.57% |
| s2 Hit@1 | 43.23% | 44.88% |
| yellow recovered | 362 | 373 |
| green recovered | 318 | 319 |
| blue recovered | 260 | 253 |
| purple recovered | 205 | 201 |
| Purple before final four | 127 | 112 |

Across all six seeds: 50 new solves and 51 lost solves; 56 new Purple recoveries and 60 lost Purple recoveries.

Original-seed checkpoints and histories are listed in `gine-floor-replication/runs.json`. Combined metrics and per-seed records are in `gine-floor-tuning/summary.json` under `replication`.

## Conclusion

No convincing harness improvement from this bounded tuning stage. The initial fresh-seed comparison was positive (83 to 92 solves), but the original-seed replication was negative (97 to 87), leaving 180 versus 179 solves across six matched seeds. Four of six seeds lose solves; the positive net result on fresh seeds was largely driven by seed 22. Retain the current floor-weight-0.1 setup as the baseline.

The selected alternative improves standalone ranking: s4 Hit@1 rises 34.85% to 36.91%, s3 37.16% to 38.57%, and s2 43.23% to 44.88%. However, s4 Hit@4 falls 52.89% to 51.79%, and harness first-guess accuracy falls 38.15% to 37.47%. The harness selects from complete partitions, so stronger top standalone choices are not sufficient.

Purple does not benefit: total recovery falls 205 to 201, and Purple selected before the forced final four falls 127 to 112. The selected architecture is unchanged (width 128, two layers, readout 128, dropout zero, 400,259 trainable parameters). Changes are LR 5.3111e-4, weight decay 3.8343e-5 and floor weight 0.01293, with beta still 0.1. Fresh-seed mean per-game harness time is 0.278 seconds for both configurations; this includes search with prepared resources, not model-loading or unseen-word embedding startup.

The search selected configurations by harness performance on one initialization seed, which did not give a stable gain across seeds. Multiple-seed selection or more validation puzzles could reduce that selection noise, but neither was tested here. All these evaluations reuse validation puzzles and do not establish unseen-puzzle performance. No test-set evaluation or notebook changes. Nine targeted tuning/loss/GINE regression tests passed after correcting the pruning boundary. All nine new final/replication retrains completed 15,000 updates.
