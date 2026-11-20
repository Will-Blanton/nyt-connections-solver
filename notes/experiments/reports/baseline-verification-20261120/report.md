# Production baseline full-run verification (2026-11-20)

Status: complete. Ran the supported training entry point at its full selected budget, then separately evaluated its saved checkpoint on the original 121 test dates. The user explicitly requested this pipeline verification. No hyperparameters, checkpoint updates, or seeds were selected using these results.

## Commands

```bash
nix develop -c uv run python scripts/train.py --output-dir weights/baseline-verification-20261120
nix develop -c uv run python scripts/evaluate_graph_feedback.py \
  weights/baseline-verification-20261120/model.pt \
  --split test --beam-width 20 --score-mode full_board \
  --max-mistakes 4 --seed 0 \
  --output experiments/baseline-verification-20261120/feedback-test.json
```

## Training and checkpoint

The ordinary path loaded the pinned Qwen model; `--cache-only` was not used. Existing feature caches were reused. Trained on 1,089 puzzles, batch 16, 1,878 updates, with exactly 626 updates each on 8/12/16-word states. Only original Qwen vectors were frozen. All active learned layers were fresh and trainable. Constant AdamW and the selected mass-plus-floor objective were unchanged; no curriculum, scheduler, early stopping, test selection, or archived-script dependency.

Training update time: 11.38s, excluding feature preparation and model loading. Final logging-window training loss: 2.6415.

Checkpoint: `weights/baseline-verification-20261120/model.pt`.

SHA-256: `7603ded0fdee2ea946d8ad59909ec7651dffba2c3ec35cd46fde627b4fd0f705`.

The production loader reconstructed this checkpoint for harness evaluation and an independent ranking audit. Exact model configuration, feature metadata, ordered edge channels, and fitted/held-out splits match the earlier selected CV seed-10 checkpoint. Learned weights are not bitwise identical to the November 7 run; this verifies the recipe and production path, not exact historical numerical reproduction. The cause of the numerical difference has not been isolated. The prior selected checkpoint remains unchanged.

## Held-out results

| Oracle state size | States | Hit@1 | MRR | Hit@4 |
|---|---:|---:|---:|---:|
| s4 | 121 | 37.19% | 0.4775 | 56.20% |
| s3 | 484 | 40.50% | 0.5168 | 61.98% |
| s2 | 726 | 46.97% | 0.5851 | 68.32% |

Ranking was audited using the production-loaded model, feature providers, canonical datasets and quartet indices. Correct candidates are the official groups; Hit@1 accepts any correct group. s3/s2 use oracle remaining groups, not feedback trajectories. Raw ranks are saved in the aggregate ranking JSON only as metrics; full harness guesses are retained separately.

- Harness solves: **26/121 (21.49%)**.
- First guess correct: **44/121 (36.36%)**.
- Mean accepted groups: 1.545.
- Mean mistakes: 3.496.
- Warm-cache harness game time: 0.313s mean; 37.85s total. Excludes model loading and the separate ranking pass; not a cold-start benchmark.

| Color | Directly selected | Forced final group |
|---|---:|---:|
| purple | 18 | 8 |
| blue | 26 | 11 |
| green | 52 | 3 |
| yellow | 65 | 4 |

All 121 games finished without invalid/repeated submissions or pipeline exceptions. No submitted group repeated in the saved histories. The original test split has been inspected previously, so this is a verification dataset, not a pristine prospective holdout. The older selected seed-10 model solved 22/121; the difference in this one rerun is not grounds for further test-based selection or a reliable performance-gain claim.

## Artifacts

- `weights/baseline-verification-20261120/`: checkpoint, effective recipe, training history, completion status and checksum.
- `experiments/baseline-verification-20261120/feedback-test.json`: all per-game feedback histories, rankings and partition diagnostics.
- `experiments/baseline-verification-20261120/ranking-test.json`: s2/s3/s4 metrics, recipe/feature/split match and checkpoint hash.

Raw artifacts remain ignored. This report is mirrored to the tracked `notes/experiments/reports/` record by the exporter.
