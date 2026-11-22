# GINE stability across matched branch seeds

Fixed text-only checkpoint and date split; up to 3,000 updates per branch. Only branch initialization and batch/state order vary across seeds. Each edge/control pair shares its seed. The text backbone stays frozen.

Variants: `gated-gine-global-control` (control), `gated-gine-global` (edges). Each run selects its best checkpoint by validation s4 Hit@1, then s4 MRR. All levels below use that same checkpoint. Update 0 is eligible. No test split or search-harness games are evaluated.

## Full-board results by seed

| Seed | Control update | Edge update | Control Hit@1 | Edge Hit@1 | Delta (pp) | Control MRR | Edge MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1400 | 2200 | 16.53% | 21.49% | +4.96 | 0.2345 | 0.2970 |
| 1 | 200 | 1500 | 15.70% | 16.53% | +0.83 | 0.2269 | 0.2453 |
| 2 | 1700 | 2300 | 17.36% | 18.18% | +0.83 | 0.2377 | 0.2560 |
| 3 | 800 | 2100 | 15.70% | 20.66% | +4.96 | 0.2174 | 0.2880 |

## Ranking across all levels

Mean ± sample standard deviation across seeds. Delta is the paired edge-minus-control mean; positive-seed counts exclude ties. Percentage deltas are percentage points.

| Level | Metric | Zero-edge control | With edges | Paired delta | Positive seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | hit1 | 16.32 ± 0.79% | 19.21 ± 2.28% | +2.8926 | 4/4 |
| s4 | mrr | 0.2291 ± 0.0090 | 0.2716 ± 0.0248 | +0.0424 | 4/4 |
| s4 | hit4 | 27.07 ± 2.17% | 31.82 ± 2.90% | +4.7521 | 3/4 |
| s3 | hit1 | 18.29 ± 0.62% | 19.78 ± 1.45% | +1.4979 | 3/4 |
| s3 | mrr | 0.2581 ± 0.0051 | 0.2948 ± 0.0159 | +0.0367 | 4/4 |
| s3 | hit4 | 30.06 ± 1.13% | 35.85 ± 2.36% | +5.7851 | 4/4 |
| s2 | hit1 | 22.42 ± 0.48% | 25.69 ± 1.75% | +3.2713 | 4/4 |
| s2 | mrr | 0.3413 ± 0.0037 | 0.3886 ± 0.0153 | +0.0473 | 4/4 |
| s2 | hit4 | 40.32 ± 0.13% | 48.07 ± 2.74% | +7.7479 | 4/4 |

## Per-color group ranking

These measure each color's specific true group, rather than any correct group.

| Level | Color | Control top1 | Edge top1 | Control top4 | Edge top4 |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | yellow | 8.68 ± 0.48% | 10.12 ± 1.56% | 13.22 ± 1.51% | 16.74 ± 2.17% |
| s4 | green | 3.93 ± 0.79% | 4.96 ± 1.79% | 7.44 ± 1.35% | 9.50 ± 1.72% |
| s4 | blue | 2.89 ± 0.48% | 3.31 ± 0.67% | 5.79 ± 0.00% | 4.96 ± 0.00% |
| s4 | purple | 0.83 ± 0.00% | 0.83 ± 0.00% | 0.83 ± 0.00% | 1.03 ± 0.41% |
| s3 | yellow | 12.67 ± 0.39% | 14.19 ± 1.65% | 18.94 ± 0.94% | 22.87 ± 2.41% |
| s3 | green | 6.54 ± 0.61% | 7.16 ± 1.03% | 12.81 ± 0.83% | 14.88 ± 0.78% |
| s3 | blue | 4.34 ± 0.14% | 4.13 ± 0.50% | 7.78 ± 0.61% | 8.47 ± 0.72% |
| s3 | purple | 0.83 ± 0.00% | 0.90 ± 0.14% | 1.17 ± 0.14% | 2.55 ± 0.82% |
| s2 | yellow | 19.21 ± 0.94% | 22.25 ± 1.70% | 30.99 ± 0.83% | 37.12 ± 3.87% |
| s2 | green | 13.98 ± 0.88% | 15.50 ± 1.20% | 27.27 ± 0.78% | 31.27 ± 0.86% |
| s2 | blue | 9.23 ± 0.66% | 9.99 ± 0.99% | 18.39 ± 1.16% | 20.73 ± 1.13% |
| s2 | purple | 2.41 ± 0.52% | 3.65 ± 0.47% | 7.30 ± 0.36% | 11.36 ± 0.99% |

## Scope and limitations

This measures branch/optimization stability conditional on one pretrained backbone and one validation split. It does not measure uncertainty across puzzle samples or backbone retraining. The same validation puzzles are reused across seeds; treat them as paired repeats, not independent new puzzle observations. s3/s2 are oracle residual states formed by removing true groups, not visited search trajectories. Results are conditional on the 3,000-update budget and the recorded stopping policy.

## Run artifacts

Reference: `/mnt/code/python_projects/nyt-connections-solver/weights/experiments/20261023T150230Z-text-only.pt`.

Reproduce a run with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant VARIANT --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --updates 3000 --min-updates-before-stop 1500 --patience 8 --seed SEED`. See each JSON configuration for exact stopping settings.

- Seed 0, `gated-gine-global-control`: `experiments/graph-features/20261023T164501Z-gated-gine-global-control.json`
- Seed 0, `gated-gine-global`: `experiments/graph-features/20261023T164605Z-gated-gine-global.json`
- Seed 1, `gated-gine-global-control`: `experiments/graph-features/20261023T164824Z-gated-gine-global-control.json`
- Seed 1, `gated-gine-global`: `experiments/graph-features/20261023T164901Z-gated-gine-global.json`
- Seed 2, `gated-gine-global-control`: `experiments/graph-features/20261023T165127Z-gated-gine-global-control.json`
- Seed 2, `gated-gine-global`: `experiments/graph-features/20261023T165221Z-gated-gine-global.json`
- Seed 3, `gated-gine-global-control`: `experiments/graph-features/20261023T165432Z-gated-gine-global-control.json`
- Seed 3, `gated-gine-global`: `experiments/graph-features/20261023T165525Z-gated-gine-global.json`
