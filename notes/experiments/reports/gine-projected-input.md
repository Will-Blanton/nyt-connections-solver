# GINE stability across matched branch seeds

Fixed text-only checkpoint and date split; up to 3,000 updates per branch. Only branch initialization and batch/state order vary across seeds. Each edge/control pair shares its seed. The text backbone stays frozen.

Variants: `gated-gine-projected-control` (control), `gated-gine-projected` (edges). Each run selects its best checkpoint by validation s4 Hit@1, then s4 MRR. All levels below use that same checkpoint. Update 0 is eligible. No test split or search-harness games are evaluated.

## Full-board results by seed

| Seed | Control update | Edge update | Control Hit@1 | Edge Hit@1 | Delta (pp) | Control MRR | Edge MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 400 | 1400 | 15.70% | 18.18% | +2.48 | 0.2200 | 0.2799 |
| 1 | 200 | 2700 | 15.70% | 26.45% | +10.74 | 0.2256 | 0.3487 |
| 2 | 200 | 1800 | 15.70% | 21.49% | +5.79 | 0.2241 | 0.2940 |
| 3 | 200 | 1600 | 16.53% | 25.62% | +9.09 | 0.2273 | 0.3387 |

## Ranking across all levels

Mean ± sample standard deviation across seeds. Delta is the paired edge-minus-control mean; positive-seed counts exclude ties. Percentage deltas are percentage points.

| Level | Metric | Zero-edge control | With edges | Paired delta | Positive seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | hit1 | 15.91 ± 0.41% | 22.93 ± 3.84% | +7.0248 | 4/4 |
| s4 | mrr | 0.2242 ± 0.0031 | 0.3153 ± 0.0335 | +0.0911 | 4/4 |
| s4 | hit4 | 27.07 ± 1.04% | 36.98 ± 3.12% | +9.9174 | 4/4 |
| s3 | hit1 | 17.72 ± 0.26% | 23.45 ± 2.33% | +5.7335 | 4/4 |
| s3 | mrr | 0.2551 ± 0.0022 | 0.3329 ± 0.0237 | +0.0778 | 4/4 |
| s3 | hit4 | 29.86 ± 0.12% | 40.08 ± 3.05% | +10.2273 | 4/4 |
| s2 | hit1 | 22.42 ± 0.13% | 28.65 ± 1.94% | +6.2328 | 4/4 |
| s2 | mrr | 0.3393 ± 0.0008 | 0.4194 ± 0.0193 | +0.0801 | 4/4 |
| s2 | hit4 | 39.77 ± 0.28% | 50.93 ± 2.66% | +11.1570 | 4/4 |

## Per-color group ranking

These measure each color's specific true group, rather than any correct group.

| Level | Color | Control top1 | Edge top1 | Control top4 | Edge top4 |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | yellow | 8.88 ± 0.41% | 11.16 ± 2.74% | 14.26 ± 1.24% | 17.36 ± 2.94% |
| s4 | green | 3.72 ± 0.83% | 6.82 ± 1.56% | 6.20 ± 0.48% | 12.81 ± 1.07% |
| s4 | blue | 2.48 ± 0.67% | 4.13 ± 0.67% | 5.79 ± 0.00% | 5.37 ± 0.48% |
| s4 | purple | 0.83 ± 0.00% | 0.83 ± 0.00% | 0.83 ± 0.00% | 1.86 ± 1.04% |
| s3 | yellow | 12.60 ± 0.35% | 15.15 ± 2.22% | 19.08 ± 0.14% | 25.07 ± 2.82% |
| s3 | green | 5.72 ± 0.14% | 9.92 ± 1.64% | 12.40 ± 0.22% | 18.80 ± 1.89% |
| s3 | blue | 4.48 ± 0.14% | 4.96 ± 0.50% | 7.37 ± 0.14% | 8.33 ± 0.41% |
| s3 | purple | 0.83 ± 0.00% | 1.24 ± 0.16% | 1.17 ± 0.14% | 2.96 ± 0.72% |
| s2 | yellow | 19.90 ± 0.35% | 24.31 ± 2.38% | 31.06 ± 0.52% | 42.01 ± 3.20% |
| s2 | green | 13.09 ± 0.16% | 18.87 ± 1.60% | 27.07 ± 0.47% | 34.02 ± 2.39% |
| s2 | blue | 8.95 ± 0.28% | 10.54 ± 0.61% | 16.80 ± 0.55% | 20.87 ± 1.85% |
| s2 | purple | 2.89 ± 0.28% | 3.58 ± 0.95% | 6.96 ± 0.14% | 10.88 ± 0.48% |

## Scope and limitations

This measures branch/optimization stability conditional on one pretrained backbone and one validation split. It does not measure uncertainty across puzzle samples or backbone retraining. The same validation puzzles are reused across seeds; treat them as paired repeats, not independent new puzzle observations. s3/s2 are oracle residual states formed by removing true groups, not visited search trajectories. Results are conditional on the 3,000-update budget and the recorded stopping policy.

## Run artifacts

Reference: `/mnt/code/python_projects/nyt-connections-solver/weights/experiments/20261023T150230Z-text-only.pt`.

Reproduce a run with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant VARIANT --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --updates 3000 --min-updates-before-stop 1500 --patience 8 --seed SEED`. See each JSON configuration for exact stopping settings.

- Seed 0, `gated-gine-projected-control`: `experiments/graph-features/20261023T164049Z-gated-gine-projected-control.json`
- Seed 0, `gated-gine-projected`: `experiments/graph-features/20261023T164424Z-gated-gine-projected.json`
- Seed 1, `gated-gine-projected-control`: `experiments/graph-features/20261023T164643Z-gated-gine-projected-control.json`
- Seed 1, `gated-gine-projected`: `experiments/graph-features/20261023T164746Z-gated-gine-projected.json`
- Seed 2, `gated-gine-projected-control`: `experiments/graph-features/20261023T164939Z-gated-gine-projected-control.json`
- Seed 2, `gated-gine-projected`: `experiments/graph-features/20261023T165032Z-gated-gine-projected.json`
- Seed 3, `gated-gine-projected-control`: `experiments/graph-features/20261023T165258Z-gated-gine-projected-control.json`
- Seed 3, `gated-gine-projected`: `experiments/graph-features/20261023T165352Z-gated-gine-projected.json`
