# GINE stability across matched branch seeds

Fixed text-only checkpoint and date split; up to 3,000 updates per branch. Only branch initialization and batch/state order vary across seeds. Each edge/control pair shares its seed. The text backbone stays frozen.

Variants: `gated-gine-zero-edge-control` (control), `gated-gine-residual` (edges). Each run selects its best checkpoint by validation s4 Hit@1, then s4 MRR. All levels below use that same checkpoint. Update 0 is eligible. No test split or search-harness games are evaluated.

## Full-board results by seed

| Seed | Control update | Edge update | Control Hit@1 | Edge Hit@1 | Delta (pp) | Control MRR | Edge MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2200 | 2200 | 16.53% | 23.97% | +7.44 | 0.2339 | 0.3107 |
| 1 | 100 | 2000 | 14.88% | 19.83% | +4.96 | 0.2221 | 0.2765 |
| 2 | 1300 | 1500 | 15.70% | 19.01% | +3.31 | 0.2280 | 0.2516 |
| 3 | 100 | 2800 | 16.53% | 21.49% | +4.96 | 0.2298 | 0.3045 |

## Ranking across all levels

Mean ± sample standard deviation across seeds. Delta is the paired edge-minus-control mean; positive-seed counts exclude ties. Percentage deltas are percentage points.

| Level | Metric | Zero-edge control | With edges | Paired delta | Positive seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | hit1 | 15.91 ± 0.79% | 21.07 ± 2.19% | +5.1653 | 4/4 |
| s4 | mrr | 0.2285 ± 0.0049 | 0.2858 ± 0.0273 | +0.0573 | 4/4 |
| s4 | hit4 | 27.89 ± 1.83% | 33.68 ± 4.50% | +5.7851 | 4/4 |
| s3 | hit1 | 17.92 ± 0.46% | 21.44 ± 1.82% | +3.5124 | 4/4 |
| s3 | mrr | 0.2569 ± 0.0031 | 0.3099 ± 0.0238 | +0.0530 | 4/4 |
| s3 | hit4 | 29.65 ± 0.36% | 36.98 ± 2.97% | +7.3347 | 4/4 |
| s2 | hit1 | 22.38 ± 0.14% | 26.58 ± 2.25% | +4.2011 | 4/4 |
| s2 | mrr | 0.3394 ± 0.0017 | 0.3956 ± 0.0252 | +0.0561 | 4/4 |
| s2 | hit4 | 40.39 ± 0.62% | 47.97 ± 3.48% | +7.5758 | 4/4 |

## Per-color group ranking

These measure each color's specific true group, rather than any correct group.

| Level | Color | Control top1 | Edge top1 | Control top4 | Edge top4 |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | yellow | 8.26 ± 0.67% | 10.74 ± 2.02% | 13.84 ± 1.56% | 17.56 ± 2.56% |
| s4 | green | 4.55 ± 0.83% | 5.58 ± 1.41% | 7.64 ± 0.79% | 9.71 ± 2.07% |
| s4 | blue | 2.27 ± 0.41% | 4.13 ± 0.67% | 5.58 ± 0.79% | 5.17 ± 0.41% |
| s4 | purple | 0.83 ± 0.00% | 0.62 ± 0.41% | 0.83 ± 0.00% | 1.24 ± 1.07% |
| s3 | yellow | 12.12 ± 0.45% | 14.05 ± 1.81% | 18.73 ± 0.64% | 22.87 ± 2.59% |
| s3 | green | 6.82 ± 1.24% | 8.40 ± 2.16% | 12.33 ± 0.65% | 16.53 ± 2.30% |
| s3 | blue | 4.13 ± 0.39% | 5.10 ± 0.73% | 7.58 ± 0.66% | 8.33 ± 0.57% |
| s3 | purple | 0.83 ± 0.00% | 1.03 ± 0.26% | 1.24 ± 0.16% | 2.62 ± 0.66% |
| s2 | yellow | 19.28 ± 0.64% | 22.18 ± 2.27% | 30.79 ± 0.72% | 36.71 ± 3.73% |
| s2 | green | 14.19 ± 1.16% | 17.70 ± 2.56% | 27.75 ± 1.06% | 32.85 ± 3.62% |
| s2 | blue | 8.61 ± 0.26% | 9.78 ± 0.57% | 17.29 ± 0.61% | 20.45 ± 1.11% |
| s2 | purple | 2.69 ± 0.35% | 3.51 ± 0.47% | 7.44 ± 0.87% | 10.26 ± 0.57% |

## Scope and limitations

This measures branch/optimization stability conditional on one pretrained backbone and one validation split. It does not measure uncertainty across puzzle samples or backbone retraining. The same validation puzzles are reused across seeds; treat them as paired repeats, not independent new puzzle observations. s3/s2 are oracle residual states formed by removing true groups, not visited search trajectories. Results are conditional on the 3,000-update budget and the recorded stopping policy.

## Run artifacts

Reference: `/mnt/code/python_projects/nyt-connections-solver/weights/experiments/20261023T150230Z-text-only.pt`.

Reproduce a run with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant VARIANT --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --updates 3000 --min-updates-before-stop 1500 --patience 8 --seed SEED`. See each JSON configuration for exact stopping settings.

- Seed 0, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T161915Z-gated-gine-zero-edge-control.json`
- Seed 0, `gated-gine-residual`: `experiments/graph-features/20261023T162019Z-gated-gine-residual.json`
- Seed 1, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T162058Z-gated-gine-zero-edge-control.json`
- Seed 1, `gated-gine-residual`: `experiments/graph-features/20261023T162200Z-gated-gine-residual.json`
- Seed 2, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T162239Z-gated-gine-zero-edge-control.json`
- Seed 2, `gated-gine-residual`: `experiments/graph-features/20261023T162333Z-gated-gine-residual.json`
- Seed 3, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T162412Z-gated-gine-zero-edge-control.json`
- Seed 3, `gated-gine-residual`: `experiments/graph-features/20261023T162517Z-gated-gine-residual.json`
