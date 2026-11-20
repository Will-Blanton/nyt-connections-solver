# GINE stability across matched branch seeds

Fixed text-only checkpoint and date split; 1,500 updates per branch. Only branch initialization and batch/state order vary across seeds. Each edge/control pair shares its seed. The text backbone stays frozen.

Each run selects its best checkpoint by validation s4 Hit@1, then s4 MRR. All levels below use that same checkpoint. Update 0 is eligible. No test split or search-harness games are evaluated.

## Full-board results by seed

| Seed | Control update | Edge update | Control Hit@1 | Edge Hit@1 | Delta (pp) | Control MRR | Edge MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1500 | 1500 | 15.70% | 21.49% | +5.79 | 0.2252 | 0.2774 |
| 1 | 100 | 1400 | 14.88% | 16.53% | +1.65 | 0.2221 | 0.2453 |
| 2 | 1300 | 1500 | 15.70% | 19.01% | +3.31 | 0.2280 | 0.2516 |
| 3 | 100 | 1500 | 16.53% | 17.36% | +0.83 | 0.2298 | 0.2522 |

## Ranking across all levels

Mean ± sample standard deviation across seeds. Delta is the paired edge-minus-control mean; positive-seed counts exclude ties. Percentage deltas are percentage points.

| Level | Metric | Zero-edge control | With edges | Paired delta | Positive seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | hit1 | 15.70 ± 0.67% | 18.60 ± 2.19% | +2.8926 | 4/4 |
| s4 | mrr | 0.2263 ± 0.0033 | 0.2566 ± 0.0142 | +0.0303 | 4/4 |
| s4 | hit4 | 27.69 ± 1.58% | 29.55 ± 1.56% | +1.8595 | 4/4 |
| s3 | hit1 | 17.87 ± 0.43% | 19.58 ± 1.50% | +1.7045 | 4/4 |
| s3 | mrr | 0.2563 ± 0.0030 | 0.2853 ± 0.0126 | +0.0290 | 4/4 |
| s3 | hit4 | 29.60 ± 0.35% | 33.78 ± 0.93% | +4.1839 | 4/4 |
| s2 | hit1 | 22.42 ± 0.17% | 24.28 ± 0.40% | +1.8595 | 4/4 |
| s2 | mrr | 0.3397 ± 0.0019 | 0.3711 ± 0.0073 | +0.0315 | 4/4 |
| s2 | hit4 | 40.36 ± 0.58% | 45.21 ± 1.50% | +4.8554 | 4/4 |

## Per-color group ranking

These measure each color's specific true group, rather than any correct group.

| Level | Color | Control top1 | Edge top1 | Control top4 | Edge top4 |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | yellow | 8.06 ± 0.79% | 9.30 ± 0.79% | 13.84 ± 1.56% | 15.29 ± 0.83% |
| s4 | green | 4.34 ± 0.41% | 4.55 ± 1.72% | 7.64 ± 0.79% | 8.26 ± 0.67% |
| s4 | blue | 2.48 ± 0.00% | 3.93 ± 0.79% | 5.37 ± 0.48% | 5.17 ± 0.41% |
| s4 | purple | 0.83 ± 0.00% | 0.83 ± 0.00% | 0.83 ± 0.00% | 1.03 ± 0.41% |
| s3 | yellow | 12.12 ± 0.45% | 13.09 ± 0.91% | 18.80 ± 0.57% | 20.87 ± 0.94% |
| s3 | green | 6.61 ± 0.95% | 7.37 ± 1.13% | 12.40 ± 0.60% | 14.46 ± 0.48% |
| s3 | blue | 4.27 ± 0.16% | 4.55 ± 0.73% | 7.37 ± 0.26% | 8.33 ± 0.47% |
| s3 | purple | 0.83 ± 0.00% | 1.10 ± 0.22% | 1.24 ± 0.16% | 2.41 ± 0.65% |
| s2 | yellow | 19.15 ± 0.83% | 19.90 ± 0.61% | 30.37 ± 1.06% | 33.06 ± 1.25% |
| s2 | green | 14.05 ± 0.98% | 15.91 ± 0.72% | 28.03 ± 1.38% | 30.17 ± 1.88% |
| s2 | blue | 8.88 ± 0.57% | 9.44 ± 0.47% | 17.63 ± 0.81% | 20.52 ± 1.14% |
| s2 | purple | 2.75 ± 0.22% | 3.31 ± 0.22% | 7.37 ± 0.85% | 9.92 ± 0.39% |

## Scope and limitations

This measures branch/optimization stability conditional on one pretrained backbone and one validation split. It does not measure uncertainty across puzzle samples or backbone retraining. The same validation puzzles are reused across seeds; treat them as paired repeats, not independent new puzzle observations. s3/s2 are oracle residual states formed by removing true groups, not visited search trajectories. Many runs select their budget endpoint, so these results are conditional on 1,500 updates.

## Run artifacts

Reference: `/mnt/code/python_projects/nyt-connections-solver/weights/experiments/20261023T150230Z-text-only.pt`.

Reproduce a run with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant VARIANT --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --min-updates-before-stop 1500 --seed SEED`.

- Seed 0, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T154802Z-gated-gine-zero-edge-control.json`
- Seed 0, `gated-gine-residual`: `experiments/graph-features/20261023T154356Z-gated-gine-residual.json`
- Seed 1, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T155510Z-gated-gine-zero-edge-control.json`
- Seed 1, `gated-gine-residual`: `experiments/graph-features/20261023T155618Z-gated-gine-residual.json`
- Seed 2, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T155704Z-gated-gine-zero-edge-control.json`
- Seed 2, `gated-gine-residual`: `experiments/graph-features/20261023T155746Z-gated-gine-residual.json`
- Seed 3, `gated-gine-zero-edge-control`: `experiments/graph-features/20261023T155830Z-gated-gine-zero-edge-control.json`
- Seed 3, `gated-gine-residual`: `experiments/graph-features/20261023T155911Z-gated-gine-residual.json`
