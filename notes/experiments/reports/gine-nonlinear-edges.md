# GINE stability across matched branch seeds

Fixed text-only checkpoint and date split; up to 3,000 updates per branch. Only branch initialization and batch/state order vary across seeds. Each edge/control pair shares its seed. The text backbone stays frozen.

Variants: `gated-gine-mlp-edge-control` (control), `gated-gine-mlp-edges` (edges). Each run selects its best checkpoint by validation s4 Hit@1, then s4 MRR. All levels below use that same checkpoint. Update 0 is eligible. No test split or search-harness games are evaluated.

## Full-board results by seed

| Seed | Control update | Edge update | Control Hit@1 | Edge Hit@1 | Delta (pp) | Control MRR | Edge MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2200 | 1400 | 16.53% | 21.49% | +4.96 | 0.2339 | 0.2778 |
| 1 | 100 | 2100 | 14.88% | 24.79% | +9.92 | 0.2221 | 0.3143 |
| 2 | 1300 | 1400 | 15.70% | 18.18% | +2.48 | 0.2280 | 0.2507 |
| 3 | 100 | 1100 | 16.53% | 15.70% | -0.83 | 0.2298 | 0.2335 |

## Ranking across all levels

Mean ± sample standard deviation across seeds. Delta is the paired edge-minus-control mean; positive-seed counts exclude ties. Percentage deltas are percentage points.

| Level | Metric | Zero-edge control | With edges | Paired delta | Positive seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | hit1 | 15.91 ± 0.79% | 20.04 ± 3.96% | +4.1322 | 3/4 |
| s4 | mrr | 0.2285 ± 0.0049 | 0.2691 ± 0.0352 | +0.0406 | 4/4 |
| s4 | hit4 | 27.89 ± 1.83% | 29.75 ± 3.02% | +1.8595 | 3/4 |
| s3 | hit1 | 17.92 ± 0.46% | 20.56 ± 2.89% | +2.6343 | 4/4 |
| s3 | mrr | 0.2569 ± 0.0031 | 0.2949 ± 0.0282 | +0.0380 | 4/4 |
| s3 | hit4 | 29.65 ± 0.36% | 34.81 ± 2.78% | +5.1653 | 4/4 |
| s2 | hit1 | 22.38 ± 0.14% | 26.03 ± 2.78% | +3.6501 | 4/4 |
| s2 | mrr | 0.3394 ± 0.0017 | 0.3868 ± 0.0257 | +0.0474 | 4/4 |
| s2 | hit4 | 40.39 ± 0.62% | 47.11 ± 2.76% | +6.7149 | 4/4 |

## Per-color group ranking

These measure each color's specific true group, rather than any correct group.

| Level | Color | Control top1 | Edge top1 | Control top4 | Edge top4 |
| --- | --- | ---: | ---: | ---: | ---: |
| s4 | yellow | 8.26 ± 0.67% | 10.12 ± 2.37% | 13.84 ± 1.56% | 15.08 ± 1.41% |
| s4 | green | 4.55 ± 0.83% | 4.55 ± 1.07% | 7.64 ± 0.79% | 8.06 ± 1.70% |
| s4 | blue | 2.27 ± 0.41% | 4.34 ± 0.79% | 5.58 ± 0.79% | 4.96 ± 0.00% |
| s4 | purple | 0.83 ± 0.00% | 1.03 ± 0.41% | 0.83 ± 0.00% | 1.65 ± 0.67% |
| s3 | yellow | 12.12 ± 0.45% | 13.98 ± 1.87% | 18.73 ± 0.64% | 21.97 ± 0.91% |
| s3 | green | 6.82 ± 1.24% | 6.96 ± 1.32% | 12.33 ± 0.65% | 14.46 ± 2.14% |
| s3 | blue | 4.13 ± 0.39% | 5.17 ± 0.76% | 7.58 ± 0.66% | 8.13 ± 0.48% |
| s3 | purple | 0.83 ± 0.00% | 1.31 ± 0.26% | 1.24 ± 0.16% | 2.75 ± 0.93% |
| s2 | yellow | 19.28 ± 0.64% | 21.69 ± 1.94% | 30.79 ± 0.72% | 35.88 ± 3.17% |
| s2 | green | 14.19 ± 1.16% | 16.05 ± 2.51% | 27.75 ± 1.06% | 31.27 ± 3.66% |
| s2 | blue | 8.61 ± 0.26% | 10.40 ± 1.20% | 17.29 ± 0.61% | 21.01 ± 1.04% |
| s2 | purple | 2.69 ± 0.35% | 3.93 ± 0.69% | 7.44 ± 0.87% | 10.40 ± 0.88% |

## Scope and limitations

This measures branch/optimization stability conditional on one pretrained backbone and one validation split. It does not measure uncertainty across puzzle samples or backbone retraining. The same validation puzzles are reused across seeds; treat them as paired repeats, not independent new puzzle observations. s3/s2 are oracle residual states formed by removing true groups, not visited search trajectories. Results are conditional on the 3,000-update budget and the recorded stopping policy.

## Run artifacts

Reference: `/mnt/code/python_projects/nyt-connections-solver/weights/experiments/20261023T150230Z-text-only.pt`.

Reproduce a run with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant VARIANT --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --updates 3000 --min-updates-before-stop 1500 --patience 8 --seed SEED`. See each JSON configuration for exact stopping settings.

- Seed 0, `gated-gine-mlp-edge-control`: `experiments/graph-features/20261023T162747Z-gated-gine-mlp-edge-control.json`
- Seed 0, `gated-gine-mlp-edges`: `experiments/graph-features/20261023T162840Z-gated-gine-mlp-edges.json`
- Seed 1, `gated-gine-mlp-edge-control`: `experiments/graph-features/20261023T162920Z-gated-gine-mlp-edge-control.json`
- Seed 1, `gated-gine-mlp-edges`: `experiments/graph-features/20261023T163026Z-gated-gine-mlp-edges.json`
- Seed 2, `gated-gine-mlp-edge-control`: `experiments/graph-features/20261023T163106Z-gated-gine-mlp-edge-control.json`
- Seed 2, `gated-gine-mlp-edges`: `experiments/graph-features/20261023T163154Z-gated-gine-mlp-edges.json`
- Seed 3, `gated-gine-mlp-edge-control`: `experiments/graph-features/20261023T163234Z-gated-gine-mlp-edge-control.json`
- Seed 3, `gated-gine-mlp-edges`: `experiments/graph-features/20261023T163313Z-gated-gine-mlp-edges.json`
