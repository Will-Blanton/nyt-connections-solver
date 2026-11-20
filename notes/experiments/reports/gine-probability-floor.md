# Weak probability-floor penalty on GINE

Validation only: three matched seeds (10/11/12), 121 puzzles per seed, 15,000 updates per run. Reuse the completed probability-mass controls from `gine-curriculum/mixed`. Six new runs, one at a time. Repeated seeds reuse the same puzzles; 363 appearances are not 363 independent puzzles. No notebooks or test-set evaluation.

## Setup

`loss = probability_mass_loss + weight * mean_correct relu(log(beta/m) - log p(group))`.

Fix beta=0.1 and try weights 0.05 and 0.1. With m=4/3/2 correct groups on 16/12/8-word training states, each correct group has a soft floor of 2.5%/3.33%/5%. The penalty becomes zero above the floor; it does not require equal positive probabilities or assign floors to incorrect candidates. The floor is an optimization target, not a guaranteed prediction bound.

Hold architecture, initialization, features and training settings fixed: standalone GINE, width 128, two layers, readout 128, original 15 edges, frozen Qwen vectors, fully trainable active norm/projection/GINE/readout, no phonetics or letter-edit additions. Batch 16, LR 0.0003016685, weight decay 0.0000950089, dropout 0, 5,000 updates per s2/s3/s4 state size, matched per-seed schedules and batch permutations. No early stopping, pruning, LR schedule or tuning beyond the two stated weights. Evaluate every 250 updates; select checkpoints by s4 Hit@1 with MRR tie-break. Compare these selected checkpoints primarily by harness solve rate; do not select checkpoints separately by harness.

Harness: normal checkpoint loader, beam 20, full-board group-score sums, four mistakes, public-feedback pruning and shuffle seed 0.

## Aggregate

| Metric | Probability mass | Floor weight 0.05 | Floor weight 0.1 |
|---|---:|---:|---:|
| s4 hit1 | 33.61% | 36.36% | 36.36% |
| s4 mrr | 0.4366 | 0.4578 | 0.4601 |
| s4 hit4 | 51.24% | 52.62% | 53.17% |
| s3 hit1 | 36.23% | 37.12% | 38.36% |
| s3 mrr | 0.4697 | 0.4785 | 0.4837 |
| s3 hit4 | 54.61% | 55.72% | 55.10% |
| s2 hit1 | 42.70% | 43.20% | 42.47% |
| s2 mrr | 0.5477 | 0.5565 | 0.5526 |
| s2 hit4 | 64.23% | 65.70% | 65.70% |
| Harness solves | 86/363 (23.69%) | 76/363 (20.94%) | 97/363 (26.72%) |
| Harness first_correct | 126/363 (34.71%) | 143/363 (39.39%) | 144/363 (39.67%) |
| yellow recovered | 179/363 | 179/363 | 185/363 |
| green recovered | 161/363 | 150/363 | 162/363 |
| blue recovered | 122/363 | 111/363 | 136/363 |
| purple recovered | 100/363 | 87/363 | 106/363 |
| Purple selected | 62/363 | 51/363 | 61/363 |
| Purple forced | 38/363 | 36/363 | 45/363 |
| Targeted letter-edit Purple recovered | 5/24 | 2/24 | 4/24 |

## Did it lift the other correct groups?

Computed on all 1,820 full-board candidates at the same selected checkpoints, using saved frozen vectors and no additional embedding inference. Other-correct mass excludes the highest-probability true group, whether or not that group is the top prediction.

| Metric | Probability mass | Weight 0.05 | Weight 0.1 |
|---|---:|---:|---:|
| correct_mass | 23.162% | 23.284% | 22.631% |
| other_correct_mass | 0.349% | 0.420% | 0.424% |
| minimum_correct_probability | 8.255e-06 | 1.101e-05 | 1.872e-05 |
| correct_groups_above_floor | 13.981% | 14.601% | 14.876% |
| dominant_share_of_correct_mass | 92.057% | 90.799% | 90.477% |
| log_floor_shortfall | 10.0819 | 7.0053 | 6.6726 |
| True-group mean_rank | 577.948 | 528.927 | 545.605 |
| True-group mrr | 0.117 | 0.125 | 0.125 |
| True-group top4_recall | 0.134 | 0.140 | 0.143 |

## Per seed

| Seed | Variant | s4 Hit@1 | Solves | Purple recovered | Selected update |
|---|---|---:|---:|---:|---:|
| 10 | probability-mass | 35.54% | 33/121 | 37/121 | 8250 |
| 10 | floor-0.05 | 36.36% | 26/121 | 31/121 | 2500 |
| 10 | floor-0.1 | 38.84% | 33/121 | 36/121 | 4000 |
| 11 | probability-mass | 33.06% | 28/121 | 35/121 | 7250 |
| 11 | floor-0.05 | 38.84% | 24/121 | 28/121 | 5500 |
| 11 | floor-0.1 | 37.19% | 33/121 | 37/121 | 2500 |
| 12 | probability-mass | 32.23% | 25/121 | 28/121 | 750 |
| 12 | floor-0.05 | 33.88% | 26/121 | 28/121 | 1750 |
| 12 | floor-0.1 | 33.06% | 31/121 | 33/121 | 1750 |

## Paired changes versus probability mass

- floor-0.05: 23 new solves, 33 lost solves; 23 new Purple recoveries, 36 lost Purple recoveries.
- floor-0.1: 30 new solves, 19 lost solves; 29 new Purple recoveries, 23 lost Purple recoveries.

All experiment weights and settings are recorded in `manifest.json`; full metrics and checkpoint paths are in `summary.json`, with per-board gold-group ranks and log probabilities in `probability-audit.json`. The default training objective remains unchanged. These fixed-setting validation comparisons are not independent confirmation on unseen puzzles.

## Interpretation

Weight 0.1 is the promising validation variant: 97/363 solves versus 86/363 (26.72% versus 23.69%, +3.03 percentage points). Per-seed solve counts are 33 to 33, 28 to 33, and 25 to 31; the gain is not concentrated in one seed. The paired changes are 30 new solves and 19 lost solves, not eleven universally improved puzzles. Weight 0.05 instead loses ten net solves. Both regularized variants have exactly 36.36% aggregate raw s4 Hit@1, but different solve rates (20.94% versus 26.72%). This supports evaluating the harness rather than selecting a penalty weight on standalone Hit@1 alone.

The penalty modestly changes the intended behavior on validation: other-correct probability mass rises from 0.349% to 0.424%, mean log-floor shortfall falls from 10.0819 to 6.6726, and the strongest correct group's mean conditional share falls from 92.06% to 90.48%. The easy correct choice still dominates. Only 14.88% of correct groups meet the 2.5% full-board floor, versus 13.98% for control; the regularizer does not enforce the floor on unseen validation groups. Total correct mass is slightly lower (23.16% to 22.63%), so probability redistribution alone is not a sufficient explanation for the solve improvement.

This is not a direct Purple breakthrough. Purple recovery rises 100 to 106, but early Purple selections fall 62 to 61 and forced last-group recoveries rise 38 to 45. Curated letter-edit Purple recoveries fall 5/24 to 4/24. Blue recovery has the largest absolute color gain, 122 to 136. The new letter-edit edge features remain disabled in this comparison.

Retain weight 0.1 as an opt-in validation candidate, not an independently confirmed replacement baseline. The same puzzles have been used repeatedly for selection and there is no new test-set evidence. Inference uses the unchanged single-GINE architecture, with no extra inference stage or floor computation. All six runs completed 15,000 updates; selected checkpoints occur at 1,750 to 5,500 updates. Fourteen targeted loss/training regression tests passed. All six checkpoints ran through the normal harness loader, and the independent rank/probability audit reproduced the s4 Hit@1 metrics for all nine runs.
