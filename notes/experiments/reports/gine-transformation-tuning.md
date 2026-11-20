# Learned-transformation edge hyperparameter tuning

Validation-only tuning of the full-transformation, text-node GINE. Architecture and probability-mass objective are fixed. Frozen Qwen vectors remain frozen; all active projection and GINE parameters train. No notebook or test-set changes.

## Setup

12 Optuna trials, seed 0. Six full startup trials include the inherited settings, lower LRs (1e-4 and 3e-5), and regularized variants. Search: LR 1e-5–5e-4 (log), weight decay 1e-5–1e-2 (log), GINE dropout 0 / .05 / .1 / .2. Hidden width 128, two GINE layers, readout 128, transformation width 64 and 16 learned edge channels remain fixed.

After trial 7, explicitly queued dropout .05 and .2 at LR 3e-4 / WD .001 to cover the proposed range instead of relying on the adaptive sampler to explore unseen dropout values. These remain within the twelve trials; the decision and exact parameters are in dropout-coverage-queue.json.

Each completed run gets 30,000 updates, batch 16, balanced 10,000 updates at each state size, validation every 500. No scheduler or early stopping. Median pruning starts after six full trials, at >=20,000 updates with at least four comparisons. Selection uses full-board Hit@1, checkpoints use MRR to break ties. Rerun selected parameters with seeds 10/11/12; also repeat original-only masking at the same parameters. The latter is not independently tuned. Existing controls have the same budget.

Diagnostics: eval-mode loss, Hit@1 and MRR on all validation states and a fixed 96-state training sample per size every 2,000 updates plus first/last. Training/validation differences are sampled estimates. Gradient probes record new encoders and base modules at updates 1, 10 and every 500. The additional GINE edge columns start at zero, so zero encoder gradient on the very first step is expected. Probes are tested to preserve updates.

Trials use one tuning seed; three-seed reruns check sensitivity to initialization, not generalization to new data. The 121 validation puzzles have been reused extensively. Harness: beam 20, full-board scores, four mistakes, shuffle seed zero.

## Selected settings

Trial 9, tuning-seed Hit@1 38.02%.

```json
{
  "hidden_width": 128,
  "gine_layers": 2,
  "readout_width": 128,
  "dropout": 0.2,
  "lr": 0.0003,
  "weight_decay": 0.001
}
```

## Trial results

| Trial | Status | LR | WD | Dropout | Updates | Best Hit@1 | Best update |
|---|---|---:|---:|---:|---:|---:|---:|
| trial-00 | complete | 0.000302 | 9.5e-05 | 0.0 | 30000 | 26.45% | 4500 |
| trial-01 | complete | 0.0001 | 9.5e-05 | 0.0 | 30000 | 28.93% | 10000 |
| trial-02 | complete | 3e-05 | 9.5e-05 | 0.0 | 30000 | 27.27% | 6000 |
| trial-03 | complete | 0.0001 | 0.001 | 0.1 | 30000 | 31.40% | 3500 |
| trial-04 | complete | 3e-05 | 0.001 | 0.1 | 30000 | 28.10% | 27000 |
| trial-05 | complete | 0.000302 | 0.001 | 0.1 | 30000 | 32.23% | 1500 |
| trial-06 | pruned | 1.13e-05 | 3.78e-05 | 0.1 | 20000 | 24.79% | 6500 |
| trial-07 | complete | 0.000491 | 1.33e-05 | 0.1 | 30000 | 34.71% | 6500 |
| trial-08 | complete | 0.0003 | 0.001 | 0.05 | 30000 | 33.06% | 6500 |
| trial-09 | complete | 0.0003 | 0.001 | 0.2 | 30000 | 38.02% | 14000 |
| trial-10 | complete | 0.00015 | 0.00947 | 0.2 | 30000 | 33.06% | 2500 |
| trial-11 | pruned | 0.000282 | 1.42e-05 | 0.1 | 20000 | 28.93% | 1500 |

## Optimization diagnostics (tuning seed)

| Trial | Update | Train s4 loss | Val s4 loss | Train s4 Hit@1 | Val s4 Hit@1 |
|---|---:|---:|---:|---:|---:|
| trial-00 | 500 | 2.4260 | 3.8767 | 47.92% | 20.66% |
| trial-00 | 30000 | 0.0021 | 15.2354 | 100.00% | 19.01% |
| trial-01 | 500 | 3.0927 | 4.0289 | 33.33% | 17.36% |
| trial-01 | 30000 | 0.0013 | 9.0969 | 100.00% | 23.97% |
| trial-02 | 500 | 3.7897 | 4.7365 | 21.88% | 11.57% |
| trial-02 | 30000 | 0.0147 | 13.7047 | 100.00% | 22.31% |
| trial-03 | 500 | 3.1116 | 4.3142 | 27.08% | 20.66% |
| trial-03 | 30000 | 0.0007 | 9.3560 | 100.00% | 25.62% |
| trial-04 | 500 | 4.3323 | 4.8493 | 17.71% | 9.92% |
| trial-04 | 30000 | 0.0045 | 8.6919 | 100.00% | 20.66% |
| trial-05 | 500 | 2.6151 | 3.6584 | 42.71% | 23.97% |
| trial-05 | 30000 | 0.0064 | 8.8094 | 100.00% | 23.97% |
| trial-06 | 500 | 5.9364 | 5.9600 | 9.38% | 10.74% |
| trial-06 | 20000 | 0.0197 | 6.6226 | 100.00% | 23.14% |
| trial-07 | 500 | 2.7044 | 3.5566 | 42.71% | 27.27% |
| trial-07 | 30000 | 0.0011 | 11.3965 | 100.00% | 31.40% |
| trial-08 | 500 | 2.4781 | 3.7914 | 50.00% | 24.79% |
| trial-08 | 30000 | 0.0006 | 8.8185 | 100.00% | 28.93% |
| trial-09 | 500 | 3.2086 | 3.8844 | 39.58% | 26.45% |
| trial-09 | 30000 | 0.0010 | 9.6053 | 100.00% | 28.93% |
| trial-10 | 500 | 3.6764 | 4.4077 | 31.25% | 21.49% |
| trial-10 | 30000 | 0.0059 | 7.1820 | 100.00% | 28.10% |
| trial-11 | 500 | 2.7641 | 3.6458 | 44.79% | 23.97% |
| trial-11 | 20000 | 0.0040 | 8.0296 | 100.00% | 23.97% |

Gradient/update measurements are available in summary.json and per-run training.gradient_diagnostics. Zero gradient into the extra branch at step 1 is expected from zero-initialized downstream columns; subsequent nonzero measurements distinguish that from a permanently blocked branch.


## Matched reruns

| Metric | Ordinary GINE | Old full | Old original-only | Tuned full | Tuned original-only |
|---|---:|---:|---:|---:|---:|
| s4 hit1 | 0.3416 | 0.2920 | 0.2865 | 0.3306 | 0.3251 |
| s4 mrr | 0.4467 | 0.3835 | 0.3852 | 0.4185 | 0.4237 |
| s4 hit4 | 0.5262 | 0.4573 | 0.4628 | 0.4793 | 0.4793 |
| s3 hit1 | 0.3609 | 0.3099 | 0.3085 | 0.3540 | 0.3437 |
| s3 mrr | 0.4758 | 0.4168 | 0.4194 | 0.4480 | 0.4495 |
| s3 hit4 | 0.5640 | 0.5014 | 0.5110 | 0.5227 | 0.5241 |
| s2 hit1 | 0.4215 | 0.3820 | 0.3719 | 0.4045 | 0.3990 |
| s2 mrr | 0.5516 | 0.5078 | 0.4982 | 0.5179 | 0.5227 |
| s2 hit4 | 0.6635 | 0.6024 | 0.5923 | 0.6010 | 0.6175 |
| Harness solves | 96/363 (26.45%) | 57/363 (15.70%) | 59/363 (16.25%) | 69/363 (19.01%) | 66/363 (18.18%) |
| Harness first_correct | 146/363 (40.22%) | 103/363 (28.37%) | 105/363 (28.93%) | 119/363 (32.78%) | 125/363 (34.44%) |
| yellow recovered | 182 | 135 | 154 | 153 | 155 |
| green recovered | 160 | 127 | 131 | 135 | 135 |
| blue recovered | 136 | 97 | 97 | 101 | 94 |
| purple recovered | 106 | 70 | 75 | 84 | 83 |
| Purple selected | 59 | 46 | 52 | 57 | 60 |
| Targeted Purple selected | 0/12 | 0/12 | 0/12 | 0/12 | 0/12 |
| Purple forced | 47 | 24 | 23 | 27 | 23 |
| Targeted Purple forced | 2/12 | 1/12 | 1/12 | 2/12 | 2/12 |

## Per seed

| Arm | Seed | Hit@1 | Last Hit@1 | Best update | Solves | Purple direct / forced |
|---|---:|---:|---:|---:|---:|---:|
| control | 10 | 34.71% | 19.83% | 7500 | 35 | 22 / 17 |
| edges-text | 10 | 26.45% | 20.66% | 16500 | 17 | 17 / 4 |
| original-only | 10 | 23.97% | 17.36% | 500 | 20 | 18 / 9 |
| tuned | 10 | 33.06% | 31.40% | 26500 | 22 | 19 / 7 |
| tuned-original-only | 10 | 32.23% | 31.40% | 18000 | 22 | 15 / 12 |
| control | 11 | 34.71% | 20.66% | 5000 | 33 | 22 / 16 |
| edges-text | 11 | 30.58% | 25.62% | 1000 | 20 | 14 / 10 |
| original-only | 11 | 30.58% | 27.27% | 22000 | 18 | 13 / 9 |
| tuned | 11 | 33.88% | 25.62% | 22500 | 22 | 18 / 10 |
| tuned-original-only | 11 | 33.06% | 27.27% | 10000 | 22 | 21 / 6 |
| control | 12 | 33.06% | 26.45% | 2000 | 28 | 15 / 14 |
| edges-text | 12 | 30.58% | 23.97% | 17000 | 20 | 15 / 10 |
| original-only | 12 | 31.40% | 28.93% | 24000 | 21 | 21 / 5 |
| tuned | 12 | 32.23% | 23.97% | 2000 | 25 | 20 / 10 |
| tuned-original-only | 12 | 32.23% | 24.79% | 9500 | 22 | 24 / 5 |

## Paired harness changes

- tuned versus edges-text: 41 gains / 29 losses; Purple 42 gains / 28 losses.
- tuned-original-only versus original-only: 31 gains / 24 losses; Purple 35 gains / 27 losses.
- tuned versus control: 23 gains / 50 losses; Purple 25 gains / 47 losses.
- tuned versus tuned-original-only: 31 gains / 28 losses; Purple 31 gains / 30 losses.

## Concrete target groups

| Category | Ordinary GINE | Old full | Old original-only | Tuned full | Tuned original-only |
|---|---:|---:|---:|---:|---:|
| starting with rock bands | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced |
| ending in synonyms for "location" | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced |
| starting with slang for zero | 0 direct / 2 forced | 0 direct / 1 forced | 0 direct / 1 forced | 0 direct / 2 forced | 0 direct / 2 forced |
| starting with synonyms for "slush" | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced | 0 direct / 0 forced |

## Artifacts

[Learning curves](gine-transformation-tuning/learning-curves.html), [train/validation diagnostics](gine-transformation-tuning/diagnostics.html), [trial curves](gine-transformation-tuning/trial-curves.html), [raw summary](gine-transformation-tuning/summary.json).


## Extended budget check

Selected seed 10 because its best checkpoint was latest among the three full-model reruns. Run 60k updates with unchanged optimizer settings, batch size and balanced state schedule; save the best checkpoint from the first 30k updates of that SAME uninterrupted run. No freezing, schedule, optimizer restart or auxiliary objective. This is a separate diagnostic, not part of the matched 30k aggregates.

The first attempted extension was interrupted and preserved after its early same-seed trajectory differed from the earlier 30k run. The cause of that cross-process stochastic difference was not established. Compare the within-run checkpoints below to isolate budget; the earlier 30k run is shown separately in the plot. Maximum first-30k Hit@1 difference from that earlier run: 0.0826.

30k selected Hit@1 34.71%; 60k selected Hit@1 34.71% at update 16000. Best after 30k: 33.06%. Within-run 30k / 60k harness solves: 17 / 17 out of 121. Purple direct/forced: 13/7.

[Extended learning curve](gine-transformation-tuning/long-check-curves.html).


## Findings

Tuning improves full-transformation s4 Hit@1 from 29.20% to 33.06%, solves from 57 to 69/363, and Purple recovery from 70 to 84/363. All three full-model seeds improve selected Hit@1 and solve count over their old settings. The original negative comparison therefore overstated the regression caused by the architecture alone: inherited optimization/regularization settings materially contributed.

The winner keeps LR essentially unchanged (3e-4) and increases dropout to .2 and WD to .001. Lower LR alone was less useful. The .05 versus .2 comparison at identical LR/WD also favors .2 on the tuning seed. This supports stronger regularization rather than a blanket conclusion that the LR was too high. Dropout .2 is the largest tested, so this study does not establish an optimum beyond the searched range. Width, depth, batch size and schedules were not swept.

Ordinary GINE remains stronger: 34.16% Hit@1, 96/363 solves, 106/363 Purple recoveries. Tuned original-only is close to tuned full: 32.51% Hit@1 and 66/363 solves. Full readings gain 31 paired solves and lose 28; their incremental benefit is small and mixed. Original-only is evaluated at the full model's selected hyperparameters rather than independently tuned.

None of the four concrete covered substring categories is solved directly by either tuned arm (0/12 group-seed opportunities each). Both recover the zero-prefix group twice as a forced last group. Broad Purple gains therefore do not establish successful use of the intended transformations.

Learning curves still show overfitting: eval-mode training-sample Hit@1 reaches roughly 100%, while validation loss increases and late validation ranking fluctuates below selected peaks. On the tuned full reruns, best checkpoints occur at 26.5k / 22.5k / 2k updates, demonstrating why short universal budgets would be misleading. The added branch receives nonzero gradients and parameter updates after its deliberately blocked first step. That rules out a permanently disconnected branch, not every possible optimization difficulty.

Keep ordinary GINE as the baseline. Retain this experiment as evidence that each new architecture needs its own tuning and measured curves. These repeatedly reused validation puzzles cannot establish gains on fresh data. No default model or notebook was changed.

The controlled extended-budget check selects 34.71% Hit@1 after the first30k and 34.71% after60k, with 17 versus 17 solves. The best score observed strictly after30k is 33.06%. For this checked trajectory, another 30k updates does not fix the regression. This does not imply every future architecture or optimizer should use the same budget.
