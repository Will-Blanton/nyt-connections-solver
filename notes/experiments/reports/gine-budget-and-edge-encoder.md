# GINE training budget and nonlinear edge encoding

Two stages, executed sequentially. The notebooks and held-out test split remain
outside this experiment.

1. Repeat linear-edge GINE and its zero-edge control for seeds 0–3 with up to
   3,000 updates. Use the same frozen text checkpoint, LR 1e-4, weight decay 1e-6,
   batch size 16, and s4 Hit@1/MRR selection. Early stopping is available after
   1,500 updates with eight stale checks (100 updates per check).
2. Use the budget evidence from stage 1 to compare a nonlinear edge encoder
   against the linear encoder, including matching zero-edge controls. Keep the
   two GINE layers, width 32, gate, and frozen text scorer fixed.

All runs are performed one at a time. Report s4/s3/s2 ranking and group-color
metrics using each run's s4-selected checkpoint. Initial zero-correction
checkpoints remain eligible. The stability assessment is conditional on this
one frozen backbone and validation split.

Future experiment noted: GINE for initial/global word encoding followed by set
attention within each candidate, versus the reverse ordering. The current path
already supplies globally and candidate-attention-encoded words to GINE; swapping
the order is a distinct ablation and is not part of these two stages.

## Stage 1

Completed eight sequential runs (four matched seed pairs). Detailed report:
[gine-long-budget.md](gine-long-budget.md).

| Level | 1,500-update edge Hit@1 | Longer-budget edge Hit@1 | Short MRR | Longer MRR |
| --- | ---: | ---: | ---: | ---: |
| s4 | 18.60% | 21.07% | 0.2566 | 0.2858 |
| s3 | 19.58% | 21.44% | 0.2853 | 0.3099 |
| s2 | 24.28% | 26.58% | 0.3711 | 0.3956 |

Seed-level s4 Hit@1 moved from 21.49/16.53/19.01/17.36% to
23.97/19.83/19.01/21.49%. Seed 2 restored the same checkpoint; three other
seeds improved. The longer-budget controls average 15.91% s4 Hit@1, so the
edge benefit remains beyond extra scoring capacity. Each row uses that run's
best s4-selected checkpoint, not the last update or a separate per-level winner.

Decision: use the same 3,000-update maximum, minimum 1,500 updates before stopping,
and eight stale evaluations for the nonlinear-edge stage. The comparison changes
both budget and early-stop availability relative to the original full-budget
1,500-update runs; it does not claim that every run completed 3,000 updates.

## Stage 2

The nonlinear encoder replaces each layer's bias-free 15 -> 32 edge projection
with bias-free 15 -> 32 -> ReLU -> 32 layers. Extra weights are initialized after
all existing parameters to preserve shared initializations under matched seeds.
Zero raw edges therefore still produce zero edge evidence. The encoder adds
2,048 trainable parameters (16,100 total branch parameters versus 14,052).
Its paired zero-edge control accounts for the added architecture.

Completed four seeds with edges and four matching zero-edge controls, all run
sequentially. Tests cover baseline-equivalent initial scoring, gradient flow,
frozen backbone, candidate permutations, reload, and zero-edge equivalence of
the linear and nonlinear message encoders.

Detailed report: [gine-nonlinear-edges.md](gine-nonlinear-edges.md).

| Seed | Linear s4 Hit@1 | Nonlinear s4 Hit@1 | Nonlinear minus linear |
| --- | ---: | ---: | ---: |
| 0 | 23.97% | 21.49% | -2.48 pp |
| 1 | 19.83% | 24.79% | +4.96 pp |
| 2 | 19.01% | 18.18% | -0.83 pp |
| 3 | 21.49% | 15.70% | -5.79 pp |

Full comparison of mean ranking performance across four matched seeds. Every
state-size metric comes from the same s4-selected checkpoint within a run.

| Model / maximum updates | Level | Hit@1 | MRR | Hit@4 |
| --- | --- | ---: | ---: | ---: |
| Linear / 1,500 | s4 | 18.60% | 0.2566 | 29.55% |
| Linear / 3,000 | s4 | 21.07% | 0.2858 | 33.68% |
| Nonlinear / 3,000 | s4 | 20.04% | 0.2691 | 29.75% |
| Linear / 1,500 | s3 | 19.58% | 0.2853 | 33.78% |
| Linear / 3,000 | s3 | 21.44% | 0.3099 | 36.98% |
| Nonlinear / 3,000 | s3 | 20.56% | 0.2949 | 34.81% |
| Linear / 1,500 | s2 | 24.28% | 0.3711 | 45.21% |
| Linear / 3,000 | s2 | 26.58% | 0.3956 | 47.97% |
| Nonlinear / 3,000 | s2 | 26.03% | 0.3868 | 47.11% |

The nonlinear encoder loses to the linear version on s4 Hit@1 in three of four
seeds and on mean Hit@1, MRR, and Hit@4 at every state size. Its s4 Hit@1 sample
standard deviation is 3.96 percentage points versus 2.19 for linear. Its best
individual seed is higher, but this is not a consistent architecture benefit.
All nonlinear zero-edge controls reproduce their linear counterparts' selected
validation metrics, supporting the intended controlled comparison.

Purple-specific mean top4 coverage:

| Level | Short linear | Longer linear | Nonlinear |
| --- | ---: | ---: | ---: |
| s4 | 1.03% | 1.24% | 1.65% |
| s3 | 2.41% | 2.62% | 2.75% |
| s2 | 9.92% | 10.26% | 10.40% |

The small Purple increases do not offset the aggregate ranking loss, and
full-board coverage remains only a handful of groups per seed. Richer encoding
of the same fifteen scalar channels has not resolved the Purple evidence gap.

## Decision and limitations

Retain linear-edge GINE with the longer budget and early stopping as the current
baseline. These sixteen new runs support extending training before adding this
particular nonlinear edge encoder. They do not establish that nonlinear edge
encoders are universally worse: the budget, optimizer, and stopping policy were
held fixed, and one nonlinear run stopped at the earliest permitted update
(1,500). No separate nonlinear-specific hyperparameter tuning was performed.

The same four seeds, frozen backbone, and validation puzzles were reused. These
are paired branch optimization experiments, not independent puzzle samples or
backbone retraining. s3/s2 are oracle residual states, not actual feedback-search
trajectories. The held-out test split was not evaluated.

Reproduce the reports:

```bash
nix develop -c uv run python scripts/summarize_gine_stability.py --updates 3000 --allow-early-stop --output experiments/gine-long-budget.md
nix develop -c uv run python scripts/summarize_gine_stability.py --updates 3000 --allow-early-stop --variants gated-gine-mlp-edge-control gated-gine-mlp-edges --output experiments/gine-nonlinear-edges.md
```

Raw JSON histories, configurations, selected checkpoint updates, and per-color
metrics are linked in those reports. Seven targeted tests pass, including both
linear and nonlinear GINE branch backward/permutation/frozen-base/reload checks.
