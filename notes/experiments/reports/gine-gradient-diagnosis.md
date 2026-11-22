# GINE gradient diagnosis

Six validation-only runs: seeds 0 and 1 for (1) transferred frozen 1024→256 projection, (2) the same transferred projection with both projection and input LayerNorm trainable, and (3) the scratch single Linear/ReLU residual adapter. Qwen embeddings remain frozen in all runs. Same split, 15 edge channels, LR 1e-4, weight decay 1e-6, up to 3000 updates, minimum stopping budget 1500, patience 8, validation every 100 updates. Each frozen/unfrozen pair starts from the same weights. All active non-Qwen modules train in the unfrozen and adapter variants; inherited unused attention modules remain frozen.

The optional `--diagnose-gradients` runner flag samples steps 1, 10, and each evaluation step. Measurements include unscaled parameter gradients (after GradScaler unscale), actual AdamW update/weight ratios, drift from initial weights, activation/activation-gradient RMS, and ReLU activity. Hooks sample training batches only. Summary statistics below use the common first 1500 updates (17 sampled batches per run) so differing early-stop times do not confound the window. Mixed state sizes and identical state/batch sequences are used within matched seeds. Frozen tensors' zero gradients are expected and excluded from vanishing-gradient conclusions.

## Findings

There is no evidence of broad vanishing gradients in these runs. Every measured trainable module received nonzero gradients on every sampled batch in the common window, including the input projection, trainable LayerNorm, both message MLPs, both edge projections, and adapter. Module-level nonzero gradients do not mean that every individual neuron or parameter is active.

Median gradient RMS by seed (0 / 1):

| Module | Frozen projection | Trainable projection + norm | Single-layer adapter |
|---|---:|---:|---:|
| Input projection / adapter | frozen | 0.0256 / 0.0260 | 0.00387 / 0.00525 |
| GINE node projection | 0.115 / 0.106 | 0.134 / 0.143 | 0.0527 / 0.0713 |
| First message MLP | 0.175 / 0.129 | 0.294 / 0.257 | 0.133 / 0.242 |
| Second message MLP | 0.0324 / 0.0355 | 0.0436 / 0.0561 | 0.0305 / 0.101 |
| First edge projection | 0.0545 / 0.0577 | 0.0281 / 0.0277 | 0.00713 / 0.0108 |
| Second edge projection | 0.0306 / 0.0197 | 0.0236 / 0.0143 | 0.00832 / 0.00990 |

These values depend on parameterization and are not direct measures of usefulness. Still, early-layer gradients do not collapse relative to later layers. The trainable 1024→256 projection's median per-step update/weight ratio is about 0.17% in both seeds. Its end-of-training weight drift is 34.6% and 42.5% from initialization. LayerNorm also receives gradients and updates. The adapter's median per-step update/weight ratio is about 0.18%. Thus these transformations are moving substantially, rather than failing to receive learning signals. Drift is measured at training end before restoring the selected checkpoint.

The single-layer adapter's median ReLU active fraction is 56.6% / 57.7%, and about 14.0% / 15.0% of units have no positive activation in an individual sampled batch. This does not identify permanently dead units. In the unfrozen projected models, some GINE MLP ReLUs have more batch-inactive units late in training (up to 44% in the first MLP), but substantial gradient flow remains. ReLU hooks cover module ReLUs; the functional ReLU inside the message formula was not instrumented separately.

At the common update 1500, frozen versus unfrozen-projection mixed-state training loss is 2.761→2.320 for seed 0 and 2.729→2.219 for seed 1. Selected validation s4 Hit@1 is 28.10→17.36% and 28.93→20.66%, respectively. Better training loss with worse validation plus substantial representation updates is more consistent with a generalization/representation-drift problem than widespread vanishing gradients. It does not establish the precise cause or rule out weak signals for particular features/groups. Edge projection gradients are smaller in the trainable/adapter models, which motivates an edge-use ablation but does not establish that edges are ignored.

Instrumented adapter reruns selected 13.22% / 17.36% s4 Hit@1. These differ from earlier adapter runs; do not substitute these two-seed diagnostic reruns for the prior four-seed architecture comparison. GPU numerical reproducibility is not enforced. The probe preserves the training objective and optimizer schedule, and a CPU test verifies identical optimizer-step tensors with/without measurement.

## Artifacts and reproduction

Full sampled records and validation histories: `experiments/gradient-diagnosis/*.json`. Aggregate common-window summaries: `experiments/gradient-diagnosis-summary.json`. Probe: `scripts/graph_training_probe.py`; regenerate summaries with `nix develop -c uv run python scripts/summarize_gradient_diagnosis.py`.

Use the normal experiment command with `--diagnose-gradients --output-dir experiments/gradient-diagnosis`. For `standalone-gine-trainable-projection`, also supply `--train-input-norm`; variants `standalone-gine-projected` and `standalone-gine-adapter-single` supply the other two conditions. Use seeds 0 and 1, 3000 updates, minimum stop 1500, patience 8, LR 1e-4, weight decay 1e-6, and the existing `20261023T150230Z-text-only.pt` reference. Only training/validation subsets were used. No notebooks, held-out test evaluations, or harness strategies changed.
