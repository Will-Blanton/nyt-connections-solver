# Phonetic node encodings and feature fusion

Three matched seeds (10–12) per configuration; 15,000 updates per run; validation-only selection and evaluation. Qwen stays frozen. The original learned text LayerNorm and 1024→256 projection, candidate-local GINE, readout, phonetic projection/gate and CNN (where present) all train jointly. No auxiliary loss, LR schedule, backbone freeze, test evaluation, notebook edits, or extra hyperparameter tuning.

## Representations and fusion

- **Phoneme n-grams:** 128-dimensional signed-hashed 1/2/3-gram vectors, including word-boundary markers. Normalize each pronunciation, average alternatives, normalize again. The vectors are fixed features; their projection is learned.
- **Learned sequence CNN:** 24-dimensional learned phoneme embeddings, three 32-channel convolutions (kernels 3/5/7), masked max pooling per pronunciation, learned 96→128 readout, averaging across valid alternatives, then L2 normalization. This encoder is trained from scratch with puzzle supervision; it is not a pretrained speech encoder.
- **Linear fusion:** `z_text + P(h_phone)` before candidate-local GINE.
- **Gated fusion:** `z_text + sigmoid(w [z_text, P(h_phone)] + b) * P(h_phone)`, with one learned scalar gate per word.

The extra projection starts at zero in every arm; the gate starts at 0.5. All shared model tensors are copied from identical per-seed initializations, and new module construction preserves the baseline RNG state. Initial predictions match the semantic model; new branches learn their contribution. Different encoders have different parameter counts, so this does not isolate representation from capacity.

Use at most eight distinct stress-free CMUdict pronunciations and 32 phonemes per pronunciation. Unknown words have zero phonetic contribution and preserve their text branch. The 1408-dimensional cached input contains 1024 text values, 128 n-gram values, and 8×32 integer phone IDs stored losslessly as float32. IDs are consumed only by the CNN, never by the text projection or a numeric similarity calculation. Resources, vocabulary order and layout are stored in checkpoint feature metadata. [Conv1d reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).

## Controlled training setup

Original tuned parameters: `{"dropout": 0.0, "gine_layers": 2, "hidden_width": 128, "lr": 0.0003016684667937828, "readout_width": 128, "weight_decay": 9.500890359387789e-05}`. Batch 16, mixed s2/s3/s4 probability-mass loss, evaluation every 250 updates, no early stopping or pruning. Select each checkpoint by validation s4 Hit@1, MRR only breaking checkpoint ties.

The sweep holds edges fixed: original 15 plus the four whole-word sound channels. The two substring-sound channels remain masked because the prior experiment did not show a consistent benefit. The edge-only control has the same node payload and text/GINE initialization, with phonetic-node fusion disabled.

Selected node-fusion family: **ngram_linear**, by mean validation s4 Hit@1 and mean MRR for ties, before the ablation. The same family is then retrained with only the added sound edges masked. “Selected nodes only” therefore means **phonetic nodes plus the original 15 edges**, not an edgeless graph. All three seed results are retained; no favorable seed is selected for the aggregate.

## Validation results

| Variant | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 | Solve rate |
|---|---:|---:|---:|---:|---:|---:|
| edge_only | 35.26% | 0.4483 | 51.79% | 36.98% | 42.98% | 25.34% |
| ngram_linear | 35.54% | 0.4584 | 53.44% | 37.67% | 42.47% | 25.62% |
| ngram_gated | 33.33% | 0.4207 | 46.28% | 35.26% | 40.96% | 24.24% |
| cnn_linear | 33.33% | 0.4201 | 45.73% | 34.16% | 40.22% | 26.72% |
| cnn_gated | 33.61% | 0.4222 | 50.14% | 34.37% | 40.31% | 24.24% |
| selected_nodes_only | 34.71% | 0.4334 | 47.93% | 35.61% | 41.05% | 25.34% |

s3/s2 ranking uses oracle reduced states, not feedback trajectories. Harness evaluations use the same 121 validation puzzles, beam 20, full-board scoring, four mistakes, shuffle seed 0 and existing public-feedback pruning. The normal checkpoint loader reconstructed each model and its original features for replay.

| Seed | Variant | s4 Hit@1 | Solves /121 | Selected update |
|---:|---|---:|---:|---:|
| 10 | edge_only | 34.71% | 28 | 3500 |
| 10 | ngram_linear | 35.54% | 28 | 7500 |
| 10 | ngram_gated | 33.88% | 27 | 9000 |
| 10 | cnn_linear | 33.06% | 31 | 9000 |
| 10 | cnn_gated | 33.88% | 29 | 1500 |
| 11 | edge_only | 37.19% | 32 | 1250 |
| 11 | ngram_linear | 35.54% | 32 | 5250 |
| 11 | ngram_gated | 33.88% | 32 | 2500 |
| 11 | cnn_linear | 33.06% | 35 | 4000 |
| 11 | cnn_gated | 33.88% | 31 | 4250 |
| 12 | edge_only | 33.88% | 32 | 3000 |
| 12 | ngram_linear | 35.54% | 33 | 2750 |
| 12 | ngram_gated | 32.23% | 29 | 1250 |
| 12 | cnn_linear | 33.88% | 31 | 6250 |
| 12 | cnn_gated | 33.06% | 28 | 1250 |
| 10 | selected_nodes_only | 35.54% | 29 | 8250 |
| 11 | selected_nodes_only | 34.71% | 35 | 9250 |
| 12 | selected_nodes_only | 33.88% | 28 | 2500 |

## Group diagnostics

| Variant | Yellow group Top4 | Green | Blue | Purple | Mean known-word gate |
|---|---:|---:|---:|---:|---:|
| edge_only | 26.72% | 15.98% | 7.99% | 4.68% | disabled |
| ngram_linear | 29.20% | 15.15% | 6.61% | 4.13% | 1.000 |
| ngram_gated | 24.79% | 15.98% | 5.23% | 3.58% | 0.563 |
| cnn_linear | 23.14% | 14.88% | 6.61% | 3.58% | 1.000 |
| cnn_gated | 25.90% | 15.70% | 6.34% | 3.86% | 0.785 |
| selected_nodes_only | 25.07% | 14.88% | 6.89% | 3.31% | 1.000 |

Group Top4 concerns each specific true group; state Hit@1 accepts any true group. A gate average near its initial value does not prove individual gates are constant or that the projected contribution is zero.

| Variant | Yellow recovered /363 | Green | Blue | Purple |
|---|---:|---:|---:|---:|
| edge_only | 193 | 154 | 129 | 109 |
| ngram_linear | 192 | 155 | 127 | 107 |
| ngram_gated | 181 | 154 | 127 | 98 |
| cnn_linear | 189 | 152 | 127 | 114 |
| cnn_gated | 167 | 146 | 121 | 99 |
| selected_nodes_only | 177 | 162 | 129 | 106 |

| Variant | Sound-labelled group Top4 | Sound-labelled MRR |
|---|---:|---:|
| edge_only | 6.25% | 0.0536 |
| ngram_linear | 6.25% | 0.0398 |
| ngram_gated | 6.25% | 0.0377 |
| cnn_linear | 2.08% | 0.0279 |
| cnn_gated | 4.17% | 0.0446 |
| selected_nodes_only | 0.00% | 0.0018 |

There are 16 label-audited sound groups per seed. Repeated measurements are correlated; the labels classify diagnostic rows only and never enter training. Full per-group ranks are in `group-ranks.json`.

| Variant | Mean phonetic contribution norm / text norm |
|---|---:|
| edge_only | 0.0000 |
| ngram_linear | 0.0273 |
| ngram_gated | 0.0185 |
| cnn_linear | 0.0633 |
| cnn_gated | 0.0170 |
| selected_nodes_only | 0.0286 |

Contribution ratios are averaged over known words within each original validation board, then across boards/seeds. They describe use of the branch, not causal feature importance or an accuracy gain.

## Coverage and cost

| Split | Distinct words | Known pronunciation | Variant cap hit | Phone-length truncation |
|---|---:|---:|---:|---:|
| train | 7611 | 7006 | 0 | 0 |
| val | 1667 | 1596 | 0 | 0 |

| Variant | Trainable parameters | Warm-cache mean game seconds |
|---|---:|---:|
| edge_only | 434,563 | 0.295 |
| ngram_linear | 434,563 | 0.292 |
| ngram_gated | 435,076 | 0.298 |
| cnn_linear | 459,555 | 0.313 |
| cnn_gated | 460,068 | 0.312 |
| selected_nodes_only | 434,563 | 0.288 |

The CNN runs once per word before candidates are enumerated; its output is reused within each forward. Resource and node payload caching is shared with the harness. Timings depend on guesses/states, cache warmth and system load; they are not a controlled inference benchmark or a claim about cold-start latency. Parameter counts use `requires_grad`; the edge-only row includes the disabled phonetic projection, which receives no gradients.

## Findings

The selected fixed n-gram + linear fusion is essentially tied with edge-only: 35.54% versus 35.26% s4 Hit@1 and 25.62% versus 25.34% solves. That amounts to one additional correct original-board top choice and one additional solve across 363 repeated seed/puzzle evaluations, not a compelling general gain.

CNN + linear fusion has the highest mean harness solve rate, 26.72%, with paired solve counts 31/35/31 versus 28/32/32 for edge-only. The mean gain is 1.38 percentage points and the third seed worsens. Its Hit@1 drops to 33.33%; MRR/Hit@4 also decline. This is a possible partition-search trade-off, not evidence of uniformly better subset decisions. Gating did not improve solve rate for either representation.

The preselected n-gram-linear node-only ablation gets 34.71% Hit@1 and 25.34% solves, versus 35.54%/25.62% with sound edges. This gives little evidence for a material combined-modality benefit. The ablation is for the family selected on Hit@1; CNN-linear did not receive its own node-only comparison, so its harness gain cannot establish node/edge complementarity.

Sound-labelled group Top4 remains 6.25% for edge-only and both n-gram fusions; it is 2.08% for CNN-linear, 4.17% for CNN-gated and zero for the selected node-only arm. Thus the modest overall differences are not a demonstrated improvement in the intended sound-group retrieval task.

The branches are trainable and used: mean known-word gates are approximately 0.56 for gated n-grams and 0.79 for gated CNNs. The phonetic contribution norm averages about 2.7% of the text norm for linear n-grams and 6.3% for linear CNNs, versus 1.9%/1.7% with gates. These are descriptive measurements; they do not prove why a variant succeeds or fails. Zero-start fusion, relative scaling, mean pooling of pronunciation alternatives and fixed baseline optimization settings remain limitations, not isolated causes.

Keep the existing notebook baseline. Retain the new variants as reproducible experiments; the CNN-linear harness result is worth confirming before promoting it. No test evaluation or further tuning was used to resolve these small validation differences.

## Limits and reproducibility

The same validation puzzles select checkpoints and the fusion family. Fresh seeds measure training variability, not new held-out data; selected-family results are exploratory and may be optimistic. No test puzzles were evaluated. The notebook baseline and default feature configuration remain unchanged; phonetic nodes are opt-in with `phonetic_nodes=True`. Saved models use the shared registry and can be loaded by Streamlit without architecture-specific UI changes.

Run: `nix develop -c uv run python scripts/run_phonetic_fusion_experiment.py`. Report: `nix develop -c uv run python scripts/summarize_phonetic_fusion.py`. Manifest, selection, progress, all per-seed metrics, checkpoint paths and replay histories: [summary.json](gine-phonetic-fusion/summary.json). Full-budget validation curves: [learning-curves.html](gine-phonetic-fusion/learning-curves.html).
