# Learned transformation edges and node representations

Keep the existing baseline. Learned transformation matching fits training data readily but reduces aggregate validation ranking and harness solves under these fixed settings. Shared-vector nodes are the strongest new variant, but still below the control.

Validation only: 121 puzzles, three matched seeds (10/11/12), four arms, 30,000 updates per run. Retrain the control at the same budget. Repeated seeds reuse the same validation puzzles and prior selection history; 363 appearances are not 363 independent puzzles. No notebooks or test-set evaluation.

## Features and architecture

Use the original word plus proper contiguous substrings of at least three letters that are English WordNet words. Only ASCII single words up to 32 characters receive substring extraction. The original remains available for all inputs. Cache frozen Qwen embeddings for every alternative. Keep eight slots, original first, then round-robin prefix/suffix/internal alternatives ordered longest first within each type. Padding is explicitly masked.

Each alternative carries its extraction type, left/right removed-character counts divided by a fixed 32, normalized start/end span and retained fraction. Fixed-scale offsets preserve exact matches such as removing one leading character across different word lengths. No puzzle labels enter features.

Learn a 64-dimensional alternative representation. For each unordered word pair, compare all valid alternative pairs through a shared MLP using absolute embedding differences, products, position differences, and exact type/left-offset/right-offset compatibility. Elementwise max yields a 16-dimensional edge vector; concatenate with the original 15 scalar edges. Edges are symmetric with zero diagonal and computed once per state before candidate extraction. New GINE edge-projection columns start at zero; the learned pair network begins receiving gradients once those columns move.

The text arm keeps the original active text projection. The fused arm combines text projection with sigmoid-gated mean and max transformation summaries through a learned parallel fusion layer. Edges retain the unpooled alternatives. The constant arm starts all nodes from one shared learned vector, disables the original text projection, and relies on edge evidence and subsequent GINE messages. The control ignores the extra payload and supplies zero learned edge vectors.

This remains the existing candidate-local GINE, with two width-128 layers and readout 128. It does not add global message passing or relation nodes. Learned edges depend on the two words, not on a joint explanation shared by four words. Original alternatives are included, so a gain could reflect a learned original-word pair scorer as well as substring information; an original-only learned-edge control would be needed to separate those explanations.

## Training and evaluation

Ordinary probability-mass loss; no floor penalty or auxiliary loss. Batch 16, fixed LR 0.0003016685, weight decay 0.0000950089, dropout zero. Frozen Qwen vectors; active projection/GINE/transformation networks trainable as appropriate to each arm. Matched initialization of shared weights, data permutations and balanced schedules (10,000 updates per 8/12/16-word state size, roughly 165 full-board training passes). Full 30,000 updates, no pruning or early stopping; evaluate every 500 updates and restore best s4 Hit@1 with MRR tie-break. All arms use four CPU threads. New modules increase capacity; parameter counts below are not a capacity-matched ablation.

Normal harness: beam 20, full-board score sums, four mistakes, shuffle seed zero and public-feedback pruning. No harness-based checkpoint selection. Streamlit and harness use registered self-contained TransformationGINE checkpoints and reconstructable feature metadata.

Two incomplete startup controls were restarted before any learned arm completed: CPU batching oversubscription was fixed, then absolute offset matching was added. Logs and recovery records are retained. All reported runs use the final feature version and full matched budget.

## Coverage

| Split | Unique words | Proper alternatives available | Truncated at eight |
|---|---:|---:|---:|
| train | 7611 | 5343 (70.20%) | 68 |
| val | 1667 | 1145 (68.69%) | 11 |

## Aggregate results

| Metric | Control | Learned edges + text | Learned edges + fused nodes | Learned edges + constant nodes |
|---|---:|---:|---:|---:|
| s4 hit1 | 34.16% | 29.20% | 30.85% | 32.23% |
| s4 mrr | 0.4467 | 0.3835 | 0.3975 | 0.4206 |
| s4 hit4 | 52.62% | 45.73% | 45.45% | 47.66% |
| s3 hit1 | 36.09% | 30.99% | 32.09% | 36.09% |
| s3 mrr | 0.4758 | 0.4168 | 0.4301 | 0.4733 |
| s3 hit4 | 56.40% | 50.14% | 51.72% | 55.23% |
| s2 hit1 | 42.15% | 38.20% | 38.93% | 43.43% |
| s2 mrr | 0.5516 | 0.5078 | 0.5139 | 0.5585 |
| s2 hit4 | 66.35% | 60.24% | 60.28% | 66.02% |
| Harness solves | 96/363 (26.45%) | 57/363 (15.70%) | 70/363 (19.28%) | 77/363 (21.21%) |
| Harness first_correct | 146/363 (40.22%) | 103/363 (28.37%) | 119/363 (32.78%) | 122/363 (33.61%) |
| yellow recovered | 182/363 | 135/363 | 158/363 | 165/363 |
| green recovered | 160/363 | 127/363 | 141/363 | 146/363 |
| blue recovered | 136/363 | 97/363 | 101/363 | 119/363 |
| purple recovered | 106/363 | 70/363 | 78/363 | 88/363 |
| Purple selected | 59/363 | 46/363 | 45/363 | 44/363 |
| Purple forced | 47/363 | 24/363 | 33/363 | 44/363 |
| Embedded-fragment Purple recovered | 14/45 | 9/45 | 10/45 | 11/45 |
| Embedded-fragment Purple directly selected | 5/45 | 4/45 | 3/45 | 2/45 |
| Concrete covered Purple recovered | 2/12 | 1/12 | 4/12 | 1/12 |
| Concrete covered Purple directly selected | 0/12 | 0/12 | 1/12 | 0/12 |

Embedded-fragment mechanism labels are the earlier regex inventory, used only for audit. They are approximate and do not establish coverage of every intended transformation. Concrete coverage separately checks four manually specified groups: rock-band prefixes (cream/journey/kiss/rush), location suffixes (place/point/site/spot), zero prefixes (jack/nada/squat/zip), and slush prefixes (goo/mush/paste/pulp). All sixteen intended readings are present within the eight-slot cap. These labels never enter training or feature construction. Per-date recovery is recorded in summary.json.

| Variant | Trainable parameters | Mean train minutes | Mean validation minutes | Mean harness seconds/puzzle |
|---|---:|---:|---:|---:|
| control | 404,355 | 4.03 | 0.24 | 0.309 |
| edges-text | 484,787 | 5.25 | 0.28 | 0.345 |
| edges-fused | 583,412 | 5.37 | 0.29 | 0.338 |
| edges-constant | 220,595 | 4.98 | 0.27 | 0.320 |

Timings exclude initial feature preparation and cold embedding inference. Harness timings include variable search work, so they are not model-only microbenchmarks. Cached lexical vectors are reused across subsets and runs.

## Per seed and update budget

| Seed | Variant | s4 Hit@1 | Best through 15k | Best update | Solves | Purple direct / forced |
|---|---|---:|---:|---:|---:|---:|
| 10 | control | 34.71% | 34.71% | 7500 | 35/121 | 22 / 17 |
| 10 | edges-text | 26.45% | 26.45% | 16500 | 17/121 | 17 / 4 |
| 10 | edges-fused | 30.58% | 29.75% | 23500 | 23/121 | 14 / 12 |
| 10 | edges-constant | 30.58% | 30.58% | 500 | 23/121 | 13 / 14 |
| 11 | control | 34.71% | 34.71% | 5000 | 33/121 | 22 / 16 |
| 11 | edges-text | 30.58% | 30.58% | 1000 | 20/121 | 14 / 10 |
| 11 | edges-fused | 32.23% | 31.40% | 18000 | 18/121 | 14 / 7 |
| 11 | edges-constant | 33.06% | 33.06% | 1500 | 25/121 | 15 / 13 |
| 12 | control | 33.06% | 33.06% | 2000 | 28/121 | 15 / 14 |
| 12 | edges-text | 30.58% | 28.10% | 17000 | 20/121 | 15 / 10 |
| 12 | edges-fused | 29.75% | 29.75% | 500 | 29/121 | 17 / 14 |
| 12 | edges-constant | 33.06% | 33.06% | 500 | 29/121 | 16 / 17 |

## Paired differences from control

- edges-text: 19 new solves, 58 lost; 19 new Purple recoveries, 55 lost.
- edges-fused: 24 new solves, 50 lost; 22 new Purple recoveries, 50 lost.
- edges-constant: 21 new solves, 40 lost; 21 new Purple recoveries, 39 lost.

## Concrete intended-mechanism recovery

Each cell is direct selections / forced final-group recoveries across three seeds. These four cases were selected for lexical coverage, not because the learned model solved them.

| Date and mechanism | Control | Edges + text | Edges + fused | Edges + constant |
|---|---:|---:|---:|---:|
| 2024-03-20 starting with rock bands | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 |
| 2025-11-18 ending in synonyms for "location" | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2026-03-08 starting with slang for zero | 0 / 2 | 0 / 1 | 0 / 3 | 0 / 1 |
| 2026-04-05 starting with synonyms for "slush" | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

## Interpretation

The ordinary control wins aggregate 16-word Hit@1, Hit@4, MRR and harness solve rate. Learned edges with text nodes lose solves in all three seeds (35/33/28 to 17/20/20). Fusion improves the new-edge aggregate but remains below control (23/18/29). Shared-vector nodes are stronger still (23/25/29), but do not recover the control (96 to 77 total solves). They match aggregate 12-word Hit@1 and slightly exceed 8-word Hit@1, while trailing on the most important 16-word state and full solves.

This does not establish that original text node embeddings are inherently misleading. Constant nodes remove about 264,000 trainable text-projection parameters, so regularization/capacity changes are entangled with input choice. Their learned edges still include original-word text embeddings. None of these runs removes ordinary semantic information from the system entirely. A shared-node model with only the original scalar edges would be a clean next isolation before further widening networks.

All runs completed 30,000 updates and sixty validation observations. Four selected checkpoints occur after 15,000 updates; three runs have strictly better Hit@1 in the second half. Thus the extra budget mattered, but did not reverse the aggregate result. Final training losses are small and validation often peaks much earlier, consistent with overfitting rather than a model unable to fit the objective. These are fixed existing optimizer settings, not tuned upper bounds for the new modules. More updates alone are not the clearest next lever.

There is one concrete direct Purple gain: fused seed 10 selects CREAMSICLE/JOURNEYMAN/KISSCAM/RUSHMORE in the rock-band-prefix puzzle that all three controls miss. It does not repeat in the other seeds. Overall Purple recovery falls from 106 to 70/78/88 and direct selections from 59 to 46/45/44. All sixteen intended readings are available in the four coverage-audited cases, yet location and slush groups remain unsolved in every arm. Availability is not reliable use of the intended mechanism, and this experiment does not justify promoting the features.

Twenty targeted regression tests pass: lexical extraction/cache layout, absolute-offset matching across lengths, padding masks, symmetric edges, transformation-slot and word permutation invariance, end-to-end edge and fusion/shared-node gradients, checkpoint reload, control equivalence and existing diagnostics. All twelve saved checkpoints ran through the normal validation harness; an independent audit of 1,452 board/model evaluations reproduces every full-board Hit@1. No notebooks or test-set evaluation changed.


## Warm model-only latency

100 warm CUDA-event forwards, batch1, all1820quartets, ten validation boards; excludes transfer, features and harness

| Variant | Median milliseconds |
|---|---:|
| control | 0.465 |
| edges-text | 0.858 |
| edges-fused | 0.990 |
| edges-constant | 0.822 |

## Artifacts

Full metrics, parameter counts, checkpoint paths and timing are in `experiments/gine-transformations/summary.json`. Per-run JSON retains every validation observation and loss interval. Normal harness JSON retains all guesses and feedback. `probability-audit.json` independently recomputes full-board ranks for every selected checkpoint. Interactive loss/validation curves are in `learning-curves.html`.

Runner: `nix develop -c uv run python scripts/run_gine_transformations.py`. Audit: `nix develop -c uv run python scripts/audit_gine_probability_floor.py --results experiments/gine-transformations`. Summary: `nix develop -c uv run python scripts/summarize_gine_transformations.py`.
