# Why a trainable GINE projection underperformed

All experiments use the same frozen 1024-dimensional Qwen word embeddings, 15 edge channels, 968 training/121 validation puzzles, probability-mass objective, four matched seeds (0–3), batch/state sampling, LR 1e-4 for GINE/readout, weight decay 1e-6, up to 3000 updates, and validation s4 Hit@1 then MRR checkpoint selection. Existing test puzzles were not used. The text-only checkpoint supplies the date split and embedding metadata; the scratch and random controls do **not** load any of its model weights.

There is no tensor-shape mismatch. The original standalone path is `1024 → LayerNorm → Linear(1024,256) → ReLU → GINE node Linear(256,32) → GINE updates → pooled readout`. The GINE node projection is trainable in every run. Unfreezing the first Linear adds 262,400 trainable parameters; training its LayerNorm adds another 2,048. The frozen transferred projection's shape is `(256,1024)`, and it is exactly unchanged in its saved standalone checkpoints. When trained at LR 1e-4 it changed by 28–41% of its original weight norm at the selected checkpoints; LR 1e-5 limited the change to 6–8% but did not close the gap.

## Validation ranking

Means across four selected checkpoints per variant. These are oracle s3/s2 states, not states reached by the feedback harness.

| Projection setup | Trainable parameters | s4 Hit@1 | s4 MRR | s3 Hit@1 | s2 Hit@1 |
|---|---:|---:|---:|---:|---:|
| Transferred 1024→256, frozen | 21,859 | 28.51% | 0.3760 | 29.55% | 36.95% |
| **Random 1024→256, frozen** | 21,859 | **26.03%** | 0.3536 | 29.29% | 37.71% |
| Transferred 1024→256, trainable at LR 1e-5 | 284,259 | 21.90% | 0.3162 | 24.64% | 32.71% |
| Random 1024→256, trainable; frozen input norm | 284,259 | 20.87% | 0.2896 | 22.06% | 28.96% |
| Random 1024→256, trainable; trainable input norm | 286,307 | 22.52% | 0.2944 | 23.50% | 28.31% |
| Random 1024→512, trainable; trainable input norm | 556,899 | 16.74% | 0.2420 | 18.49% | 25.10% |
| Random 1024→128, trainable; trainable input norm | 151,011 | 20.87% | 0.2837 | 21.80% | 27.79% |
| Random 1024→64, trainable; trainable input norm | 83,363 | 22.31% | 0.3193 | 24.38% | 31.23% |
| Direct normalized 1024→32 GINE node projection | 46,435 | 17.15% | 0.2544 | 19.16% | 27.51% |

The exact random-initialization pair is the strongest control: the frozen and trainable 1024→256 variants start from identical weights, keep the input norm frozen, and differ only in whether the 1024→256 Linear is optimized. At update 1500, mean mixed-state training loss is 2.892 frozen versus 2.384 trainable, while mean validation s4 Hit@1 is 21.07% versus 16.74%. After selected-checkpoint restoration, the frozen version still leads 26.03% versus 20.87%. This is consistent with the extra capacity fitting the small training set at the expense of validation; it does not prove whether the mechanism is memorization, excessive representation drift, or optimizer dynamics.

The transferred frozen projection beats the random frozen one by only 2.48 s4 Hit@1 points on this validation split. Thus much of its advantage over the trainable variants comes from **holding a wide projection fixed**, not specifically from copying a learned projection. The transferred input norm is also part of that comparison, so the 2.48-point difference cannot be attributed to projection weights alone. Both the transferred and random frozen models use a pretrained embedding model; only the former copies supervised text-only-model weights.

The dimension question is not settled by the direct 1024→32 run: it also removes the intermediate ReLU and changes parameter count and depth. The 64/128/256/512 scratch sweep is more informative; none matched the frozen random control, and performance was not monotonic in width. Width 32 also had large seed variation (s4 Hit@1 6.61–27.27%).

## Larger projection follow-up

The 512-width run changed only the scratch projection width relative to the 256-width variant; GINE still uses 32 hidden units. Across matched seeds, s4 Hit@1 was 19.83%, 18.18%, 22.31%, and 6.61% at width 512, versus 22.31%, 22.31%, 23.97%, and 21.49% at width 256. The 512 model has 556,899 trainable parameters, about twice the 256 model's 286,307. Its mean s4 Hit@1 dropped from 22.52% to 16.74%; s4 MRR dropped from 0.2944 to 0.2420. This does not support moving to width 1024 under the same training setup.

Both widths were replayed on the same 121 validation puzzles with four mistakes, beam width 20, and `score_mode=full_board`:

| Seed | 256 solves | 512 solves | 256 first-guess accuracy | 512 first-guess accuracy |
|---:|---:|---:|---:|---:|
| 0 | 14 | 10 | 25.62% | 19.01% |
| 1 | 14 | 14 | 25.62% | 19.01% |
| 2 | 8 | 10 | 22.31% | 23.14% |
| 3 | 9 | 6 | 23.14% | 6.61% |
| Mean / 121 | 11.25 (9.30%) | 10.0 (8.26%) | 24.17% | 16.94% |

The solve difference is modest and seed-dependent; the ranking and first-guess differences are clearer. These are repeated measurements on the same validation puzzles. Harness histories are in `experiments/harness-scratch-{256,512}-{seed}-val.json`.

For a clean end-to-end model, the next useful experiment is a small standalone scorer with a **trainable, explicitly regularized** projection, evaluated against the random-frozen control. Candidates include stronger weight decay or dropout on the projection and earlier stopping; each changes a different factor. The current diagnostic variants reuse `StandaloneCandidateGINE` as an experiment scaffold, so its checkpoints contain unused inherited attention modules. A production clean model should remove those modules. None of these results justifies silently keeping transferred frozen projection weights as the default architecture.

Training logs and JSON summaries are under `experiments/graph-features/standalone-gine-*.log` and `*standalone-gine-*.json`; checkpoints are under `weights/experiments/`. The previous trainable-projection/harness result is in [standalone-gine-trainable-projection.md](standalone-gine-trainable-projection.md). Harness replay was run for the scratch 256/512 comparison, but not the other scratch and random controls.
