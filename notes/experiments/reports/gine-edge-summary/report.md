# Shared concept and letter-edit witnesses

Completed 9/9 runs. Validation only; no test evaluation.

Three matched arms: original core edges, added shared concept, or added exact edit targets. No Wiktionary, phonetic, or semantic-edit additions. Every arm has the same 26-channel parameter shape, zero-initialized added columns, matched fresh base parameters, schedule and batches. Unused channels are masked. Frozen Qwen vectors; all active projections, LayerNorm, GINE and readout trainable. No learned reference parameters are loaded. Candidate-specific witness values override only their corresponding channels.

Budget 30,000 updates per run, balanced s2/s3/s4, batch 16, evaluation every 500, no pruning/early stopping/scheduler. Checkpoint selection by validation s4 Hit@1, MRR for ties. Existing probability-floor loss (.1/.1) and coherence scale 0 held constant. Fixed tuned architecture/AdamW settings; no retuning. The ongoing CV sweep shares GPU resources, so elapsed times are not benchmark comparisons.

| Arm | Runs | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 | Solves | Direct Purple | Forced Purple |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 3 | 0.3554 | 0.4612 | 0.5399 | 0.3705 | 0.4385 | 94/363 | 73 | 38 |
| concept | 3 | 0.3829 | 0.4872 | 0.5730 | 0.3981 | 0.4417 | 97/363 | 67 | 43 |
| edit_targets | 3 | 0.3581 | 0.4625 | 0.5427 | 0.3781 | 0.4408 | 81/363 | 68 | 34 |

Repeated seeds reuse the same 121 validation puzzles; appearances are correlated. Mechanism annotations are model-reviewed diagnostic labels, not training inputs. Full histories, group ranks, checkpoint paths, and training curves are retained in the per-run JSON files. Partial-run aggregates are not final comparisons.

Direct edge-summary branch: width 16; per-channel mean, minimum, and weakest-word average on raw candidate edges after exact-witness overrides. Small two-linear-layer MLP, concatenated into the GINE readout; added readout columns initialize zero. No edge amplification or gate. See edge-summary-comparison.md for the matched regular-GINE comparison.
