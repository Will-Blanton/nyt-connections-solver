# Shared concept and letter-edit witnesses

Completed 9/9 runs. Validation only; no test evaluation.

Three matched arms: original core edges, added shared concept, or added exact edit targets. No Wiktionary, phonetic, or semantic-edit additions. Every arm has the same 26-channel parameter shape, zero-initialized added columns, matched fresh base parameters, schedule and batches. Unused channels are masked. Frozen Qwen vectors; all active projections, LayerNorm, GINE and readout trainable. No learned reference parameters are loaded. Candidate-specific witness values override only their corresponding channels.

Budget 30,000 updates per run, balanced s2/s3/s4, batch 16, evaluation every 500, no pruning/early stopping/scheduler. Checkpoint selection by validation s4 Hit@1, MRR for ties. Existing probability-floor loss (.1/.1) and coherence scale 0 held constant. Fixed tuned architecture/AdamW settings; no retuning. The ongoing CV sweep shares GPU resources, so elapsed times are not benchmark comparisons.

| Arm | Runs | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 | Solves | Direct Purple | Forced Purple |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 3 | 0.3388 | 0.4377 | 0.5096 | 0.3595 | 0.4187 | 101/363 | 65 | 48 |
| concept | 3 | 0.3719 | 0.4760 | 0.5510 | 0.3898 | 0.4348 | 87/363 | 52 | 40 |
| edit_targets | 3 | 0.3581 | 0.4582 | 0.5344 | 0.3829 | 0.4467 | 99/363 | 61 | 47 |

Repeated seeds reuse the same 121 validation puzzles; appearances are correlated. Mechanism annotations are model-reviewed diagnostic labels, not training inputs. Full histories, group ranks, checkpoint paths, and training curves are retained in the per-run JSON files. Partial-run aggregates are not final comparisons.
