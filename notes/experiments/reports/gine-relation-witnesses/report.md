# Shared concept and letter-edit witnesses

Completed 9/9 runs. Validation only; no test evaluation.

Three matched arms: original core edges, added shared concept, or added exact edit targets. No Wiktionary, phonetic, or semantic-edit additions. Every arm has the same 26-channel parameter shape, zero-initialized added columns, matched fresh base parameters, schedule and batches. Unused channels are masked. Frozen Qwen vectors; all active projections, LayerNorm, GINE and readout trainable. No learned reference parameters are loaded. Candidate-specific witness values override only their corresponding channels.

Budget 30,000 updates per run, balanced s2/s3/s4, batch 16, evaluation every 500, no pruning/early stopping/scheduler. Checkpoint selection by validation s4 Hit@1, MRR for ties. Existing probability-floor loss (.1/.1) and coherence scale 1 held constant. Fixed tuned architecture/AdamW settings; no retuning. The ongoing CV sweep shares GPU resources, so elapsed times are not benchmark comparisons.

| Arm | Runs | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 | Solves | Direct Purple | Forced Purple |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 3 | 0.3499 | 0.4532 | 0.5427 | 0.3788 | 0.4518 | 92/363 | 76 | 32 |
| concept | 3 | 0.4022 | 0.5066 | 0.5868 | 0.4215 | 0.4697 | 94/363 | 58 | 48 |
| edit_targets | 3 | 0.3581 | 0.4582 | 0.5344 | 0.3843 | 0.4481 | 90/363 | 72 | 35 |

Repeated seeds reuse the same 121 validation puzzles; appearances are correlated. Mechanism annotations are model-reviewed diagnostic labels, not training inputs. Full histories, group ranks, checkpoint paths, and training curves are retained in the per-run JSON files. Partial-run aggregates are not final comparisons.

## Standalone signal audit

| Score | Train expected Hit@1 | Validation expected Hit@1 |
|---|---:|---:|
| shared_wordnet_concept_specificity | 35.09% | 23.21% |
| existing_rare_concept_mean | 21.81% | 14.88% |
| existing_rare_concept_minimum | 21.45% | 15.71% |
| edit_target_family_max | 0.22% | 1.04% |

Standalone scores rank all 1,820 quartets without model training; ties at the maximum receive uniform tie-breaking credit. The exact edit family takes the maximum over its ten fixed channels. These are candidate retrieval metrics, not complete-puzzle solve rates.

Exact edit witnesses support only one training gold group (bough/cough/dough/tough) and one validation gold group (wind/wine/wing/wink). A second nonzero channel on the same quartet does not constitute another covered group. In particular, the exact shared deletion target has no positive training quartet: the last-position channel fires on one incorrect training quartet, and the anywhere channel on six. The training objective therefore supplies only negative examples for these deletion channels. The ten operation/alignment channels also keep that supervision separate from the lone positive substitution example. Sparse supervision limits what this experiment can say about learning that new relation.

## Mechanism-specific results

| Mechanism | Arm | Group appearances | Top4 | Direct | Forced |
|---|---|---:|---:|---:|---:|
| anagram | control | 9 | 66.67% | 8 | 0 |
| letter_edit | control | 33 | 0.00% | 4 | 0 |
| named_entity_or_cultural | control | 189 | 5.82% | 32 | 12 |
| orthographic_pattern | control | 12 | 50.00% | 9 | 0 |
| phrase_completion | control | 204 | 4.90% | 39 | 20 |
| polysemy | control | 15 | 0.00% | 1 | 2 |
| semantic_category | control | 501 | 12.57% | 178 | 26 |
| sound | control | 48 | 0.00% | 3 | 1 |
| synonym | control | 390 | 30.00% | 197 | 19 |
| word_containment | control | 51 | 0.00% | 7 | 12 |
| anagram | concept | 9 | 66.67% | 9 | 0 |
| letter_edit | concept | 33 | 0.00% | 6 | 1 |
| named_entity_or_cultural | concept | 189 | 5.29% | 31 | 16 |
| orthographic_pattern | concept | 12 | 58.33% | 9 | 0 |
| phrase_completion | concept | 204 | 7.35% | 32 | 31 |
| polysemy | concept | 15 | 0.00% | 0 | 1 |
| semantic_category | concept | 501 | 18.36% | 209 | 21 |
| sound | concept | 48 | 0.00% | 6 | 1 |
| synonym | concept | 390 | 28.97% | 202 | 10 |
| word_containment | concept | 51 | 0.00% | 2 | 13 |
| anagram | edit_targets | 9 | 66.67% | 7 | 0 |
| letter_edit | edit_targets | 33 | 3.03% | 7 | 0 |
| named_entity_or_cultural | edit_targets | 189 | 5.29% | 35 | 14 |
| orthographic_pattern | edit_targets | 12 | 58.33% | 9 | 0 |
| phrase_completion | edit_targets | 204 | 4.90% | 33 | 22 |
| polysemy | edit_targets | 15 | 0.00% | 0 | 1 |
| semantic_category | edit_targets | 501 | 11.58% | 171 | 27 |
| sound | edit_targets | 48 | 0.00% | 4 | 1 |
| synonym | edit_targets | 390 | 31.79% | 200 | 13 |
| word_containment | edit_targets | 51 | 0.00% | 9 | 12 |

## Per-run checkpoints and learning

| Seed | Arm | s4 Hit@1 | Solves | Best update | Final s4 Hit@1 |
|---|---|---:|---:|---:|---:|
| 10 | control | 0.3306 | 30/121 | 7000 | 0.1901 |
| 10 | concept | 0.4132 | 28/121 | 8500 | 0.2479 |
| 10 | edit_targets | 0.3554 | 32/121 | 5000 | 0.1818 |
| 11 | control | 0.3636 | 33/121 | 3000 | 0.1736 |
| 11 | concept | 0.4050 | 36/121 | 1000 | 0.2810 |
| 11 | edit_targets | 0.3719 | 30/121 | 3000 | 0.2066 |
| 12 | control | 0.3554 | 29/121 | 4000 | 0.2231 |
| 12 | concept | 0.3884 | 30/121 | 3000 | 0.1653 |
| 12 | edit_targets | 0.3471 | 28/121 | 6500 | 0.2149 |

## Win + letter target

| Arm | Seed | Rank | Directly recovered | Forced |
|---|---:|---:|---|---|
| control | 10 | 1820 | False | False |
| concept | 10 | 1820 | False | False |
| edit_targets | 10 | 1820 | False | False |
| control | 11 | 1820 | False | False |
| concept | 11 | 1671 | False | False |
| edit_targets | 11 | 1820 | False | False |
| control | 12 | 1057 | False | False |
| concept | 12 | 966 | False | False |
| edit_targets | 12 | 1820 | False | False |

The direct/forced distinction is based on actual guess histories, not recovered group counts alone. All reported group ranks are independently recomputed after strict state-dict reload; full-board Hit@1 must reproduce training validation. No test data or notebook execution. See learning-curves.html, group_results.csv, mechanism_results.csv and summary.json for retained evidence.

## Conclusions

| Arm | Actual harness first-guess accuracy | Solve rate |
|---|---:|---:|
| control | 36.91% | 25.34% |
| concept | 42.15% | 25.90% |
| edit_targets | 39.67% | 24.79% |

Shared concepts improve mean full-board Hit@1 by 5.23 percentage points, with improvements in all three matched seeds. Reduced-state Hit@1 and MRR also improve. Actual harness solves change by only +2: 31 gained and 29 lost appearances. This supports a useful ranking feature, not a demonstrated reliable solve-rate gain.

Direct Purple recoveries change from 76 to 58; forced Purple recoveries from 32 to 48. The added concept signal primarily improves semantic categories; it does not resolve the wordplay bottleneck.

Exact edit targets change full-board Hit@1 by +0.83 points and solves by -2. The intended win quartet ranks last in all three edit-target runs and is never recovered. There is no demonstrated targeted mechanism benefit. Only one positive training quartet and separate rule channels make this a poor test of learning a broadly useful edit mechanic; do not interpret it as evidence against all transformation features.

Keep both additions opt-in. The concept feature merits further comparison for ranking/first-choice quality; exact shared-target evidence should not be promoted on these results. All nine runs completed 30,000 updates, while selected checkpoints lie between 1,000 and 8,500; later training consistently overfits.

## Verification

Four new relation-witness tests pass, including matched initialization and nonzero gradients for enabled added columns. Full regression suite: 148/156 pass; eight unchanged tests still reference removed notebook classes or the old NDCG API. No notebooks were edited. `git diff --check` passes.

All nine saved checkpoints independently reconstruct through the production loader on CPU using cached frozen Qwen vectors and reproduce their training validation Hit@1 across 1,089 board/model evaluations. See checkpoint-audit.json.