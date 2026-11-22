# Direct edge-summary channel with regular candidate GINE

Validation only; three matched seeds (10, 11, 12), 30,000 updates per run, balanced s2/s3/s4, batch 16, evaluation every 500 updates. Same 121 validation puzzles repeated across seeds: 363 correlated appearances, not 363 independent puzzles. Existing completed regular-GINE runs are reused as controls. No test evaluation or notebook changes.

The new branch computes three values per relation channel from each candidate: off-diagonal mean, off-diagonal minimum, and minimum over words of mean incident edge support. Raw masked candidate edges include exact shared-witness overrides. A 78 -> 16 -> 16 ReLU MLP embeds these statistics, then concatenates them with the GINE node mean/max embedding before the existing readout MLP. Added readout columns initialize zero; existing parameters and initial scores match the control. The branch adds 3,584 trainable parameters. Coherence strength is zero in all runs. No new gate, loss, scheduler, or hyperparameter tuning.

Both models train all active projections, LayerNorm, GINE, and readout from scratch; only original cached Qwen vectors are frozen. New branch parameters are also trainable. Checkpoint selection is validation s4 Hit@1, MRR for ties. All runs receive their full budget. Harness: beam 20, original full-board score sums, four mistakes, shuffle seed 0.

| Features | Architecture | s4 Hit@1 | MRR | Hit@4 | s3 Hit@1 | s2 Hit@1 | First guess | Solves | Direct Purple | Forced Purple |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | regular | 33.88% | 0.4377 | 50.96% | 35.95% | 41.87% | 39.12% | 101/363 | 65 | 48 |
| control | edge_summary | 35.54% | 0.4612 | 53.99% | 37.05% | 43.85% | 37.47% | 94/363 | 73 | 38 |
| concept | regular | 37.19% | 0.4760 | 55.10% | 38.98% | 43.48% | 39.39% | 87/363 | 52 | 40 |
| concept | edge_summary | 38.29% | 0.4872 | 57.30% | 39.81% | 44.17% | 41.87% | 97/363 | 67 | 43 |
| edit_targets | regular | 35.81% | 0.4582 | 53.44% | 38.29% | 44.67% | 38.02% | 99/363 | 61 | 47 |
| edit_targets | edge_summary | 35.81% | 0.4625 | 54.27% | 37.81% | 44.08% | 35.81% | 81/363 | 68 | 34 |

## Paired effects of adding the summary branch

| Features | Hit@1 difference | Solve difference | Solves gained | Solves lost |
|---|---:|---:|---:|---:|
| control | +1.65 pp | -7 | 18 | 25 |
| concept | +1.10 pp | +10 | 31 | 21 |
| edit_targets | +0.00 pp | -18 | 15 | 33 |

## Per-seed results and learning

| Seed | Features | Architecture | Hit@1 | Solves | Best update | Final Hit@1 |
|---|---|---|---:|---:|---:|---:|
| 10 | control | regular | 33.06% | 32 | 8000 | 19.83% |
| 10 | control | edge_summary | 33.06% | 34 | 3000 | 25.62% |
| 10 | concept | regular | 33.88% | 33 | 5000 | 20.66% |
| 10 | concept | edge_summary | 38.02% | 27 | 14000 | 28.93% |
| 10 | edit_targets | regular | 34.71% | 35 | 9000 | 14.88% |
| 10 | edit_targets | edge_summary | 34.71% | 24 | 18000 | 22.31% |
| 11 | control | regular | 34.71% | 32 | 7500 | 17.36% |
| 11 | control | edge_summary | 38.84% | 28 | 7000 | 14.05% |
| 11 | concept | regular | 41.32% | 28 | 2000 | 22.31% |
| 11 | concept | edge_summary | 37.19% | 34 | 4000 | 14.88% |
| 11 | edit_targets | regular | 37.19% | 33 | 2500 | 18.18% |
| 11 | edit_targets | edge_summary | 38.02% | 31 | 2000 | 17.36% |
| 12 | control | regular | 33.88% | 37 | 2500 | 19.01% |
| 12 | control | edge_summary | 34.71% | 32 | 4000 | 19.01% |
| 12 | concept | regular | 36.36% | 26 | 1000 | 19.83% |
| 12 | concept | edge_summary | 39.67% | 36 | 4000 | 23.14% |
| 12 | edit_targets | regular | 35.54% | 31 | 1500 | 23.97% |
| 12 | edit_targets | edge_summary | 34.71% | 26 | 2000 | 22.31% |

## Mechanism diagnostics

| Mechanism | Features | Architecture | Top4 | Direct | Forced |
|---|---|---|---:|---:|---:|
| anagram | control | regular | 66.67% | 8 | 0 |
| letter_edit | control | regular | 0.00% | 7 | 1 |
| named_entity_or_cultural | control | regular | 4.76% | 37 | 14 |
| orthographic_pattern | control | regular | 50.00% | 9 | 0 |
| phrase_completion | control | regular | 2.45% | 38 | 28 |
| polysemy | control | regular | 0.00% | 0 | 3 |
| semantic_category | control | regular | 12.57% | 186 | 23 |
| sound | control | regular | 0.00% | 5 | 4 |
| synonym | control | regular | 30.00% | 209 | 15 |
| word_containment | control | regular | 0.00% | 8 | 13 |
| anagram | concept | regular | 66.67% | 6 | 0 |
| letter_edit | concept | regular | 3.03% | 3 | 2 |
| named_entity_or_cultural | concept | regular | 6.35% | 32 | 12 |
| orthographic_pattern | concept | regular | 33.33% | 6 | 0 |
| phrase_completion | concept | regular | 1.96% | 27 | 28 |
| polysemy | concept | regular | 0.00% | 0 | 3 |
| semantic_category | concept | regular | 15.57% | 198 | 16 |
| sound | concept | regular | 0.00% | 8 | 2 |
| synonym | concept | regular | 29.74% | 193 | 14 |
| word_containment | concept | regular | 0.00% | 5 | 10 |
| anagram | edit_targets | regular | 66.67% | 6 | 0 |
| letter_edit | edit_targets | regular | 0.00% | 6 | 0 |
| named_entity_or_cultural | edit_targets | regular | 4.23% | 34 | 14 |
| orthographic_pattern | edit_targets | regular | 58.33% | 9 | 0 |
| phrase_completion | edit_targets | regular | 2.45% | 40 | 23 |
| polysemy | edit_targets | regular | 0.00% | 0 | 3 |
| semantic_category | edit_targets | regular | 12.57% | 187 | 23 |
| sound | edit_targets | regular | 0.00% | 2 | 6 |
| synonym | edit_targets | regular | 31.03% | 207 | 16 |
| word_containment | edit_targets | regular | 0.00% | 5 | 14 |
| anagram | control | edge_summary | 66.67% | 9 | 0 |
| letter_edit | control | edge_summary | 0.00% | 7 | 1 |
| named_entity_or_cultural | control | edge_summary | 5.29% | 33 | 13 |
| orthographic_pattern | control | edge_summary | 50.00% | 8 | 0 |
| phrase_completion | control | edge_summary | 3.92% | 43 | 20 |
| polysemy | control | edge_summary | 0.00% | 1 | 2 |
| semantic_category | control | edge_summary | 11.78% | 180 | 24 |
| sound | control | edge_summary | 0.00% | 6 | 1 |
| synonym | control | edge_summary | 31.79% | 206 | 17 |
| word_containment | control | edge_summary | 0.00% | 6 | 16 |
| anagram | concept | edge_summary | 66.67% | 9 | 0 |
| letter_edit | concept | edge_summary | 0.00% | 5 | 0 |
| named_entity_or_cultural | concept | edge_summary | 5.82% | 30 | 15 |
| orthographic_pattern | concept | edge_summary | 58.33% | 9 | 0 |
| phrase_completion | concept | edge_summary | 6.86% | 33 | 29 |
| polysemy | concept | edge_summary | 0.00% | 0 | 2 |
| semantic_category | concept | edge_summary | 16.17% | 202 | 20 |
| sound | concept | edge_summary | 0.00% | 6 | 2 |
| synonym | concept | edge_summary | 29.74% | 199 | 15 |
| word_containment | concept | edge_summary | 0.00% | 8 | 14 |
| anagram | edit_targets | edge_summary | 66.67% | 7 | 0 |
| letter_edit | edit_targets | edge_summary | 0.00% | 5 | 0 |
| named_entity_or_cultural | edit_targets | edge_summary | 6.35% | 31 | 10 |
| orthographic_pattern | edit_targets | edge_summary | 66.67% | 8 | 1 |
| phrase_completion | edit_targets | edge_summary | 5.39% | 42 | 16 |
| polysemy | edit_targets | edge_summary | 0.00% | 0 | 2 |
| semantic_category | edit_targets | edge_summary | 13.17% | 168 | 26 |
| sound | edit_targets | edge_summary | 0.00% | 3 | 3 |
| synonym | edit_targets | edge_summary | 28.72% | 202 | 13 |
| word_containment | edit_targets | edge_summary | 0.00% | 8 | 10 |

## Intended exact edit target

wind/wine/wing/wink is a Green group, not Purple. Exact shared-edit targets have only one positive training quartet and none for deletion channels, so supervision remains sparse.

| Architecture | Features | Seed | Rank | Direct | Forced |
|---|---|---:|---:|---|---|
| regular | control | 10 | 1693 | False | False |
| regular | concept | 10 | 1558 | False | False |
| regular | edit_targets | 10 | 1820 | False | False |
| regular | control | 11 | 1819 | False | False |
| regular | concept | 11 | 1820 | False | False |
| regular | edit_targets | 11 | 1820 | False | False |
| regular | control | 12 | 1808 | False | False |
| regular | concept | 12 | 631 | False | False |
| regular | edit_targets | 12 | 1820 | False | False |
| edge_summary | control | 10 | 669 | False | False |
| edge_summary | concept | 10 | 1210 | False | False |
| edge_summary | edit_targets | 10 | 1820 | False | False |
| edge_summary | control | 11 | 1156 | False | False |
| edge_summary | concept | 11 | 1396 | False | False |
| edge_summary | edit_targets | 11 | 1820 | False | False |
| edge_summary | control | 12 | 948 | False | False |
| edge_summary | concept | 12 | 1571 | False | False |
| edge_summary | edit_targets | 12 | 1820 | False | False |

## Conclusions

- control: the branch changes mean full-board Hit@1 by +1.65 percentage points and full solves by -7 appearances.
- concept: the branch changes mean full-board Hit@1 by +1.10 percentage points and full solves by +10 appearances.
- edit_targets: the branch changes mean full-board Hit@1 by +0.00 percentage points and full solves by -18 appearances.

The largest observed solve count is regular/control: 101/363. This is descriptive; neither the repeated seeds nor this repeatedly inspected validation set provide an independent confirmation of superiority. Keep the summary branch opt-in pending stronger evidence.

## Verification and interpretation

All saved checkpoints strictly reload and independently reproduce training Hit@1 before harness evaluation. Independent production-loader audits pass for both sets of checkpoints. Feature metadata and manifests match except the added summary branch; model configurations match except edge_summary_width. Focused tests verify diagonal exclusion, permutation invariance, candidate locality, witness overrides, inactive-channel masking, matched initial scores, gradients after the zero initialization, and strict reload.

9/9 trained summary branches have nonzero added readout weights and activations that vary across candidates within validation boards. This checks that trained, varying branch representations reach the readout; it does not establish that they help. See checkpoint-audit.json for the measured norms and activation variation. 43 focused tests pass, including three new branch tests.

This is a fixed-hyperparameter ablation, not separate architecture tuning. Mechanism labels are diagnostic annotations, not training inputs. Reduced-state metrics are oracle states, not actual feedback trajectories. Repeated validation reuse limits strong conclusions. Ranking improvements alone do not establish a puzzle solve-rate improvement. The separate five-fold tuning experiment is unchanged.
