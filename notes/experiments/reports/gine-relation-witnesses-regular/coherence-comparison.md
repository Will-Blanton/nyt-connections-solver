# Regular versus coherence GINE with new relation features

Validation only. Same 121 puzzles reused across three matched seeds; 363 appearances are correlated. Regular GINE uses coherence strength 0; earlier runs use strength 1. All other model configurations, fresh initial parameters, minibatch seeds, splits, 30,000-update budgets and checkpoint selection rules match. Each of the three feature arms has the same 26-channel parameter shape; unused channels are masked. Only original Qwen vectors are frozen. No hyperparameter retuning, scheduler or auxiliary loss changes. Checkpoint selection uses s4 Hit@1, then MRR. Harness: beam 20, original full-board score sums, four mistakes, shuffle seed 0.

| Features | Architecture | s4 Hit@1 | MRR | Hit@4 | s3 Hit@1 | s2 Hit@1 | First guess | Solves | Direct Purple | Forced Purple |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | regular | 33.88% | 0.4377 | 50.96% | 35.95% | 41.87% | 39.12% | 101/363 | 65 | 48 |
| control | coherence | 34.99% | 0.4532 | 54.27% | 37.88% | 45.18% | 36.91% | 92/363 | 76 | 32 |
| concept | regular | 37.19% | 0.4760 | 55.10% | 38.98% | 43.48% | 39.39% | 87/363 | 52 | 40 |
| concept | coherence | 40.22% | 0.5066 | 58.68% | 42.15% | 46.97% | 42.15% | 94/363 | 58 | 48 |
| edit_targets | regular | 35.81% | 0.4582 | 53.44% | 38.29% | 44.67% | 38.02% | 99/363 | 61 | 47 |
| edit_targets | coherence | 35.81% | 0.4582 | 53.44% | 38.43% | 44.81% | 39.67% | 90/363 | 72 | 35 |

## Matched coherence effects

| Features | s4 Hit@1 difference | Solve difference | Solves gained | Solves lost |
|---|---:|---:|---:|---:|
| control | +1.10 pp | -9 | 18 | 27 |
| concept | +3.03 pp | +7 | 30 | 23 |
| edit_targets | +0.00 pp | -9 | 21 | 30 |

Differences above are coherence minus regular. Feature effects must also be compared against the control with the same architecture:

| Architecture | Added feature | s4 Hit@1 difference from control | Solve difference from control |
|---|---|---:|---:|
| regular | concept | +3.31 pp | -14 |
| regular | edit_targets | +1.93 pp | -2 |
| coherence | concept | +5.23 pp | +2 |
| coherence | edit_targets | +0.83 pp | -2 |

## Per-seed results and training

| Seed | Features | Architecture | s4 Hit@1 | Solves | Best update | Final s4 Hit@1 |
|---|---|---|---:|---:|---:|---:|
| 10 | control | regular | 33.06% | 32 | 8000 | 19.83% |
| 10 | control | coherence | 33.06% | 30 | 7000 | 19.01% |
| 10 | concept | regular | 33.88% | 33 | 5000 | 20.66% |
| 10 | concept | coherence | 41.32% | 28 | 8500 | 24.79% |
| 10 | edit_targets | regular | 34.71% | 35 | 9000 | 14.88% |
| 10 | edit_targets | coherence | 35.54% | 32 | 5000 | 18.18% |
| 11 | control | regular | 34.71% | 32 | 7500 | 17.36% |
| 11 | control | coherence | 36.36% | 33 | 3000 | 17.36% |
| 11 | concept | regular | 41.32% | 28 | 2000 | 22.31% |
| 11 | concept | coherence | 40.50% | 36 | 1000 | 28.10% |
| 11 | edit_targets | regular | 37.19% | 33 | 2500 | 18.18% |
| 11 | edit_targets | coherence | 37.19% | 30 | 3000 | 20.66% |
| 12 | control | regular | 33.88% | 37 | 2500 | 19.01% |
| 12 | control | coherence | 35.54% | 29 | 4000 | 22.31% |
| 12 | concept | regular | 36.36% | 26 | 1000 | 19.83% |
| 12 | concept | coherence | 38.84% | 30 | 3000 | 16.53% |
| 12 | edit_targets | regular | 35.54% | 31 | 1500 | 23.97% |
| 12 | edit_targets | coherence | 34.71% | 28 | 6500 | 21.49% |

## Mechanism-specific retrieval and recovery

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
| anagram | control | coherence | 66.67% | 8 | 0 |
| letter_edit | control | coherence | 0.00% | 4 | 0 |
| named_entity_or_cultural | control | coherence | 5.82% | 32 | 12 |
| orthographic_pattern | control | coherence | 50.00% | 9 | 0 |
| phrase_completion | control | coherence | 4.90% | 39 | 20 |
| polysemy | control | coherence | 0.00% | 1 | 2 |
| semantic_category | control | coherence | 12.57% | 178 | 26 |
| sound | control | coherence | 0.00% | 3 | 1 |
| synonym | control | coherence | 30.00% | 197 | 19 |
| word_containment | control | coherence | 0.00% | 7 | 12 |
| anagram | concept | coherence | 66.67% | 9 | 0 |
| letter_edit | concept | coherence | 0.00% | 6 | 1 |
| named_entity_or_cultural | concept | coherence | 5.29% | 31 | 16 |
| orthographic_pattern | concept | coherence | 58.33% | 9 | 0 |
| phrase_completion | concept | coherence | 7.35% | 32 | 31 |
| polysemy | concept | coherence | 0.00% | 0 | 1 |
| semantic_category | concept | coherence | 18.36% | 209 | 21 |
| sound | concept | coherence | 0.00% | 6 | 1 |
| synonym | concept | coherence | 28.97% | 202 | 10 |
| word_containment | concept | coherence | 0.00% | 2 | 13 |
| anagram | edit_targets | coherence | 66.67% | 7 | 0 |
| letter_edit | edit_targets | coherence | 3.03% | 7 | 0 |
| named_entity_or_cultural | edit_targets | coherence | 5.29% | 35 | 14 |
| orthographic_pattern | edit_targets | coherence | 58.33% | 9 | 0 |
| phrase_completion | edit_targets | coherence | 4.90% | 33 | 22 |
| polysemy | edit_targets | coherence | 0.00% | 0 | 1 |
| semantic_category | edit_targets | coherence | 11.58% | 171 | 27 |
| sound | edit_targets | coherence | 0.00% | 4 | 1 |
| synonym | edit_targets | coherence | 31.79% | 200 | 13 |
| word_containment | edit_targets | coherence | 0.00% | 9 | 12 |

## Exact win target

The intended wind/wine/wing/wink target is Green. Exact edit witnesses have only one positive training quartet, and none for the deletion channels. This experiment does not establish the value of broadly supervised transformation features.

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
| coherence | control | 10 | 1820 | False | False |
| coherence | concept | 10 | 1820 | False | False |
| coherence | edit_targets | 10 | 1820 | False | False |
| coherence | control | 11 | 1820 | False | False |
| coherence | concept | 11 | 1671 | False | False |
| coherence | edit_targets | 11 | 1820 | False | False |
| coherence | control | 12 | 1057 | False | False |
| coherence | concept | 12 | 966 | False | False |
| coherence | edit_targets | 12 | 1820 | False | False |

## Conclusions

Coherence is not a consistent solve-rate improvement at these fixed settings. The interaction depends on the feature arm:

- control: coherence changes mean full-board Hit@1 by +1.10 percentage points and complete solves by -9 appearances.
- concept: coherence changes mean full-board Hit@1 by +3.03 percentage points and complete solves by +7 appearances.
- edit_targets: coherence changes mean full-board Hit@1 by +0.00 percentage points and complete solves by -9 appearances.

The largest solve count in this comparison is regular/control: 101/363. This is descriptive, not an independent confirmation of superiority. Seed variation and repeated validation reuse limit strong conclusions. Adding a feature may improve ranking without improving full-partition decisions; the control comparisons above are necessary before promotion.

All runs finish their full budget. The learning curves show late overfitting; longer training does not eliminate the ranking-versus-solve discrepancy. This ablation does not replace the ongoing separate five-fold tuning comparison.

## Verification and limits

All 18 runs strictly reload parameters and independently reproduce training s4 Hit@1 before harness evaluation. Manifests and model configurations match except coherence strength. A focused regression test verifies identical initial parameters across coherence strengths. Separate checkpoint-audit.json files validate reconstruction through the production loader. This is a fixed-hyperparameter ablation, not a comparison of separately tuned architectures. Repeated reuse of this validation set makes the results exploratory. The ongoing five-fold tuning is separate and unchanged. No test evaluation or notebook changes.
