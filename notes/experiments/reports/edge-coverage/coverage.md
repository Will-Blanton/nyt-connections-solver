# Existing edge feature coverage

Full-board validation only: 121 puzzles, four true groups per puzzle. Model misses use four existing long-budget candidate-input GINE seeds. No training or test evaluation.

A miss means the specific true group ranks below the first four candidates. Coverage is counted once per puzzle/group; missed diagnostics count group appearances across model seeds and are correlated. Nonzero support is not proof of a shared explanation.

Mean-evidence percentiles compare each true group's mean of six edge values with all 1,816 incorrect quads, excluding all four true groups. Ties receive half credit; 0.5 is uninformative. Hard-negative percentiles use the four highest model-ranked incorrect quads. These univariate diagnostics cannot establish a multichannel model's feature ceiling.

## Purple coverage and discrimination

Dense semantic channels can be nonzero everywhere. Sparse channels that fire on only one pair may not explain all four words. All-six support also need not have one common witness.

| Channel | Any pair | All six pairs | True vs incorrect percentile | Missed: vs incorrect | Missed: vs hard negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic_cosine | 100.0% | 100.0% | 0.520 | 0.516 | 0.258 |
| sense_jaccard | 1.7% | 0.0% | 0.469 | 0.468 | 0.363 |
| rare_concept_jaccard | 95.0% | 48.8% | 0.448 | 0.452 | 0.292 |
| longest_common_substring | 56.2% | 1.7% | 0.511 | 0.507 | 0.446 |
| common_prefix | 6.6% | 0.0% | 0.492 | 0.492 | 0.478 |
| common_suffix | 15.7% | 0.8% | 0.524 | 0.524 | 0.494 |
| prefix_semantic_similarity | 69.4% | 22.3% | 0.553 | 0.552 | 0.551 |
| suffix_semantic_similarity | 57.9% | 9.9% | 0.467 | 0.463 | 0.440 |
| shared_preceding_word | 18.2% | 1.7% | 0.533 | 0.532 | 0.457 |
| shared_following_word | 24.8% | 3.3% | 0.529 | 0.528 | 0.441 |
| exact_anagram | 0.8% | 0.8% | 0.503 | 0.501 | 0.503 |
| character_multiset_overlap | 100.0% | 42.1% | 0.530 | 0.528 | 0.432 |
| both_palindromes | 0.0% | 0.0% | 0.496 | 0.496 | 0.493 |
| both_initial_double | 0.0% | 0.0% | 0.500 | 0.500 | 0.500 |
| both_consecutive_double_pairs | 0.0% | 0.0% | 0.500 | 0.500 | 0.500 |

## Phrase evidence by color

| Color | Direction | Any pair /121 | All six pairs /121 | Shared completion for all four /121 | All pairs without shared completion |
| --- | --- | ---: | ---: | ---: | ---: |
| yellow | preceding | 18 | 0 | 0 | 0 |
| yellow | following | 35 | 1 | 1 | 0 |
| green | preceding | 27 | 0 | 0 | 0 |
| green | following | 40 | 1 | 1 | 0 |
| blue | preceding | 23 | 0 | 0 | 0 |
| blue | following | 30 | 0 | 0 | 0 |
| purple | preceding | 22 | 2 | 2 | 0 |
| purple | following | 30 | 4 | 4 | 0 |

## Explicit blank-template coverage

Labels containing two or more underscores are a narrow diagnostic subset, not a comprehensive classifier of phrase-based categories. Labels are not model inputs.

| Color | Explicit templates | Common four-word completion available |
| --- | ---: | ---: |
| yellow | 0 | 0 |
| green | 0 | 0 |
| blue | 1 | 0 |
| purple | 39 | 6 |

## Purple examples with a common phrase completion

Ranks are of the true group among all candidates, in seed order 0, 1, 2, 3.

- 2024-01-31: ___ pit; ['barbecue', 'orchestra', 'snake', 'tar']; preceding=[]; following=['pit']; ranks=[203, 159, 36, 226]
- 2024-09-28: ___ control; ['birth', 'cruise', 'quality', 'remote']; preceding=[]; following=['control']; ranks=[416, 543, 465, 675]
- 2024-11-19: double ___; ['agent', 'dribble', 'jeopardy', 'standard']; preceding=['double']; following=[]; ranks=[42, 92, 546, 83]
- 2025-05-13: ___ cleaner; ['bathroom', 'dry', 'pipe', 'vacuum']; preceding=[]; following=['cleaner']; ranks=[98, 46, 40, 108]
- 2025-11-15: ___ zone; ['buffer', 'comfort', 'time', 'twilight']; preceding=[]; following=['zone']; ranks=[786, 456, 557, 114]
- 2026-05-28: ski ___; ['jump', 'lift', 'lodge', 'slope']; preceding=['ski']; following=[]; ranks=[4, 4, 31, 13]

Raw per-group evidence, all color/channel summaries, and per-seed group ranks are stored in `coverage.json`. Category descriptions are inspected only for diagnosis; they are not inputs to feature construction or the trained models.
