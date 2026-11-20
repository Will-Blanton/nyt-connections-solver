# Existing edge feature coverage

Full-board validation only: 121 puzzles, four true groups per puzzle. Model misses use four existing long-budget `gated-gine-projected` seeds. No training or test evaluation.

A miss means the specific true group ranks below the first four candidates. Coverage is counted once per puzzle/group; missed diagnostics count group appearances across model seeds and are correlated. Nonzero support is not proof of a shared explanation.

Mean-evidence percentiles compare each true group's mean of six edge values with all 1,816 incorrect quads, excluding all four true groups. Ties receive half credit; 0.5 is uninformative. Hard-negative percentiles use the four highest model-ranked incorrect quads. These univariate diagnostics cannot establish a multichannel model's feature ceiling.

## Purple coverage and discrimination

Dense semantic channels can be nonzero everywhere. Sparse channels that fire on only one pair may not explain all four words. All-six support also need not have one common witness.

| Channel | Any pair | All six pairs | True vs incorrect percentile | Missed: vs incorrect | Missed: vs hard negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic_cosine | 100.0% | 100.0% | 0.520 | 0.514 | 0.232 |
| sense_jaccard | 1.7% | 0.0% | 0.469 | 0.468 | 0.330 |
| rare_concept_jaccard | 95.0% | 48.8% | 0.448 | 0.452 | 0.250 |
| longest_common_substring | 56.2% | 1.7% | 0.511 | 0.503 | 0.445 |
| common_prefix | 6.6% | 0.0% | 0.492 | 0.491 | 0.480 |
| common_suffix | 15.7% | 0.8% | 0.524 | 0.524 | 0.492 |
| prefix_semantic_similarity | 69.4% | 22.3% | 0.553 | 0.549 | 0.545 |
| suffix_semantic_similarity | 57.9% | 9.9% | 0.467 | 0.464 | 0.437 |
| shared_preceding_word | 18.2% | 1.7% | 0.533 | 0.533 | 0.451 |
| shared_following_word | 24.8% | 3.3% | 0.529 | 0.528 | 0.429 |
| exact_anagram | 0.8% | 0.8% | 0.503 | 0.499 | 0.501 |
| character_multiset_overlap | 100.0% | 42.1% | 0.530 | 0.525 | 0.433 |
| both_palindromes | 0.0% | 0.0% | 0.496 | 0.496 | 0.494 |
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

| Color | Explicit templates | Current phrase index: common completion | Closed compounds in same vocabulary |
| --- | ---: | ---: | ---: |
| yellow | 0 | 0 | 0 |
| green | 0 | 0 | 0 |
| blue | 1 | 0 | 0 |
| purple | 39 | 6 | 6 |

## Purple examples with a common phrase completion

Ranks are of the true group among all candidates, in seed order 0, 1, 2, 3.

- 2024-01-31: ___ pit; ['barbecue', 'orchestra', 'snake', 'tar']; preceding=[]; following=['pit']; ranks=[203, 315, 537, 355]
- 2024-09-28: ___ control; ['birth', 'cruise', 'quality', 'remote']; preceding=[]; following=['control']; ranks=[189, 159, 245, 119]
- 2024-11-19: double ___; ['agent', 'dribble', 'jeopardy', 'standard']; preceding=['double']; following=[]; ranks=[21, 31, 26, 40]
- 2025-05-13: ___ cleaner; ['bathroom', 'dry', 'pipe', 'vacuum']; preceding=[]; following=['cleaner']; ranks=[72, 168, 83, 101]
- 2025-11-15: ___ zone; ['buffer', 'comfort', 'time', 'twilight']; preceding=[]; following=['zone']; ranks=[340, 356, 607, 558]
- 2026-05-28: ski ___; ['jump', 'lift', 'lodge', 'slope']; preceding=['ski']; following=[]; ranks=[4, 7, 5, 5]

## Closed-compound evidence omitted by the current phrase index

Diagnostic only: all four concatenations and the completion itself must occur in the same WordNet English vocabulary. No new features were added. Prefixes that are not standalone vocabulary words are excluded; this is a conservative coverage check.

- 2023-06-18: ___ fish that aren't fish; ['cray', 'jelly', 'silver', 'star']; {'preceding': [], 'following': ['fish']}
- 2024-02-24: weights in boxing; ['feather', 'heavy', 'light', 'middle']; {'preceding': [], 'following': ['weight']}
- 2024-02-26: art movements, with -ism; ['expression', 'manner', 'romantic', 'surreal']; {'preceding': [], 'following': ['ism']}
- 2024-08-06: ___horn; ['bull', 'fog', 'matter', 'shoe']; {'preceding': [], 'following': ['horn']}
- 2024-12-07: long___; ['bow', 'fellow', 'horn', 'legs']; {'preceding': ['long'], 'following': []}
- 2025-02-17: ___fish; ['blow', 'cat', 'gold', 'sword']; {'preceding': [], 'following': ['fish']}
- 2025-03-08: ___box; ['chatter', 'juke', 'shadow', 'soap']; {'preceding': [], 'following': ['box']}
- 2026-03-24: heteronyms; ['bow', 'row', 'sow', 'wind']; {'preceding': [], 'following': ['er']}
- 2026-04-07: ___life; ['after', 'low', 'night', 'wild']; {'preceding': [], 'following': ['life']}

Raw per-group evidence, all color/channel summaries, and per-seed group ranks are stored in `coverage.json`. Category descriptions are inspected only for diagnosis; they are not inputs to feature construction or the trained models.
