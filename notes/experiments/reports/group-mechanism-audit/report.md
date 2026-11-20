# Group mechanisms and similarity evidence

New annotations cover 4,356 groups from 1,089 original training/validation puzzles. The 484 original validation groups were individually model-reviewed by three reviewers; 3,872 training groups retain regex/fallback labels and are explicitly marked unreviewed. The old test set was excluded. Original source data, notebooks and training features were not changed.

## Annotation quality and scope

Model review changed 201/484 primary regex/fallback assignments. This is disagreement with the draft labels, not measured annotation accuracy. Model-reviewed labels are not human ground truth. Multiple mechanisms can apply; secondary labels, subtype, confidence and notes preserve those distinctions. Training fallbacks are broad hypotheses, not a complete verified taxonomy. All unreviewed rows are flagged `needs_review`.

Shared-phrase constructions can overlap ordinary categories: clear compounds such as butter/pocket/etc. + knife are phrase-first, while ordinary category membership is semantic-first with phrase secondary when relevant. Named entities versus familiar general categories also require judgment. These labels are analysis only and never supplied to training, feature construction or the harness.

| Primary mechanism | Purple | Blue | Green | Yellow |
|---|---:|---:|---:|---:|
| anagram | 2 | 0 | 1 | 0 |
| letter_edit | 9 | 1 | 1 | 0 |
| named_entity_or_cultural | 13 | 33 | 11 | 6 |
| orthographic_pattern | 0 | 1 | 3 | 0 |
| phrase_completion | 54 | 12 | 2 | 0 |
| polysemy | 2 | 3 | 0 | 0 |
| semantic_category | 13 | 45 | 60 | 49 |
| sound | 12 | 4 | 0 | 0 |
| synonym | 1 | 20 | 43 | 66 |
| word_containment | 15 | 2 | 0 | 0 |

## What the feature audit measures

CPU-only, two threads, cached frozen embeddings; no model inference or second GPU job. All 41 existing scalar channels were inspected: 15 core channels used by the current CV models plus six optional sound and twenty optional letter-edit channels. Optional channels are audited evidence, not inputs those current models already possess.

For every true quartet, compare each channel aggregated by mean, minimum, or weakest-word mean against all 1,816 incorrect quartets on the same board. Separately compare against the 192 three-of-four near misses. Percentiles give ties half credit: .5 is uninformative, .95 beats/ties approximately 95% of competing candidates and does not imply Top4. Nonzero values alone do not establish useful discrimination. These are univariate diagnostics, not estimates of multichannel-model accuracy or causal ablations.

## Evidence in the intended feature families

The table counts true validation groups with at least one relevant channel at or above the 95th percentile. Channel choice is post-hoc per group, so this is an optimistic diagnostic, not an executable prediction rule. Different mechanisms have different numbers of candidate channels. Sound/letter-edit rows include optional features.

| Mechanism | Groups | Mean strong | Minimum strong | Weakest-word strong |
|---|---:|---:|---:|---:|
| anagram | 3 | 2/3 | 2/3 | 2/3 |
| letter_edit | 11 | 9/11 | 9/11 | 9/11 |
| named_entity_or_cultural | 63 | 20/63 | 21/63 | 18/63 |
| orthographic_pattern | 4 | 4/4 | 4/4 | 4/4 |
| phrase_completion | 68 | 27/68 | 6/68 | 13/68 |
| polysemy | 5 | 1/5 | 0/5 | 0/5 |
| semantic_category | 167 | 85/167 | 76/167 | 76/167 |
| sound | 16 | 11/16 | 7/16 | 8/16 |
| synonym | 130 | 101/130 | 72/130 | 82/130 |
| word_containment | 17 | 6/17 | 7/17 | 7/17 |

## Standalone feature rankings

Rank all 1,820 quartets by just one channel, with no training. Hit@1 below is the probability of a correct first choice if ties at the maximum are broken uniformly. Thus an all-zero channel receives chance credit (4/1,820), rather than an artificial win or loss from candidate order. These are diagnostic channel rankings, not harness solves. The table shows the ten strongest channel/aggregation combinations, selected post-hoc on validation; this selection makes them optimistic as recommendations.

| Channel | Aggregation | Expected Hit@1 | Purple contribution |
|---|---|---:|---:|
| semantic_cosine | mean | 16.53% | 0.83% |
| rare_concept_jaccard | minimum | 15.71% | 1.65% |
| rare_concept_jaccard | mean | 14.88% | 1.65% |
| rare_concept_jaccard | weakest_word | 12.40% | 1.65% |
| semantic_cosine | weakest_word | 10.74% | 0.00% |
| semantic_cosine | minimum | 10.33% | 0.00% |
| phoneme_edit_similarity | mean | 5.79% | 1.65% |
| shared_following_word | mean | 5.42% | 2.65% |
| shared_following_word | weakest_word | 5.31% | 2.66% |
| phoneme_edit_similarity | weakest_word | 4.96% | 0.83% |

## Strongest individual channels by mechanism

Channels below are selected post-hoc within the intended family, using mean validation percentile under weakest-word aggregation; small group counts and multiple comparisons limit interpretation.

| Mechanism | Channel | Current CV input? | Mean percentile | Versus near misses |
|---|---|---|---:|---:|
| anagram | character_multiset_overlap | yes | 0.910 | 0.930 |
| letter_edit | deletion_anywhere_semantic | optional | 0.768 | 0.742 |
| named_entity_or_cultural | semantic_cosine | yes | 0.579 | 0.555 |
| orthographic_pattern | both_palindromes | yes | 0.875 | 0.872 |
| phrase_completion | shared_following_word | yes | 0.574 | 0.571 |
| polysemy | sense_jaccard | yes | 0.500 | 0.500 |
| semantic_category | rare_concept_jaccard | yes | 0.696 | 0.658 |
| sound | homophone_alternative_semantic_similarity | optional | 0.738 | 0.727 |
| synonym | rare_concept_jaccard | yes | 0.819 | 0.787 |
| word_containment | prefix_semantic_similarity | yes | 0.747 | 0.736 |

## Shared phrase witnesses

Check whether all six pairwise matches use one common completion. A clique of pairwise phrase edges need not logically have a global witness. Actual corpus counts below test how often that issue arises here. Closed compounds are audited separately and are not among the 41 feature channels inspected.

| Split | Direction | True all-pair quartets | True shared witness | Incorrect all-pair quartets | Incorrect without shared witness |
|---|---|---:|---:|---:|---:|
| train | closed_following | 76 | 75 | 5625 | 767 |
| train | closed_preceding | 16 | 15 | 187 | 55 |
| train | following | 69 | 66 | 1337 | 179 |
| train | preceding | 33 | 33 | 84 | 16 |
| val | closed_following | 14 | 14 | 875 | 83 |
| val | closed_preceding | 1 | 1 | 74 | 9 |
| val | following | 6 | 6 | 102 | 20 |
| val | preceding | 2 | 2 | 6 | 1 |

Among 68 model-reviewed validation phrase groups, the current spaced-phrase index has a common completion for 6; closed compounds have one for 7 and add 7 distinct groups. An attested completion can be incidental rather than the intended explanation.

## Strong core evidence missed by the previous model

129 validation groups have a relevant core channel at the 95th percentile under weakest-word aggregation but median true-group rank below Top4 for both previous control and amplified models. These ranks reuse the completed three-seed fixed-split experiment; they are not results from the ongoing CV search. High percentile does not prove that all witnesses are consistent or that Top4 is attainable.

A stricter check finds 19 groups whose best relevant core channel has a tie-aware average rank of at most four against incorrect quartets, while both models have median rank worse than twenty. This diagnostic excludes the other three true groups from the single-channel competitor pool. It is stronger evidence of an integration opportunity, but still does not prove a causal model failure.

| Date | Color | Category | Core channel | Percentile | Baseline median rank | Amplified median rank |
|---|---|---|---|---:|---:|---:|
| 2025-10-12 | blue | birds | rare_concept_jaccard | 1.000 | 216 | 286 |
| 2025-03-01 | green | to-bo- compound words | common_prefix | 1.000 | 161 | 240 |
| 2026-01-23 | yellow | facial features | rare_concept_jaccard | 1.000 | 50 | 78 |
| 2025-04-17 | green | components of a pizza | semantic_cosine | 1.000 | 82 | 70 |
| 2025-11-15 | purple | ___ zone | shared_following_word | 1.000 | 179 | 53 |
| 2023-06-28 | yellow | neckwear | rare_concept_jaccard | 1.000 | 19 | 43 |
| 2026-07-26 | blue | expressions for rain | semantic_cosine | 1.000 | 69 | 35 |
| 2026-08-25 | green | sci-fi phenomena | semantic_cosine | 1.000 | 13 | 28 |
| 2026-04-25 | yellow | body coverings | semantic_cosine | 1.000 | 10 | 12 |
| 2024-02-26 | yellow | basic tastes | semantic_cosine | 1.000 | 8 | 11 |
| 2026-04-07 | blue | words for unspecified choices | semantic_cosine | 1.000 | 118 | 5 |
| 2024-08-21 | blue | kinds of charts | shared_following_word | 0.999 | 695 | 434 |
| 2025-09-26 | yellow | stir vigorously | rare_concept_jaccard | 0.999 | 49 | 244 |
| 2025-11-18 | purple | ending in synonyms for "location" | suffix_semantic_similarity | 0.999 | 206 | 157 |
| 2023-08-20 | green | insult | semantic_cosine | 0.999 | 22 | 39 |

## Conclusions supported by this audit

1. **There is useful deterministic signal.** Mean semantic cosine alone reaches 16.53% expected first-choice accuracy on these validation boards; minimum rare-concept overlap reaches 15.71%, against 0.22% chance. Their successes may overlap and must not be added. Neither statistic measures complete-puzzle solves.
2. **No aggregation dominates.** Semantic mean beats semantic weakest-word pooling (16.53% versus 10.74% standalone Hit@1). Phrase-family evidence clears the 95th percentile for 27/68 groups under mean, 13/68 under weakest-word and 6/68 under minimum. Missing pair evidence is common; this argues for supplying multiple summaries rather than enforcing one hard coherence rule.
3. **Phrase-resource coverage is a concrete gap.** Only 6/68 phrase groups have a shared current-index completion. Closed compounds add seven distinct groups, but previous training experiments with those features did not establish a gain. Coverage improvements still need a matched model experiment.
4. **Witness consistency is useful but narrower.** Of 108 incorrect validation quartet–direction occurrences with all-six-pair spaced-phrase support, 21 have no global completion witness. None of the eight supported true quartet–direction occurrences lose support in validation; three training occurrences do. This is a candidate evidence feature, not a justified hard pruning rule.
5. **There are integration opportunities as well as missing information.** The stricter core-evidence/model-rank check identifies 19 groups, including ordinary semantic cases. This supports trying explicit group summary features at scoring time, not only adding more edge types. The relevant channel is selected post-hoc here, so the model must still learn when to trust it.
6. **Optional sound/edit signals are not absent.** At least one intended-family channel has mean evidence at or above the 95th percentile for 11/16 sound groups and 9/11 letter-edit groups. Those are optimistic per-group choices among multiple channels and are not proof of a solve benefit. These optional channels are not enabled in the ongoing core-15 CV sweep.

The strongest bounded follow-up is the proposed separate group-summary scoring channel (retain mean and coherence information), followed by a common-phrase-witness feature and targeted phrase-resource coverage work. Avoid treating a label taxonomy or a high feature percentile as an oracle for group correctness.


## Files and reproduction

- `group_mechanisms.csv`: new annotation dataset, including words, provenance, confidence, secondary mechanisms, review status and CV fold.
- `group_edge_evidence.csv`: all per-group/per-channel values and percentiles.
- `mechanism_feature_discrimination.csv`: per-mechanism channel summaries.
- `intended_family_evidence.csv`: descriptive family coverage by aggregation.
- `phrase_witness_groups.csv` and `phrase_witness_boards.csv`: actual completions and inconsistencies.
- `feature_model_gaps.csv`: relevant core/optional evidence beside previous model ranks.
- `standalone_feature_rankings.csv` and `standalone_feature_summary.csv`: tie-aware single-channel first-choice performance.

- `evidence-vs-model.html`: interactive core-evidence versus model-rank diagnostic.

Reproduce with `nix develop -c uv run python scripts/audit_group_mechanisms.py`, then `nix develop -c uv run python scripts/summarize_group_mechanism_audit.py`. The three `*-review.csv` files preserve reviewer decisions; rerunning regex does not overwrite them. The ongoing five-fold sweep is unchanged.
