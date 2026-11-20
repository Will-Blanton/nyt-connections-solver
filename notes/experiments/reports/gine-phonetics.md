# Pronunciation edges on tuned standalone GINE

Three matched seeds (10–12), validation only, 15,000 updates each, batch 16. Same frozen Qwen inputs, fully trainable active LayerNorm/projection/GINE/readout, tuned hidden width 128, two GINE layers, readout 128, dropout 0, LR 0.0003016684667937828, AdamW weight decay 9.500890359387789e-05. No auxiliary loss, LR schedule, early stopping, or Optuna in this experiment. Checkpoint selection uses validation s4 Hit@1, with MRR only breaking checkpoint ties. Notebooks and test puzzles were not used.

## Added features

1. Exact homophones from any CMUdict pronunciation, ignoring lexical stress.
2. Maximum normalized phoneme-token edit similarity over pronunciation alternatives.
3. Shared rhyme from the last primary/secondary stressed vowel onward.
4. Maximum frozen semantic similarity between dictionary-attested homophone alternatives on both sides. This targets groups such as `base/liar/loot/symbol` → `bass/lyre/lute/cymbal`, rather than requiring the original four words to sound alike.
5. Maximum phonetic similarity of valid proper WordNet prefixes.
6. The corresponding proper-suffix channel.

Substring extraction matches the existing semantic channels: alphabetic normalization, minimum three letters, proper prefix/suffix only, and short-fragment confidence penalties. This does not compare arbitrary internal substrings or all phoneme subsequences. Pronunciations of lexical fragments are looked up independently and may differ from their sounds inside the full word. Unknown pronunciations contribute zero and are reported as unknown in the audit, not known negative evidence. Multiple pronunciations are retained; multiword composition is capped deterministically at 32 alternatives.

NLTK CMUdict resource and lexical/embedding identities are fingerprinted in feature metadata. Sound features are opt-in (`include_phonetics=True`), preserving existing 15-channel checkpoint identities and behavior. No new dependency or corpus download was required. [CMUdict documentation](https://www.nltk.org/api/nltk.corpus.reader.cmudict.html).

## Matched comparison

All arms have identical 21-channel architecture and shared initial weights. Added edge-input columns start at zero; the control masks all six additions, whole-word masks only the two substring channels, and the full arm enables all six. Every old parameter is copied from the same per-seed original-width initialization. Thus random initialization and extra parameter count do not explain differences between arms. The control is retrained, not substituted with an older reported score.

| Metric | Control | Whole-word sound | Whole-word + substrings |
|---|---:|---:|---:|
| s4 hit1 | 34.44% | 34.16% | 34.16% |
| s4 mrr | 0.4306 | 0.4340 | 0.4353 |
| s4 hit4 | 48.21% | 51.52% | 50.96% |
| s3 hit1 | 35.67% | 36.57% | 34.99% |
| s3 mrr | 0.4577 | 0.4682 | 0.4586 |
| s3 hit4 | 52.96% | 53.86% | 53.65% |
| s2 hit1 | 40.45% | 42.19% | 41.37% |
| s2 mrr | 0.5283 | 0.5434 | 0.5445 |
| s2 hit4 | 61.89% | 63.87% | 65.11% |
| Harness solve_rate | 23.69% | 24.79% | 23.14% |
| Harness first_guess_accuracy | 35.54% | 35.81% | 34.16% |
| Harness mean_correct_groups | 1.543 | 1.556 | 1.499 |

Harness: same 121 validation puzzles, beam width 20, full-board scoring, four allowed mistakes, shuffle seed 0, existing public-feedback pruning. Saved checkpoints were reconstructed by the normal loader for every harness run. s3/s2 ranking uses oracle reduced states, not feedback trajectories.

| Seed | Variant | s4 Hit@1 | s4 MRR | Solves /121 | Selected update |
|---:|---|---:|---:|---:|---:|
| 10 | control | 35.54% | 0.4432 | 30 | 8250 |
| 10 | whole_word | 35.54% | 0.4500 | 30 | 1750 |
| 10 | whole_word_and_substrings | 34.71% | 0.4443 | 31 | 1750 |
| 11 | control | 33.88% | 0.4207 | 25 | 1500 |
| 11 | whole_word | 33.88% | 0.4173 | 26 | 9000 |
| 11 | whole_word_and_substrings | 32.23% | 0.4220 | 29 | 1500 |
| 12 | control | 33.88% | 0.4281 | 31 | 6500 |
| 12 | whole_word | 33.06% | 0.4348 | 34 | 2000 |
| 12 | whole_word_and_substrings | 35.54% | 0.4396 | 24 | 1500 |

| Color | Control s4 group Top4 | Whole-word | With substrings |
|---|---:|---:|---:|
| yellow | 25.90% | 30.03% | 26.72% |
| green | 14.05% | 14.88% | 16.53% |
| blue | 7.16% | 6.89% | 6.34% |
| purple | 3.58% | 3.03% | 4.41% |

| Color recovered in harness | Control | Whole-word | With substrings |
|---|---:|---:|---:|
| yellow | 183/363 | 187/363 | 180/363 |
| green | 156/363 | 153/363 | 148/363 |
| blue | 122/363 | 124/363 | 118/363 |
| purple | 99/363 | 101/363 | 98/363 |

## Specific group ranks by audited mechanism

These measure the rank of the specific true group on the original board, not whether some other correct group is chosen. Mechanisms are inferred from labels for analysis only; definitions and every assignment are in the inventory JSON.

| Mechanism | Groups per seed | Control group Top4 | Whole-word | With substrings |
|---|---:|---:|---:|---:|
| phrase_template | 48 | 3.47% | 2.08% | 3.47% |
| other_semantic_or_associative | 333 | 16.92% | 18.52% | 17.92% |
| sound | 16 | 0.00% | 4.17% | 4.17% |
| named_or_cultural_association | 53 | 2.52% | 1.89% | 1.89% |
| polysemy | 2 | 0.00% | 0.00% | 0.00% |
| embedded_fragment | 17 | 0.00% | 0.00% | 0.00% |
| letter_edit | 12 | 0.00% | 0.00% | 2.78% |
| anagram | 3 | 66.67% | 66.67% | 66.67% |

### Sound-labelled examples

| Date | Category | Control mean rank | Whole-word mean rank | With substrings mean rank |
|---|---|---:|---:|---:|
| 2023-06-24 | homophones | 804.3 | 2.7 | 3.7 |
| 2023-06-24 | silent "g" | 602.3 | 822.3 | 871.0 |
| 2024-01-31 | rhymes | 1779.7 | 1430.7 | 1688.0 |
| 2024-05-13 | homophones of musicals | 1697.7 | 1622.3 | 1636.3 |
| 2024-07-26 | instrument homophones | 1711.7 | 1696.7 | 1721.7 |
| 2024-09-30 | capital city homophones | 1282.0 | 977.0 | 684.3 |
| 2024-12-08 | homophones of music genres | 1398.0 | 1265.0 | 1206.7 |
| 2024-12-20 | complaint homophones | 1257.0 | 1274.7 | 1052.7 |
| 2025-10-06 | starting with homophones of greetings | 1275.3 | 1283.3 | 1318.3 |
| 2026-02-28 | name homophones | 350.0 | 460.3 | 834.3 |
| 2026-03-16 | silent "w" | 883.3 | 998.3 | 1098.0 |
| 2026-03-16 | words that sound like state abbreviations | 1072.3 | 1013.0 | 742.3 |
| 2026-03-24 | heteronyms | 568.7 | 370.3 | 351.3 |
| 2026-05-15 | homophones of kinds of dogs, familiarly | 1038.7 | 805.7 | 804.0 |
| 2026-07-24 | homophones of religious roles | 1706.3 | 1742.3 | 1736.3 |
| 2026-08-19 | homophones of cgi-animated films | 1775.3 | 1634.0 | 1290.7 |

## Static feature coverage on sound-labelled groups

16 sound-labelled groups across colors (including homophones, rhymes, heteronyms and silent-letter groups). Mean edge evidence is compared with all 1,816 incorrect quads on the same board; ties get half credit. This is a univariate diagnostic, not a model performance estimate or a proof of explanatory coverage.

| Channel | Any pair | All six pairs | Mean true-vs-incorrect percentile |
|---|---:|---:|---:|
| homophones | 6.25% | 6.25% | 0.524 |
| phoneme_edit_similarity | 100.00% | 18.75% | 0.654 |
| shared_rhyme | 18.75% | 12.50% | 0.581 |
| homophone_alternative_semantic_similarity | 93.75% | 56.25% | 0.863 |
| prefix_phonetic_similarity | 37.50% | 0.00% | 0.528 |
| suffix_phonetic_similarity | 18.75% | 0.00% | 0.516 |

## Interpretation

Whole-word features give a small mean harness improvement (23.69% → 24.79%), with solve counts 30→30, 25→26, 31→34. s4 Hit@1 is essentially unchanged (34.44% → 34.16%); MRR and Hit@4 improve slightly. This is not sufficient evidence to promote the feature set as a new baseline after only three seeds on the reused validation puzzles.

Substring sound features do not add a consistent gain: mean solve rate falls to 23.14%, with paired solve counts 30→31, 25→29, 31→24. They remain opt-in experimental features rather than default inputs.

The strongest specific success is `gnu/knew/new/nu`: control ranks 981, 520, 912 become 1, 6, 1 with whole-word features (2, 7, 2 including substrings). Most homophones-of-category groups remain far below the top candidates. For example, instrument homophones stay around mean rank 1697 despite stronger edge evidence. CMUdict gives `liar` the alternative `leier` rather than `lyre`; it gives `keto` alternatives such as `cato` rather than the intended city `Quito`. Dictionary support and correct intended readings must be audited separately.

There is also a learning/use-of-evidence gap: the music-genre group `indy/mettle/seoul/wrap` has homophone-alternative mean evidence at the 99.2nd percentile versus incorrect quads, yet the whole-word model ranks it about 1265 on average. This does not prove the edge is a complete explanation, but it cautions against assuming every failure needs another feature. The any-correct-group objective permits prioritizing easier groups; whether that contributes to neglecting these signals is a hypothesis, not isolated by this experiment.

No baseline notebook, harness policy, or existing checkpoint feature definition was changed. For a next small feature experiment, the audit favors semantic comparisons after one-letter transformations, then alternative-anagram semantics. These still need matched controls and validation evaluation.

Pronunciation resources and word-pair computations are shared/cached. New live boards may require additional frozen embeddings of homophone alternatives; no cold-start latency improvement is claimed from cached harness timings.

Repeated seeds play the same puzzles; these are not independent held-out datasets. Color Top4 measures the rank of that specific group, while Hit@1 accepts any correct group. Homophone-alternative semantic similarity is a noisy maximum over dictionary senses; a high edge does not establish one shared four-word explanation.

Coverage and next-feature hypotheses: [mechanism audit](connection-mechanism-gaps.md). Raw metrics/checkpoints/histories: [summary.json](gine-phonetics/summary.json).

Reproduce: `nix develop -c uv run python scripts/run_phonetic_experiment.py`, then `nix develop -c uv run python scripts/summarize_phonetic_experiment.py`.
