# One-letter substitution and deletion semantics on GINE

Three matched seeds (10–12), 15,000 updates per arm, batch 16, validation only. Same tuned standalone GINE, original 15 channels, frozen Qwen text embeddings, fully trainable active LayerNorm/projection/GINE/readout, probability-mass loss and balanced shuffled s2/s3/s4 board-size updates as the baseline. No new curriculum, phonetic features, auxiliary loss, LR schedule, early stopping or tuning. Notebooks and test set untouched.

## Added evidence

For each single ASCII alphabetic input, enumerate exact one-letter substitutions and deletions whose outputs are WordNet English vocabulary words. The source need not be a valid word. Target readings must have at least three letters; source length is capped at 32. No insertion, multi-letter edit, multilingual translation, entity lookup or category labels are used.

Each operation has first-letter, last-letter, aligned internal position from the start, aligned internal position from the end, and anywhere channels. Internal aligned maxima only compare edits at matching offsets. Anywhere permits different offsets, needed for taxonomic ranks such as `border/classy/foamily/genius`.

Each alignment supplies two channels: maximum nonnegative cosine between frozen embeddings of distinct transformed words, and reading specificity `1/(1+ln(n))`, where n counts distinct aligned ordered reading pairs. Specificity is a lexical ambiguity descriptor, not a calibrated probability. It is separate from the semantic maximum, so the model learns how to use it. Identical transformed targets are excluded to avoid trivial shared-spelling shortcuts; `wind/wine/wing/wink` → `win` is intentionally not new semantic evidence. Empty/unsupported pairs are zero.

Twenty optional channels are appended to the original 15. Every arm has identical 35-channel parameter shapes and identical shared initial weights; added input columns start at zero. Control masks all additions, substitution/deletion masks the other family, and both enables all. New columns can learn only in enabled arms. Shared lexical lookups, frozen vectors, and per-provider board matrices are cached. Resources, parameters, channel order and model revision are fingerprinted for reconstruction by the normal checkpoint loader and Streamlit.

All runs complete the full budget, evaluating every 250 updates. Select by validation full-board Hit@1 with MRR tie-break. These are fresh matched controls, not older reported scores. Seeds reuse the same 121 validation puzzles; repeated validation selection makes absolute scores optimistic.

## Aggregate results

| Metric | Control | Substitution | Deletion | Both |
|---|---:|---:|---:|---:|
| s4 hit1 | 34.44% | 33.33% | 33.06% | 32.23% |
| s4 mrr | 0.4410 | 0.4240 | 0.4312 | 0.4188 |
| s4 hit4 | 50.69% | 47.66% | 50.69% | 48.76% |
| s3 hit1 | 35.47% | 34.92% | 35.40% | 34.71% |
| s3 mrr | 0.4623 | 0.4525 | 0.4616 | 0.4499 |
| s3 hit4 | 54.06% | 52.20% | 54.13% | 52.82% |
| s2 hit1 | 40.45% | 41.18% | 41.60% | 40.22% |
| s2 mrr | 0.5330 | 0.5331 | 0.5400 | 0.5246 |
| s2 hit4 | 63.82% | 63.41% | 63.50% | 61.98% |
| Harness solve_rate | 24.24% | 24.52% | 25.90% | 25.62% |
| Harness first_guess_accuracy | 38.84% | 33.06% | 38.02% | 36.36% |
| Harness mean_correct_groups | 1.576 | 1.526 | 1.614 | 1.603 |

Harness: the same 121 validation games, beam 20, original full-board scoring, four allowed mistakes, shuffle seed 0, standard feedback pruning. All checkpoints were reconstructed through the shared loader. s2/s3 ranking uses oracle reduced states, not feedback trajectories.

| Seed | Variant | s4 Hit@1 | Solves /121 | Selected update |
|---:|---|---:|---:|---:|
| 10 | control | 35.54% | 29 | 12750 |
| 10 | substitution | 34.71% | 31 | 2500 |
| 10 | deletion | 33.06% | 33 | 2750 |
| 10 | both | 33.88% | 31 | 4000 |
| 11 | control | 34.71% | 32 | 9250 |
| 11 | substitution | 32.23% | 31 | 8000 |
| 11 | deletion | 32.23% | 31 | 3750 |
| 11 | both | 31.40% | 31 | 1250 |
| 12 | control | 33.06% | 27 | 5250 |
| 12 | substitution | 33.06% | 27 | 3750 |
| 12 | deletion | 33.88% | 30 | 6250 |
| 12 | both | 31.40% | 31 | 8250 |

| Color-specific group Top4 | Control | Substitution | Deletion | Both |
|---|---:|---:|---:|---:|
| yellow | 27.27% | 25.07% | 25.07% | 24.52% |
| green | 16.80% | 16.53% | 16.25% | 15.70% |
| blue | 7.16% | 7.44% | 7.71% | 6.34% |
| purple | 3.31% | 2.48% | 3.86% | 4.13% |

| Harness color recovery | Control | Substitution | Deletion | Both |
|---|---:|---:|---:|---:|
| yellow | 51.52% | 47.11% | 51.79% | 50.69% |
| green | 44.35% | 43.53% | 44.63% | 43.80% |
| blue | 34.71% | 33.33% | 36.09% | 36.36% |
| purple | 27.00% | 28.65% | 28.93% | 29.48% |

## Specific letter-edit groups

The existing coarse inventory identifies 12 letter-edit groups among 484 validation groups. Eight Purple groups directly fit these operations: two substitution groups and six deletion groups. Other labels require insertion or multi-letter edits, or reduce all entries to one identical target. The broader letter-edit bucket is not the same as directly supported coverage.

| Metric on the eight directly supported groups | Control | Substitution | Deletion | Both |
|---|---:|---:|---:|---:|
| Specific-group top1 | 0.00% | 0.00% | 0.00% | 0.00% |
| Specific-group top4 | 0.00% | 0.00% | 0.00% | 0.00% |
| Specific-group mrr | 0.0021 | 0.0014 | 0.0013 | 0.0017 |
| substitution target-group top4 | 0.00% | 0.00% | 0.00% | 0.00% |
| substitution target-group mrr | 0.0050 | 0.0019 | 0.0018 | 0.0021 |
| deletion target-group top4 | 0.00% | 0.00% | 0.00% | 0.00% |
| deletion target-group mrr | 0.0011 | 0.0012 | 0.0011 | 0.0016 |
| Full solves on the eight target puzzles | 3/24 | 2/24 | 3/24 | 4/24 |
| Purple target groups recovered in harness | 3/24 | 2/24 | 3/24 | 5/24 |

These are the same eight groups repeated across three seeds (24 appearances), not 24 independent categories. Ranks refer to that specific group among all 1,820 quads on its original board, rather than whether another correct group reaches the top. Labels/intended readings are used for analysis only.

| Date | Category | Intended readings retrieved | Control mean rank | Substitution | Deletion | Both |
|---|---|---:|---:|---:|---:|---:|
| 2025-03-07 | fruits with their second letters changed | 4/4 | 196.0 | 651.3 | 678.3 | 556.3 |
| 2026-01-04 | synonyms for butt plus starting letter | 4/4 | 1552.7 | 1795.0 | 1736.7 | 1805.3 |
| 2026-01-13 | parts of the foot plus starting letter | 4/4 | 1284.7 | 1738.7 | 1436.7 | 1669.0 |
| 2026-02-28 | astronomical terms plus a letter | 4/4 | 766.3 | 1056.0 | 846.3 | 663.3 |
| 2026-05-01 | numbers with first letter changed | 4/4 | 467.0 | 905.0 | 800.3 | 527.7 |
| 2026-05-12 | currencies plus a letter | 4/4 | 1214.0 | 954.0 | 1179.0 | 856.7 |
| 2026-05-31 | kinds of wood plus "s" | 4/4 | 838.7 | 499.0 | 522.0 | 341.0 |
| 2026-09-25 | taxonomic ranks plus a letter | 4/4 | 1323.7 | 1172.7 | 1568.3 | 1445.0 |

## Static evidence and limitations

For each true group, mean edge evidence is ranked against the 1,816 incorrect quads on the same board, with half credit for ties. These are single-channel diagnostics, not model accuracy or proof of a consistent four-word explanation. Each pair can maximize over different lexical readings. The JSON retains maximizing witness words for the eight examples.

| Semantic channel | Eight supported | Two substitution targets | Six deletion targets | All Purple |
|---|---:|---:|---:|---:|
| prefix_semantic_similarity | 0.639 | 0.543 | 0.671 | 0.553 |
| suffix_semantic_similarity | 0.577 | 0.237 | 0.691 | 0.467 |
| substitution_first_semantic | 0.362 | 0.760 | 0.229 | 0.527 |
| substitution_last_semantic | 0.610 | 0.421 | 0.674 | 0.515 |
| substitution_internal_from_start_semantic | 0.570 | 0.751 | 0.510 | 0.494 |
| substitution_internal_from_end_semantic | 0.597 | 0.835 | 0.518 | 0.479 |
| substitution_anywhere_semantic | 0.616 | 0.945 | 0.507 | 0.526 |
| deletion_first_semantic | 0.662 | 0.426 | 0.740 | 0.510 |
| deletion_last_semantic | 0.599 | 0.347 | 0.682 | 0.515 |
| deletion_internal_from_start_semantic | 0.497 | 0.464 | 0.507 | 0.514 |
| deletion_internal_from_end_semantic | 0.544 | 0.475 | 0.566 | 0.514 |
| deletion_anywhere_semantic | 0.778 | 0.199 | 0.972 | 0.500 |

First/last deletion partly overlaps the existing valid prefix/suffix semantic channels. Substitution and internal deletion add genuinely different readings. Improvements cannot be assumed just from finding an intended target in the dictionary; alternative readings and high maxima can introduce noise.

## Interpretation

Substitution alone leaves overall solves almost unchanged (88→89/363), while deletion rises to 94/363 and both to 93/363. Deletion has the best mean solve rate, 25.90% versus 24.24%, with paired solves 29→33, 32→31, 27→30. All additions lower mean full-board Hit@1: control 34.44%, substitution 33.33%, deletion 33.06%, both 32.23%. The small solve gains do not justify promoting a new baseline from this reused validation set.

The stronger finding is a gap between supplied evidence and selected-model retrieval. All 32 intended readings are retrieved. The appropriate positional channel ranks fruit evidence at the 99.72nd percentile versus incorrect quads; numbers and astronomical terms reach 100%. Nevertheless, none of the eight specific groups reaches Top4 in any selected model/seed. Substitution makes the fruit group mean rank worse (196→651) and numbers worse (467→905). Both improves wood-related ranks (839→341) but they remain far from the top.

On the eight target puzzles, the harness fully solves 3/24 control attempts, 2/24 substitution, 3/24 deletion and 4/24 both. It recovers the Purple target on 3/24, 2/24, 3/24 and 5/24 attempts respectively. Feedback and forced final groups can recover targets absent from initial Top4; these small counts still do not establish reliable success on the intended mechanism.

The features therefore add demonstrably selective evidence for these cases, but the present training/selection setup does not convert it into reliable target-group decisions. The any-correct-group objective can ignore difficult groups, making all-group or partition training a useful next controlled test. This experiment does not isolate the cause: noisy lexical maxima, representation limitations and rare training examples remain possible contributors.

Maxima need scrutiny too. In the unrestricted fruit channel, only one of six maximizing reading pairs matches the intended fruits; alternatives include `amble/rote`. Positional alignment restores much more selective fruit evidence. Thus broad maxima are not equivalent to one consistent explanation, even when the intended readings exist. Preserve the opt-in implementation and keep the durable notebook baseline unchanged.

## Feature cost

Train/validation union: 8190 inputs, 9402 distinct readings, median 1.0 and maximum 33 readings per input. Incremental prefetch took 81.7s and graph preparation 32.3s with the existing cache.

warm_resource_uncached_family: median 15.30 ms, p95 31.46 ms per full board on 4 CPU threads.

all_families_warm_matrix_cache: median 0.54 ms, p95 0.58 ms per full board on 4 CPU threads.

These timings exclude Qwen inference for unseen alternatives and initial model/corpus loading. Cached matrices can be sliced for all subgraphs. Cold live boards can still require embedding new lexical readings; no fully cold latency claim is made.

Raw results and checkpoints: [summary.json](gine-letter-edits/summary.json). Per-mechanism ranks: [group-ranks.json](gine-letter-edits/group-ranks.json). Witnesses/coverage: [feature-coverage.json](gine-letter-edits/feature-coverage.json). Curves: [learning-curves.html](gine-letter-edits/learning-curves.html).

Reproduce: `nix develop -c uv run python scripts/run_letter_edit_experiment.py`, then `nix develop -c uv run python scripts/audit_letter_edit_coverage.py`, `nix develop -c uv run python scripts/audit_phonetic_group_ranks.py --results experiments/gine-letter-edits`, and `nix develop -c uv run python scripts/summarize_letter_edit_experiment.py`.
