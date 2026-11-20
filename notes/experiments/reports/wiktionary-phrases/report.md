# Wiktionary phrase coverage

Validation only: 121 boards, 484 groups, including 68 reviewed phrase groups. Indexes use dictionary titles only; puzzle categories are used only for evaluation. No model training or test evaluation. Unified variants use alphabetic, space/hyphen-insensitive completion matching. The spaced variant requires an actual word boundary, and word_completion requires the completion to be a standalone lexical title. Current WordNet uses its existing spaced index.

| Source | Phrase groups | Purple phrase groups | All gold groups | Incorrect quartet-direction matches | Warm lookup ms/board |
|---|---:|---:|---:|---:|---:|
| wordnet_current | 6/68 | 6/54 | 8/484 | 87 | 0.013 |
| wordnet_unified | 16/68 | 16/54 | 27/484 | 1339 | 0.007 |
| wiktionary_lexical | 46/68 | 38/54 | 169/484 | 47629 | 0.007 |
| wiktionary_spaced | 25/68 | 19/54 | 38/484 | 1885 | 0.007 |
| wiktionary_all | 48/68 | 40/54 | 183/484 | 58156 | 0.007 |
| union_lexical | 48/68 | 40/54 | 173/484 | 48036 | 0.008 |
| wiktionary_word_completion | 45/68 | 37/54 | 168/484 | 47620 | 0.251 |

Model inspection of the lexical Wiktionary witnesses confirms the intended template in **38/68 groups**, including **32/54 Purple phrase groups**. Eight of the 46 any-witness hits use unrelated completions; see witness_review.csv and incidental_witness_review.csv. This review is subjective evaluation annotation, not independently human-verified ground truth.

Building all 5 indexes took 6.3s. Warm pairwise tensor construction on one 16-word board averaged 1.55ms over 20 repetitions.

A match means one identical completion supports all four words. Incorrect counts count each quartet once per direction, even if it has multiple witnesses. Accidental substrings remain possible: these are ANY-witness counts, including completions unrelated to the intended mechanism. Coverage is not retrieval accuracy or solve rate. Warm lookup timing covers witness retrieval, not pairwise tensor construction or model inference.

Snapshot: https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz (SHA256 `4d03b6cba283744756c57ff661c2c6cbfe34fcf8ff513a966576cf7d97e91a99`). English Wiktionary contributors, extracted by Kaikki/Wiktextract; https://en.wiktionary.org/wiki/Wiktionary:Copyrights. The upstream download is mutable; preserve the downloaded file and hash for reproducibility.

## Newly covered phrase groups versus unified WordNet

- 2023-06-28 purple — ___ triangle: following: triangle
- 2023-08-20 purple — happy ___: preceding: happy
- 2023-09-04 purple — ___ chair: following: chair
- 2023-09-10 purple — sea ___: preceding: sea
- 2023-11-24 purple — ___ pad: following: pad
- 2024-01-06 purple — ___ ball: following: ball
- 2024-01-29 purple — words with “fruit”: following: fruit, less
- 2024-03-25 purple — tele___: preceding: tele
- 2024-08-12 purple — kinds of knives: following: knife
- 2024-11-15 purple — ___ of time: following: oftime
- 2025-01-03 purple — pot___: preceding: pot; following: less
- 2025-02-22 purple — words after “golden”: preceding: golden; following: like
- 2025-03-04 purple — ___ chain: following: chain
- 2025-03-17 blue — kinds of trucks: following: truck
- 2025-05-20 purple — words before “bed”: following: bed
- 2025-06-21 blue — apple products: preceding: apple; following: less, like
- 2025-08-13 purple — star___: preceding: star; following: man
- 2025-09-26 purple — dirty ___: preceding: dirty
- 2025-09-26 blue — kinds of bears: preceding: anti; following: bear, ly
- 2026-02-09 purple — draw ___: preceding: draw
- 2026-03-24 blue — kinds of trucks: following: truck
- 2026-05-12 blue — "long" things: preceding: long
- 2026-08-19 blue — words before "ball" in sport names: following: ball, baller, less, like
- 2026-09-20 purple — ___boards you ride on: following: board, boarder

## Conclusion

Wiktionary is a useful and inexpensive offline resource. The 509 MiB snapshot produced a 4.8 MiB compressed headword cache. It supplies many missing intended phrases, but unrestricted substring matching creates substantial noise, and requiring a standalone completion barely reduces it. Boundary-respecting matching is a safer first training ablation. Keep spaced and closed-compound evidence distinguishable, retain witnesses, and consider downweighting broadly productive affixes before testing unrestricted closed matching. No solve-rate improvement has been demonstrated.

The provider is opt-in; existing feature order, model checkpoints, notebooks, and the running cross-validation sweep are unchanged. Offline loading example:

```python
from nyt_connections_solver.edge_features.phrase_lexicon import CompletionLexicon, AttestedPhraseCompletions
lexicon = CompletionLexicon.from_headword_cache('data/wiktionary/headwords.jsonl.gz', spaced_only=True)
provider = AttestedPhraseCompletions(lexicon)
```

Reproduce: `nix develop -c uv run python scripts/audit_wiktionary_phrases.py`
