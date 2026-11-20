# Validation group-mechanism inventory

121 validation puzzles / 484 groups. No test groups inspected. This is a coarse, first-match inventory inferred from historical labels, not a trained category classifier or a complete taxonomy. Buckets can overlap in reality; ambiguous labels remain in the broad semantic/association bucket. The JSON lists every assignment for review. Labels and bucket assignments never enter feature construction or training.

| Mechanism | Purple | Blue | Green | Yellow |
|---|---:|---:|---:|---:|
| anagram | 2 | 0 | 1 | 0 |
| embedded_fragment | 15 | 2 | 0 | 0 |
| letter_edit | 10 | 1 | 1 | 0 |
| named_or_cultural_association | 15 | 30 | 6 | 2 |
| other_semantic_or_associative | 20 | 81 | 113 | 119 |
| phrase_template | 45 | 3 | 0 | 0 |
| polysemy | 2 | 0 | 0 | 0 |
| sound | 12 | 4 | 0 | 0 |

## Sound-labelled examples and dictionary coverage

A shared exact pronunciation is different from four homophones of different category members. Dialect/stress variation and dictionary omissions are limitations. Word coverage below counts any pronunciation, not whether the intended reading or spelling is present.

| Date | Color | Label | Words | Direct homophone pairs /6 | Words with pronunciation /4 |
|---|---|---|---|---:|---:|
| 2023-06-24 | purple | homophones | gnu, knew, new, nu | 6 | 4 |
| 2023-06-24 | blue | silent "g" | gnat, gnaw, gnocchi, gnome | 0 | 4 |
| 2024-01-31 | blue | rhymes | choir, fire, liar, fryer | 0 | 4 |
| 2024-05-13 | purple | homophones of musicals | greece, hare, katz, maim | 0 | 4 |
| 2024-07-26 | purple | instrument homophones | base, liar, loot, symbol | 0 | 4 |
| 2024-09-30 | purple | capital city homophones | keto, roam, sophia, soul | 0 | 4 |
| 2024-12-08 | purple | homophones of music genres | indy, mettle, seoul, wrap | 0 | 4 |
| 2024-12-20 | purple | complaint homophones | grown, mown, whale, wine | 0 | 3 |
| 2025-10-06 | purple | starting with homophones of greetings | haiku, haystack, helonium, yeoman | 0 | 3 |
| 2026-02-28 | blue | name homophones | earnest, kneel, rustle, tailor | 0 | 4 |
| 2026-03-16 | purple | words that sound like state abbreviations | any, emmy, envy, okay | 0 | 4 |
| 2026-03-16 | blue | silent "w" | cartwright, two, wrath, wrestle | 0 | 4 |
| 2026-03-24 | purple | heteronyms | bow, row, sow, wind | 0 | 4 |
| 2026-05-15 | purple | homophones of kinds of dogs, familiarly | ciao, palm, peek, pitt | 0 | 4 |
| 2026-07-24 | purple | homophones of religious roles | canter, fryer, llama, none | 0 | 4 |
| 2026-08-19 | purple | homophones of cgi-animated films | ants, cocoa, rheo, seoul | 0 | 3 |

## Mechanisms without dedicated evidence in the original 15 channels

- **Semantic categories after one-letter edits.** Examples: `ample/dote/lumon/poach` (fruits with second letters changed), `hive/mix/pour/wight` (numbers with first letters changed), `cometh/novak/start/sung` (astronomical terms plus a letter). Existing character overlap compares puzzle words; it does not retrieve their transformed lexical readings.
- **Anagrams of different category members.** `abut/bulge/groan/voila` rearrange to `tuba/bugle/organ/viola`. The existing exact-anagram edge only links words with the same letter multiset, so it does not directly cover this task.
- **Truncated names and hidden entity fragments.** `ear/mar/mer/sat` start planet names; `autoharp/contrabass/pepperoni/rosebud` end with beer brands. Prefix/suffix semantic channels cover only WordNet-attested proper fragments and do not resolve those fragments to entities.
- **Pronunciation properties rather than pair similarity.** Heteronyms `bow/row/sow/wind` share a property of having distinct readings; silent-G `gnat/gnaw/gnocchi/gnome` requires spelling-to-sound alignment. Direct homophone/rhyme/edit edges do not explicitly encode either property.
- **Implicit shared senses.** `assignment/defeat/tempo/tired` are readings of “beat”; `carapace/gas station/pastry crust/rowing boat` refer to “shell”. Pairwise semantic similarity does not explicitly identify one shared ambiguous headword.
- **Entity membership, titles, roles, and cultural facts.** `glass/gross/king/stern` (radio hall of fame), `brave/flow/frozen/up` (animated-feature winners), and `berry/madison/stone/swank` (actress winners) lack an explicit knowledge relation. Embeddings may contain some of this information; that is different from dedicated coverage.

## Next bounded experiments suggested by the inventory

1. **One-letter transformation semantics:** retrieve valid words after a deletion or substitution, then compare their frozen embeddings; keep operation types separate and cap candidate counts. Use confidence/rarity and controls: taking maxima over many lexical alternatives creates false positives.
2. **Anagram-neighbor semantics:** index the existing vocabulary by sorted letters and compare embeddings of alternative anagrams. This reuses WordNet and the existing embedding cache.
3. **Pronunciation-property node features:** a few cheap indicators for distinct pronunciations, syllable count, and stress patterns; do not assume these alone identify silent letters.
4. **Broader phrase resources:** improves dictionary coverage, but pairwise witnesses can differ across a group and an earlier closed-compound experiment did not improve overall ranking.

These are hypotheses, not established improvements. Existing nonzero edge values are not proof of explanatory coverage, and lack of a dedicated edge does not prove the model lacks the information.

## Lexical feasibility of selected letter-transform cases

Eight manually interpreted Purple examples have the following intended transformed words in the current WordNet vocabulary. This checks resource availability, not whether a retrieval/scoring algorithm would identify the intended alternatives among distractors. The examples are selected, not an unbiased coverage estimate.

| Date | Intended targets | WordNet-attested /4 |
|---|---|---:|
| 2025-03-07 | apple, date, lemon, peach | 4 |
| 2026-01-04 | rear, tail, rump, can | 4 |
| 2026-01-13 | toe, arch, ankle, heel | 4 |
| 2026-02-28 | comet, nova, star, sun | 4 |
| 2026-05-01 | five, six, four, eight | 4 |
| 2026-05-12 | franc, rand, real, won | 4 |
| 2026-05-31 | ash, oak, pine, teak | 4 |
| 2026-09-25 | order, class, family, genus | 4 |

## A bounded knowledge-graph option

Wikipedia does not require a full graph download for a first test: resolve puzzle entries to candidate pages, cache outgoing links/categories, and compare neighbor overlap or verified two-hop connections on those local neighborhoods. This is bounded evidence, not exact global shortest-path distance. Entity ambiguity (e.g. “up”, “flow”, or “stone”) and generic high-degree pages are likely bigger challenges than raw graph size. It is a separate experiment; no Wikipedia-derived features or network retrieval were added here. MediaWiki reference: [API:Links](https://www.mediawiki.org/wiki/API:Links).

Reproduce: `nix develop -c uv run python scripts/audit_connection_mechanisms.py`.
