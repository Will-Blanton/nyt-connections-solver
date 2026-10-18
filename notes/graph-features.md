# Graph feature workflow

Run `connections-graph.ipynb` from the top. Training remains in the existing
baseline, attention, and Optuna cells; feature preparation occurs before them.
The graph notebook uses the same Qwen/Qwen3-Embedding-0.6B model, pinned to its
previously cached revision. No WordNet gloss/sense vectors are included in nodes.
The set notebook imports the extracted dataset helpers but retains its original
node feature configuration.

## Nodes and edges

Nodes concatenate frozen, unprompted text vectors, 128 hashed character unigram/
bigram/trigram counts (L2-normalized), and 13 deterministic structural features.
Hashing is stable across processes and does not fit a vocabulary on held-out data.
Lengths use log1p; other structural channels are fractions or flags. The model's
ordinary input normalization and learned projection consume the concatenation.

The default 15 edge channels are:

- Semantic cosine mapped from [-1, 1] to [0, 1].
- English WordNet synset Jaccard and puzzle-local rarity-weighted ancestor Jaccard.
- Longest common contiguous substring, common prefix, and common suffix lengths,
  normalized by the mean word length; matches shorter than two characters are zero.
- Maximum semantic cosine of valid proper prefixes and suffixes, clipped to [0, 1].
  Fragments must have at least three letters and occur in WordNet's English
  vocabulary. Each fragment contributes a confidence factor min(1, length / 4).
  Whole words are excluded from fragment candidates.
- Shared preceding and following phrase-completion existence (binary).
  Index English WordNet multiword lemmas at every token boundary, so multiword
  puzzle entries/completions also work. Missing phrase evidence gives zero.
- Exact anagram equality and character-multiset Dice similarity.
- Both palindromes, both beginning with a doubled letter, and both containing two
  consecutive doubled-letter pairs. Mutual absence produces no edge.

Character and spelling channels lowercase and ignore nonletters; phrase lookups
normalize whitespace, hyphens, and underscores. These are pairwise evidence,
not guarantees of group membership. No named-category resource is used. WordNet
phrase coverage is limited, and the semantic fragment channels can have accidental
matches. The existing semantic and WordNet channels remain individually observable.

Install WordNet once if missing: `.venv/bin/python -m nltk.downloader wordnet`.
Providers perform no network calls or automatic corpus downloads during encoding.

## Reuse and cache configuration

`build_graph_features()` initializes one `EmbeddingStore`, one `EnglishLexicon`,
a node builder, and edge providers sharing these resources. The notebook passes
`graph_edge_providers` and the shared cache to `make_state_loaders()`.

`data_loader.py` contains `ConnectionsDataset`, `GraphConnectionsDataset`,
`collate_keep_wordlists`, and `make_state_loaders`. The latter prepares each full
board once for all 2/3/4-group datasets. PyTorch DataLoader batches already prepared
tensors. Full-board edges are sliced on both axes for subsets and beam-search
states, preserving the original context used by the WordNet rarity weights.

`FEATURE_CACHE_DIR` defaults to `data/feature_cache/graph`; the ignored `data/`
directory keeps cache artifacts out of Git. The SQLite cache stores text vectors
per string, node tensors per ordered board, and edge tensors per provider and
ordered board. In-memory reuse avoids disk access within a run. Set the cache
directory to `None` for memory-only operation. Set `REFRESH_FEATURE_CACHE = True`
to recompute each encountered key once in that cache instance, then reset it for
subsequent runs. Cached data contain only features, not labels or split membership.

Adding/reordering providers preserves cached results for unchanged providers.
Keys include the provider's channel names and JSON-serializable `cache_identity`:
algorithm version, parameters, lexical fingerprint, and embedding model revision
as appropriate. Custom providers must expose these alongside `encode(words)` and
`feature_names`. Bump their version when changing the calculation. Models without
an available Hugging Face commit hash require an explicit stable revision.
Changing model weights/settings under the same declared revision requires a new
revision identity or a deliberate cache refresh.

## Checkpoints and checks

Graph checkpoints save node/provider metadata and options. `load_graph_harness()`
rebuilds these resources and rejects mismatched metadata or dimensions. It uses the
same node builder and cached full-board edges as training. Older sense-node graph
checkpoints are intentionally rejected by this loader. The existing non-graph
harness path remains supported.

Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` for spelling,
phrase/fragment, cache, subset/permutation, backward, search, and checkpoint checks.
Tests use deterministic small embeddings and the local English WordNet corpus;
they do not train Qwen or evaluate held-out puzzle accuracy.

## Faster tuning defaults

Graph tuning now selects by mean NDCG across 2/3/4 remaining sets, without beam
search. Each trial has at most 25 epochs, stops after six epochs without a
meaningful NDCG improvement (1e-4), and restores its best weights. The scheduler
halves LR after a plateau, down to 1e-6; initial LR is searched over 3e-5 to 2e-4.
Optuna pruning starts after two completed trials with three warmup steps.

An epoch uses `len(train_state_loaders[4])` updates rather than the largest
partial-state loader's length, while sampling state sizes uniformly. Validation
uses the same deterministic, evenly spread subset of at most 128 states per size
for every trial and computes NDCG only (no pairwise ranking loss). Change
`TUNING_STEPS_PER_EPOCH` and `TUNING_VAL_MAX_STATES_PER_SIZE` to adjust the budget;
set the latter to `None` for full validation. These settings deliberately reduce
training exposure and validation coverage during tuning.

`RUN_BASELINES = False` skips the separate baseline/attention runs by default.
`RUN_FINAL_SEARCH = False` skips the optional final beam-search evaluation. Full
validation/test NDCG is still reported once after model selection on the configured
device. Epoch logs and checkpoint metadata record training/validation timings,
epoch count, best epoch, and the tuning budget. Only the best completed trial's
weights are retained in memory. The training loss itself is unchanged.
