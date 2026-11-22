# Wiktionary phrase features

`SpecificPhraseCompletions` supplies two additional channels, preceding and
following completion. It uses the cached lexical Wiktionary titles, retaining
closed compounds and mixed space/hyphen spellings. It does not use puzzle
categories, solutions, or the mechanism audit annotations.

Pairwise evidence is the maximum specificity weight of a shared completion.
For candidate scoring, all four words must share the **same completion**. The
candidate value replaces that channel's off-diagonal edges before GINE message
passing and optional coherence amplification; other edge channels are unchanged.
A zero value suppresses phrase evidence, not the candidate or harness path.

Specificity is `log((N+1)/(df+1))/log(N+1)`, where `N` is the number of distinct
compact dictionary titles and `df` counts titles with that completion at the
relevant end, excluding an empty source. Soft weighting is the default, so valid
productive affixes such as `-ism` remain available. `wiktionary_max_frequency=1000`
optionally enables the audited hard cutoff; it is not the default.

Enable explicitly when constructing a new training configuration:

```python
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE
from nyt_connections_solver.candidate_features import candidate_edge_kwargs

nodes, edges, metadata = build_graph_features(
    embedder, embedding_model_name,
    revision=embedding_revision, use_spelling=False,
    cache_dir="data/feature_cache/graph",
    include_wiktionary=True,
    wiktionary_cache_path="data/wiktionary/headwords.jsonl.gz",
    wiktionary_max_frequency=None,
)
# Use a fresh model with the new edge dimension; preserve other chosen settings.
config = {**base_model_config, "in_dim": nodes.dim, "edge_dim": edges.dim,
          **edges.model_options}
model = StandaloneCandidateGINE(group_idx=group_idx, **config)

# make_state_loaders(..., feature_builder=nodes, edge_providers=edges) includes
# candidate_edge_features automatically. All candidates must follow group_idx.
scores = model(
    batch["x"].to(device), group_idx=group_idx,
    edge_features=batch["edge_features"].to(device),
    **candidate_edge_kwargs(batch, device),
)
```

The shared experiment `score_batch` forwards candidate features automatically.
Custom loops (including notebook-local loops) must use the optional keyword as
shown above. A configured GINE raises if required candidate features are missing,
so the consistency rule cannot silently be skipped.

The composer precomputes and caches an `(1820, 2)` table per original board.
Datasets slice its rows for 8/12-word states. Search uses the same original table,
including after feedback and for permuted residual word orders. This costs about
14 KiB per original board before serialization. Attach `edges.encode` to
`SearchHarness.from_model` as usual; it discovers the candidate builder from the
bound composer. Both feedback search and the ordinary beam harness route it.

Save `config`, `metadata`, and `list(edges.feature_names)` in the normal checkpoint
fields. Loading reconstructs both feature builders, verifies the dictionary and
ordered channels, and works through the existing Streamlit checkpoint loader.
The dictionary cache must be available at its saved path. The local snapshot hash,
normalized entry fingerprint, weighting rule, and optional cutoff identify cache
entries and checkpoints.

These features are opt-in: the original 15-channel setup, existing checkpoints,
notebooks, and the running CV comparison keep their original definitions. The
integration has been tested but no model has yet been trained with these channels.

Validation of this integration: six focused tests pass, covering inconsistent
pair cliques, specificity/cutoff behavior, disk-cache reuse, permuted residual
states, gradients, both search paths, and the production checkpoint loader. A
CPU smoke test with the real dictionary took 3.24 s to load the dictionary,
0.102 s to compute/cache both tables for one validation board, and 0.49 ms for
warm retrieval of both tables (20 repetitions). These are one-board timings,
not end-to-end model latency; details are in
`experiments/wiktionary-filters/integration-smoke.json`.

The full suite passed 144/152 tests. The remaining eight tests reference removed
notebook definitions or the old NDCG training API; the notebooks were not changed
as part of this integration.
