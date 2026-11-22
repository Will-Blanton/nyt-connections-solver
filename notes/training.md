# Production baseline training

Run from the repository root:

```bash
nix develop -c uv run python scripts/train.py
```

The checked-in [`config/gine-baseline.json`](../config/gine-baseline.json) is the
selected regular GINE recipe, including its pinned Qwen revision, feature/resource
identities, ordered 15 edge channels, architecture, optimizer settings, and explicit
date splits. No old checkpoint, experiment output, notebook execution, or archived
script is needed. The Python implementation is in `src/nyt_connections_solver/training.py`.

Inputs are the historical `data/connections.csv`, the pinned Qwen embedding model,
and the matching English WordNet corpus. Install WordNet if absent:

```bash
nix develop -c uv run python -m nltk.downloader wordnet
```

The embedding model can be downloaded by SentenceTransformers or supplied via its
normal local cache. Deterministic features are cached in
`data/feature_cache/graph`. If matching vectors are already cached, avoid loading
the embedding model entirely:

```bash
nix develop -c uv run python scripts/train.py --cache-only
```

Feature metadata must match the selected recipe exactly, including lexical
fingerprints. A mismatch fails rather than silently creating a different model.
Only original Qwen vectors are frozen; text LayerNorm/projection, GINE and readout
initialize fresh and train together. Nodes are Qwen text vectors without spelling,
sense, phonetic or transformed node embeddings. The original 15 edge channels
include semantic, WordNet, substring, phrase and spelling relations.

Training uses constant AdamW, batch 16, 1,878 updates, and balanced interleaved
8/12/16-word states from the beginning. This is not a curriculum. The selected
probability-mass objective includes its small floor penalty (.1 weight, .1 beta),
encouraging some probability on other correct groups. No scheduler, auxiliary
task, staged freezing, early stopping, validation selection or test evaluation
runs in this script. It saves the final fixed-update weights.

The default fit pool is the original 1,089 train+validation puzzles; the 121 old
test puzzles remain excluded. Additional CSV dates are ignored. To change the
fit pool intentionally, supply a new JSON configuration with explicit disjoint
`train`, `val`, and `test` date lists. Do not automatically ingest old held-out
puzzles into training. A changed pool or budget is a new run, not exact replication
of the previous result.

Each invocation creates a new directory under `weights/` with `model.pt`, effective
`config.json`, `training-summary.json` and `status.json`. Existing output directories
are rejected to avoid overwriting runs. The checkpoint records dataset/config
hashes, resource metadata, actual splits and training settings, and loads through
the normal harness and Streamlit registry. `status.json` includes the final
checkpoint SHA-256. Dataset and cache locations can be overridden with `--dataset`
and `--cache-dir`; use `--help` for seed, update, device and output options.

For the selected model's search behavior, use beam 20, `score_mode=full_board`,
and four allowed mistakes. The evaluator now defaults to full-board scoring;
held-out evaluation is always a separate explicit action. Do not use the old test
split for more model selection. See the [experiment handoff](experiments/README.md).

Run the active production test suite:

```bash
nix develop -c uv run python -m unittest discover -s tests -p 'test_*.py' -v
```

Only production training, baseline features/cache/data alignment, checkpoint/UI,
and feedback-search checks remain under `tests/`. Historical research runners
and tests are preserved under [`archive/research/`](../archive/research/README.md).
They are excluded from default test discovery and are not runtime dependencies.
Older report commands refer to their original pre-archive locations.
