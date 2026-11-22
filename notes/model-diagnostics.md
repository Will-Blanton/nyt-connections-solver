# Streamlit model diagnostics

Launch from the repository root:

```bash
nix develop -c uv run streamlit run scripts/diagnose.py
```

Open the local address Streamlit prints (normally `http://localhost:8501`).
For access from another machine, use your usual SSH port forwarding.

The selector discovers registered graph checkpoints recursively under the weights
directory, including `weights/experiments/` and the standalone GINE notebook's
new filenames. Graph Set Transformer, edge-residual, GINE-residual, and standalone
GINE checkpoints are supported. Select a model, device, and historical dataset,
then click **Load model**. The app loads
the checkpoint's original embedding revision, feature options, and lexical
resources and validates the reconstructed feature metadata/channel order.
Checkpoints without the registered model/feature metadata are excluded, including
legacy sense models, partition models, and color classifiers.

Model loading is explicit. A session retains one model and reuses feature and
residual-score caches while stepping through a board. Changing the checkpoint
requires loading again. Choose beam width, allowed incorrect guesses, and shuffle
seed in **Run settings**, then click **Apply run settings** to restart the current
puzzle using them. Changing a number without applying does not change a run.
The notebook is not executed by the app. `graph_models.py` contains the matching
architecture, and `checkpoints.py` reconstructs it directly from saved metadata.

## Inspect a puzzle

Choose a checkpoint's validation, test, or training split and a date. The default
is validation. Board order is shuffled reproducibly; seed/date, beam width, and
mistake budget are configurable. Four mistakes lose by default. **Allowed
incorrect guesses per puzzle** counts misses, including one-away responses;
correct groups do not consume the limit. The applied settings are shown beside
the controls and above the batch start button.

The app previews the next decision without submitting it. **Submit next harness
guess** applies the historical environment's feedback and previews the following
decision. **Run puzzle to completion** applies the same loop automatically.
Use **Replay / inspect guess (display only)** to revisit earlier states. This
slider does not change the guess budget or run search. Counters show the state
before the inspected guess; the overall game status appears separately.
**Restart puzzle** resets the public feedback history.

- **Candidate rankings:** raw model rank, softmax mass, exact feedback feasibility,
  rank/probability after pruning, selected guess, and actual 0-to-4 gold overlap.
  Top candidates, all true groups, and the chosen guess are displayed together;
  download the full ranking as CSV. The maximum gold overlap is an analysis
  overlay. The policy itself receives only correct/one-away/incorrect feedback.
- **Search paths:** predicted remaining groups, their gold matches, and retained
  beams at each depth. **Search scoring** defaults to `full_board`, matching the
  GINE experiments; `path` remains available in the applied run settings.
- **Word projections & edges:** 2D PCA of frozen text embeddings and optional
  model-provided representations, with word labels and gold group colors;
  pairwise input-edge heatmaps for each named feature channel.
- **Ground truth & history:** answer groups and the submitted guess transcript.

Gold answers are inspected after the harness chooses and never enter scoring,
beam search, or pruning. Pruning uses only public feedback. Coloring answers is
a UI option. Historical difficulty colors are inferred from the user-confirmed
CSV row order: Purple, Blue, Green, Yellow. The loader preserves that order, and
the board, projection legend, and answer table use the same explicit mapping.
Colors are not stored in a separate CSV column. Custom puzzles without ordered
answers use neutral numbered group labels instead.

Raw softmax mass is a next-choice score, not a calibrated probability of being
correct. After-pruning probabilities renormalize scores over **every** feasible
choice at the current state, before beam truncation. In `full_board` mode, search
scores complete partitions using all four groups' original-board scores, including
the last group. In `path` mode it sums conditional log probabilities recomputed
at residual states. Both recompute the feasible search after feedback changes;
cached raw model scores remain reusable. The harness can choose a lower-ranked
next group because it scores complete paths.
Beam search approximates the score optimum while feasibility pruning is exact.

PCA is fit once per original board and remains fixed as guesses are made.
Standalone GINE displays the actual learned input to candidate message passing.
It does not run the unused inherited attention encoder: GINE's later word vectors
depend on the candidate's other three words and have no single board-wide value.
Attention models can additionally display their full-board encoder vectors.
Models without a representation hook still work with the raw node-feature view;
models without edge features omit the heatmap. The explained-variance caption shows how much information the
2D view retains. Edge heatmaps show supplied input features, not attention
weights or causal feature attribution.

## Batch evaluation

Select dates or **Run entire split**, then click **Run selected puzzles**. The app
shows progress, solve rate, accepted groups, guesses, mistakes, and separate
feature/search/diagnostic timings. Download the summary as CSV and select any
completed puzzle to replay its decisions and rankings. Batch settings are kept
with the results and included in the summary CSV. Applying new settings does not
rerun completed batches. Results with different settings or a different dataset
are marked stale and continue to show their recorded settings; click **Run
selected puzzles** to recompute. Loading a model clears the old batch.

Beam width limits distinct feasible paths retained at each search depth. The
search-path tab shows the actual retained count alongside the configured limit.
A wider beam can consider more alternatives yet still select the same guess.
The candidate-count slider and search-depth selector only change the view.

Batch evaluation is explicit and does not run on every UI rerun. It runs the same
feedback policy as interactive mode. Saving diagnostics increases time and memory
relative to ordinary evaluation: every candidate ranking and retained beam is
kept for replay. Start with a few dates before processing an entire split.

## Adding a model

The UI does not select architectures itself. `checkpoints.py` owns a shared registry;
register a trusted `torch.nn.Module` with `register_graph_model(MyModel)` in code
loaded by the application. Its constructor accepts `group_idx` and `model_config`.
Its forward interface is `forward(x, group_idx=..., edge_features=...)` and returns
one finite score per candidate, shaped `(batch, candidates)` or `(batch, candidates, 1)`.
Save the usual feature, embedding, configuration and split metadata. Models needing
new feature providers must also supply reproducible reconstruction for those providers.

`SearchHarness` handles feature encoding; `ModelGroupScorer` adapts it to the existing
`GroupScorer` protocol; `FeedbackSearchHarness` handles feedback and partition search.
Interactive play and batch play use that same chain. No architecture-specific UI
branch or new game/search harness is needed for a model using this interface.

For optional learned PCA views, implement
`diagnostic_representations(nodes, edges) -> dict[str, Tensor]`, returning one row
per original word. This hook is separate from scoring and receives no answers.

## Verification

```bash
nix develop -c uv run python -m unittest discover -s tests -p 'test_diagnostics.py' -v
```

The tests use a small real graph model and Streamlit's AppTest to exercise model
loading, preview/submission, completion, batch execution, and replay without
downloading an embedding model. Additional checks cover exact notebook/shared
model agreement, probability renormalization, and unchanged decisions when
search tracing is enabled.
