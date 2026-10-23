# Feedback game evaluation

`game/base.py` defines the backend boundary: `ConnectionsEnvironment.observe()` returns
an immutable `GameState`, and `submit(four_words)` returns `GuessFeedback` and
updates the observed state. A browser adapter can implement those two methods.
`play_game(environment, policy)` needs no answers or backend-specific code.

The historical implementation in `game/historical.py` accepts private answer groups
and uses the live rules:

- Correct: accept the group and remove its four words.
- One away: some answer group overlaps the guess in exactly three words. The
  identities of those three words remain unknown.
- Incorrect: every answer group overlaps the guess in at most two words.
- Both kinds of miss consume one mistake. Four mistakes lose by default.
- Submit the last group explicitly; invalid or repeated guesses raise without
  changing state or spending a mistake.

The interface intentionally does not reveal exact 0/1/2 overlaps. This keeps local
evaluation compatible with the feedback available from the live game.

## Use an existing model harness

```python
from nyt_connections_solver.feedback_search import FeedbackSearchHarness
from nyt_connections_solver.game import LocalConnectionsEnvironment, play_game

scorer_harness = load_graph_harness(weights_path)
policy = FeedbackSearchHarness.from_harness(scorer_harness, beam_width=20)
environment = LocalConnectionsEnvironment(displayed_words, private_answer_groups)
result = play_game(environment, policy)
print(result.won, result.final_state.history)
```

For an in-memory model, `FeedbackSearchHarness.from_model(model, word_embedder,
input_dim=..., node_feature_builder=..., edge_feature_builder=..., device=...)`
accepts the feature options of `SearchHarness.from_model`. The embedder can be
omitted when a node feature provider is supplied. A custom scorer can instead
implement `GroupScorer.prepare(words)` and `score_states(states)`. Return every
four-word candidate with its conditional log probability for each state.

## Pruning and scoring

Each decision searches complete partitions of the remaining words and submits
the first group of the best path. At every hypothetical residual state, choices
with no feedback-consistent completion are removed. The remaining model scores
are renormalized to sum to one **before** selecting the beam; path scores sum
these conditional log probabilities. After feedback it searches again, reusing
cached raw scores and recomputing pruning and normalization for the new history.

Accepted groups remain fixed. A rejected guess cannot appear in any surviving
partition. Each one-away response requires at least one group with a three-word
intersection, including previously accepted groups. An ordinary miss rules out
any group with an intersection of three or four. An exact, memoized completion
check prunes partial paths that cannot satisfy all these constraints together.
This includes cases where a needed triple has been split by earlier choices.

Search retains at most `beam_width` feasible partial paths and merges equivalent
remaining states with the same satisfied constraints. Thus feasibility pruning
is exact, but the highest-scoring partition is approximated by beam search.
Contradictory feedback raises an error rather than silently dropping constraints.

Full-board node and edge features are computed once per board. Residual-state
model scores are batched, reused across retries, and bounded to 1,024 cached
states by default. Existing feature providers retain their disk caching.

## Evaluation modules

`evaluation/search.py` contains the existing complete-partition evaluation and
puzzle loading. `evaluation/feedback.py` contains interactive game evaluation.
Both are exported from `nyt_connections_solver.evaluation`, so existing search
notebook imports and the evaluation CLI continue to work.

## Notebook evaluation

Set `RUN_FINAL_SEARCH = True` in `connections-prob-mass.ipynb` to play the selected
model on held-out test puzzles. This runs once after selection, outside Optuna
and epoch validation. Configure `FINAL_MAX_MISTAKES` and `FINAL_BEAM_WIDTH` there.

`evaluate_feedback_games` shuffles board order with a reproducible seed so CSV
answer ordering cannot leak into the decisions. It reports solve rate, mean
accepted groups, first-guess accuracy, guesses, mistakes, and latency, plus a
0-to-4 accepted-group histogram. `test_search_metrics["games"]` retains the public
transcript and elapsed time per puzzle. Timings include first-use feature
construction and model inference; they exclude CSV loading.
