# NYT Connections Solver

A machine learning solver for [NYT Connections](https://www.nytimes.com/games/connections).
The puzzle asks players to divide 16 words into four groups of four, with a shared
connection in each group. Correct guesses reveal groups; incorrect guesses may
reveal that three words belong together. Four mistakes end the game.

The project combines a learned group scorer with feedback-aware search. Connections
range from synonyms and common phrases to spelling tricks and cultural references,
making this a useful setting for exploring how learned representations and explicit
relational features work together with limited training data.

## Current model

The selected baseline is a candidate-local **GINE** graph neural network. It scores
every four-word candidate: 1,820 candidates on a full 16-word board. Each candidate
is a complete four-node graph whose nodes represent words and whose edges contain
relation features.

The active scoring path is:

```text
Frozen Qwen word vectors (1024 dimensions)
  → learned LayerNorm and Linear(1024, 256) + ReLU
  → learned Linear(256, 64)
  → three GINE message-passing layers within each candidate
  → concatenate mean and max node pooling (128 dimensions)
  → Linear(128, 64) + ReLU → Linear(64, 1)
  → candidate score
```

Each GINE layer learns how edge features modify incoming node messages, sums
messages from the other three words, and updates the receiving word through an
MLP. All active projection, normalization, GINE and readout parameters train from
scratch; only the original Qwen vectors stay frozen. There is no active global
message-passing or attention encoder in this baseline. Board-wide compatibility
is handled by the search over complete partitions.

### Features

Nodes use 1024-dimensional vectors from `Qwen/Qwen3-Embedding-0.6B`, with the
model revision pinned in [the baseline configuration](config/gine-baseline.json).
The selected model uses text-only node inputs. Its 15 ordered edge channels are:

| Relation family | Channels |
|---|---|
| Semantics | Word embedding cosine; WordNet sense overlap; rare-concept overlap |
| Characters | Longest common substring; common prefix; common suffix |
| Fragment semantics | Similarity between valid word prefixes; similarity between valid word suffixes |
| Phrase completion | Shared preceding word; shared following word |
| Anagrams | Exact anagram; character-multiset overlap |
| Spelling structure | Both palindromes; both initial double letters; both consecutive double-letter pairs |

WordNet supplies English lexical resources. Its senses contribute to edge
features, not additional node embeddings. Features are cached and reused across
candidate groups and residual boards. Checkpoints record the embedding revision,
lexical fingerprints, feature options and channel order; loading rejects mismatches.
Phonetic, transformation, Wiktionary and other experimental variants remain
available for research or checkpoint compatibility, but are not enabled in the
selected recipe.

### Training objective

Training maximizes the total softmax probability assigned to any official correct
group in the current state. A small probability-floor penalty discourages assigning
negligible probability to the other correct groups. This replaces the earlier
graded-relevance/NDCG training formulation.

The selected refit uses 1,089 original train+validation puzzles, batch size 16,
and 1,878 optimizer updates. Training interleaves states with two, three and four
remaining groups equally from the start. It uses constant AdamW learning rate
`0.0009802617130802425`, weight decay `0.005032124249909019`, dropout zero, and
floor weight/beta `0.1/0.1`. There is no curriculum, scheduler, staged freezing,
early stopping or test-based checkpoint selection during this refit.

The configuration contains explicit training and held-out dates. Additional dates
in the dataset are not automatically added to training. See the
[training guide](notes/training.md) for changing inputs or running a new recipe.

## Feedback-aware search

The harness uses beam search over disjoint groups, with beam width 20 in the
selected setup. In `full_board` mode it scores a complete partition by summing
all four groups' scores from the original board, including the forced final group.
These are model scores, not calibrated probabilities of solving the puzzle.

After each guess, it recomputes the feasible search using public feedback:

- A correct guess removes that group.
- An ordinary incorrect guess excludes candidate groups sharing three or four
  words with it.
- A one-away response requires some true group to share exactly three words with
  the guess, without assuming which triple is correct.

Retained paths must admit a complete partition consistent with all feedback.
Failed groups cannot be submitted again. Feedback feasibility checks are exact;
beam search approximates the best-scoring feasible partition. Gold answers are
used by the historical environment and diagnostics, not by the scoring policy.

The environment protocol separates game feedback from the model and search.
The current implementation plays historical puzzles; a live website/backend is
the next development stage.

## Setup and training

Run commands from the repository root. The development environment uses Nix,
Python 3.11 and uv, with PyTorch CUDA 12.8 wheels configured in `pyproject.toml`.

```bash
nix develop -c uv sync
nix develop -c uv run python -m nltk.downloader wordnet
nix develop -c uv run python scripts/train.py
```

Supply the historical CSV at `data/connections.csv`, or use `--dataset` to override
its location. The historical format contains a row index, `date`, `category` and
`connections`; `connections` stores a four-word Python list literal. Each puzzle
has four rows, ordered Purple, Blue, Green, Yellow. Dataset files, model weights
and caches are not included in Git and must be transferred or prepared separately.
The pinned Qwen model is loaded through SentenceTransformers.

Training creates a new directory under `weights/` containing `model.pt`, the
effective configuration, training history, completion status and checkpoint
checksum. Existing output directories are rejected. If matching features are
already cached, use `--cache-only` to avoid loading Qwen. Training does not require
an old checkpoint, archived runner, or notebook execution.

## Evaluation and diagnostics

Evaluate a saved model with the historical feedback harness, replacing the
checkpoint path with your run directory:

```bash
nix develop -c uv run python scripts/evaluate_graph_feedback.py \
  weights/RUN_DIRECTORY/model.pt \
  --split test --beam-width 20 --score-mode full_board \
  --output experiments/feedback-evaluation.json
```

The default refit has no validation dates because it fits the original
train+validation pool. Evaluation is a separate explicit action. The original test
split has already been inspected repeatedly; use newly collected chronological
puzzles or future live games for prospective evaluation and further model selection.

Inspect models interactively:

```bash
nix develop -c uv run streamlit run scripts/diagnose.py
```

The app supports historical step-through play, batch runs and replay, candidate
rankings, predicted partitions, edge heatmaps, and 2D word projections alongside
ground truth. It reconstructs registered checkpoints through the shared loader.
See the [diagnostics guide](notes/model-diagnostics.md) for controls and interpretation.

Run the production tests:

```bash
nix develop -c uv run python -m unittest discover -s tests -p 'test_*.py' -v
```

### Latest pipeline verification

On November 20, 2026, a fresh full-budget run completed training, checkpoint
reconstruction, ranking evaluation and all 121 historical held-out games:

| Metric | Result |
|---|---:|
| Complete harness solves | 26/121 (21.5%) |
| Harness first-guess accuracy | 36.4% |
| Full-board model Hit@1 | 37.2% |
| Full-board model MRR | 0.4775 |
| Full-board model Hit@4 | 56.2% |

Raw model Hit@1 accepts any correct group; the harness can choose a different
first guess because it scores complete partitions. Reduced-state ranking uses
oracle remaining groups and is distinct from actual game trajectories. The run
matches the selected recipe and feature definitions, but its learned weights are
not bitwise identical to the earlier refit. This verifies the production pipeline;
it is not a new unbiased benchmark or evidence of a reliable performance gain.
See the [verification report](notes/experiments/reports/baseline-verification-20261120/report.md).

## Research record and next steps

The dataset is small relative to the variety of connection types. Similarity can
solve many straightforward groups, but factual knowledge, word transformations,
and superficially plausible near misses remain difficult. A better group-ranking
metric also does not necessarily produce more complete solves.

The next branch will focus on website infrastructure, live-board input and feedback,
session persistence, and prospective evaluation with the current model. Further
feature learning or Connections-specific pretraining remains a research direction.

- [Experiment overview, selected model and deployment handoff](notes/experiments/README.md)
- [Report inventory and checkpoint provenance](notes/experiments/manifest.json)
- [Training configuration and workflow](notes/training.md)
- [Archived research scripts and tests](archive/research/README.md)

Generated outputs under `experiments/` are ignored. Small report snapshots and
the manifest are tracked under `notes/experiments/`; update them on the research
machine with `scripts/export_experiment_notes.py`. The archive preserves historical
source for auditability and is not a production dependency.
