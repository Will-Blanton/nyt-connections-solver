# Graph feature ablation

This experiment series measures which existing inputs improve held-out ranking
before adding edge functions or changing the graph architecture. Runs use the
date split and fixed model/optimizer configuration from
`graph_set_transformer_optuna_text_spelling_binary_20261018_213815.pt`.

The script trains one variant at a time with the probability-mass objective and
uniformly interleaved 2-, 3-, and 4-group states. Checkpoint selection uses
validation full-board (s4) Hit@1, with s4 MRR as the tie-breaker. The held-out
test split is not used or evaluated.

## Controls

- Seed: 0 for Python, NumPy, Torch, CUDA, model initialization, and data loaders.
- Architecture: one 256-wide attention layer, eight heads, two 512-wide scorer
  layers, dropout 0, and edge bias scale 0.5 when edges are enabled.
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-6.
- Budget: at most 1,500 updates, validation every 100 updates, and early stopping
  after update 700 with six stale validation checks.
- Batch size: 16.
- Primary comparison: validation s4 Hit@1 and MRR. s2/s3 metrics are diagnostic.

## Runs

### 1. Full feature control

- Command: `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant full`
- Result: `experiments/graph-features/20261023T150107Z-full.json`
- Stopped at update 1,400 and restored update 800.
- Validation s4: Hit@1 0.0661, MRR 0.1082, Hit@4 0.1074.
- Validation s3: Hit@1 0.0806, MRR 0.1548.
- Validation s2: Hit@1 0.1543, MRR 0.2757.

The control learned clear signal, but full-board validation flattened while the
training loss continued from 4.80 to below 1.0. This is consistent with a
generalization bottleneck. The run is suitable as the matched workflow control,
so the next run removes both spelling-node features and active edge biases.

### 2. Text-only ablation

- Command: `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant text-only`
- Result: `experiments/graph-features/20261023T150230Z-text-only.json`
- Stopped at update 1,000 and restored update 400.
- Validation s4: Hit@1 0.1405, MRR 0.2177, Hit@4 0.2810.
- Validation s3: Hit@1 0.1736, MRR 0.2530.
- Validation s2: Hit@1 0.2218, MRR 0.3372.

Text-only improved s4 Hit@1 by 0.0744 (nine of 121 validation boards) and more
than doubled s4 MRR relative to the full control. It also improved every reported
metric at every state size. The result does not distinguish harmful spelling
nodes from harmful edge biases because both were removed together.

The next run should be `text-all-edges`: keep the stronger 1,024-dimensional
text nodes and enable all edge channels. If that remains near text-only, the
node spelling concatenation is the likely problem. If it falls toward the full
control, the attention-bias edge path is the likely problem. A subsequent
`text-spelling` run would complete the two-factor isolation.

One plausible node issue is scale and fusion. Qwen embeddings are normalized,
while hashed spelling vectors and raw structural values are concatenated before
one LayerNorm and projection. Separate normalization/projection for semantic and
spelling blocks would be the first spelling-path repair to test if the isolation
confirms that diagnosis.

### 3. Text embeddings with all edge biases

- Command: `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant text-all-edges`
- Result: `experiments/graph-features/20261023T151002Z-text-all-edges.json`
- Stopped at update 1,000 and restored update 400.
- Validation s4: Hit@1 0.1488, MRR 0.1999, Hit@4 0.1983.
- Validation s3: Hit@1 0.1508, MRR 0.2316.
- Validation s2: Hit@1 0.2066, MRR 0.3326.

Relative to text-only, enabling all edge biases moved one additional validation
board to rank 1, but reduced s4 MRR by 0.0177 and s4 Hit@4 by 0.0826. It also
reduced s2 and s3 Hit@1. This is a well-controlled comparison because both runs
have the same 1,024-dimensional nodes and the same parameter shapes and seed;
only use of the already-present edge projections changes.

The edge biases therefore do not explain the full model's large degradation.
They may slightly sharpen the first-ranked choice while harming the remainder of
the ranking, but a one-board Hit@1 difference is too small for a firm positive
claim. The spelling-node concatenation is now the leading explanation for the
full model's poor result. Running `text-spelling` next would test that directly.

| Variant | Best update | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 800 | 0.0661 | 0.1082 | 0.1074 | 0.0806 | 0.1543 |
| Text only | 400 | 0.1405 | 0.2177 | 0.2810 | 0.1736 | 0.2218 |
| Text + all edges | 400 | 0.1488 | 0.1999 | 0.1983 | 0.1508 | 0.2066 |

These are single-seed workflow runs. The nine-board s4 difference is large
enough to prioritize isolation, but feature conclusions should be confirmed
across multiple matched seeds after identifying which path causes the drop.

## Validation breakdown by group color

The runner now records `by_color` in the restored checkpoint's final validation
metrics for each state size. Colors follow the historical CSV row convention:
Purple, Blue, Green, Yellow. They are difficulty/color labels, not semantic
category annotations. Attribution uses `source_group_idx`, including when only
two or three original groups remain; the dataset's legacy color strings are
not used.

Each color's metrics measure the rank of that specific true group among all
candidate subsets: `top1`, `top4`, `mrr`, `mean_rank`, and `groups` (denominator).
These differ from the existing any-correct-group Hit@1/MRR. A correct Yellow
first choice counts for Yellow only; Purple may still rank poorly. There are
121 groups per color for s4 and 363 group appearances per color for s2/s3.
Partial-state appearances from one puzzle are correlated. Checkpoint selection
continues to use any-correct-group s4 Hit@1 with s4 MRR breaking ties.

All three saved checkpoints were reevaluated sequentially without training or
test evaluation. The files end in `-validation-by-color.json`; the original
training histories remain unchanged. Reproduce, for example:

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant text-all-edges --evaluate-only weights/experiments/20261023T151002Z-text-all-edges.pt
```

Full-board validation (121 puzzles); percentages indicate how often that color's
true group occurs in the first four ranked candidates:

| Color | Full top4 | Text only top4 | Text + edges top4 | Text only MRR | Text + edges MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yellow | 5.79% | 14.88% | 8.26% | 0.1139 | 0.0958 |
| Green | 4.13% | 7.44% | 6.61% | 0.0610 | 0.0552 |
| Blue | 1.65% | 4.96% | 4.13% | 0.0399 | 0.0477 |
| Purple | 0.00% | 0.83% | 0.83% | 0.0103 | 0.0105 |

The two text-node models find Purple in the top four on only one puzzle each,
with Purple mean ranks around 964 and 946 out of 1,820 candidates. Existing
handcrafted edges do not materially close that gap in this seed. The drop in
full-board Hit@4 with edges is most visible for Yellow, although Blue MRR
improves. This suggests future feature experiments should explicitly measure
Purple coverage and preserve Yellow ranking quality.

Validation: a targeted synthetic test checks that a partial state drawn from
original Blue and Yellow groups assigns their ranks to those colors and gives
Yellow reciprocal rank 0.5 when Blue is first and Yellow second. Aggregate
ranking metrics from checkpoint reevaluation reproduce the original run results.

## 4. Frozen text scorer with gated candidate edge correction

Implemented `GatedEdgeResidual` in `graph_models.py`. Each candidate's six
unordered pairs passes through a shared 15 -> 32 -> 32 ReLU MLP. Mean and max
pooling produces 64 features. These concatenate with the frozen 256-dimensional
semantic candidate representation. Separate linear heads produce a signed score
correction and a sigmoid gate; the final score is the base score plus their
product. The correction head starts at zero. All base parameters are frozen and
its modules stay in evaluation mode; only the added branch trains. Attention
edge biases stay disabled.

Command:

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-edge-residual --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt
```

The text-only base was selected at update 400 of its earlier run. The branch
trained for 1,500 additional updates with the same state-size mixture, split,
seed, batch size, LR (1e-4), and weight decay (1e-6). Its initial validation
metrics reproduced the text-only baseline. Update 0 remained eligible for
selection; best checkpoint selection used s4 Hit@1 and MRR. The best checkpoint
was the final update 1,500, so the branch was still improving at the budget limit.
This is an added-branch experiment, not an equal-total-training-budget comparison
against continued text-only training.

Result: `graph-features/20261023T152709Z-gated-edge-residual.json`.
Checkpoint: `weights/experiments/20261023T152709Z-gated-edge-residual.pt`.
The experiment CLI supports reloading this model with `--evaluate-only`.
The existing Streamlit checkpoint loader does not yet support this model class.

| Model | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen text baseline | 14.05% | 0.2177 | 28.10% | 17.36% | 22.18% |
| Text with attention edge biases | 14.88% | 0.1999 | 19.83% | 15.08% | 20.66% |
| Frozen text + gated edge branch | 19.01% | 0.2549 | 30.58% | 19.63% | 23.42% |

Full-board color breakdown:

| Color | Base top1 | Branch top1 | Base top4 | Branch top4 | Base MRR | Branch MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Yellow | 7.44% | 10.74% | 14.88% | 15.70% | 0.1139 | 0.1360 |
| Green | 4.13% | 4.96% | 7.44% | 8.26% | 0.0610 | 0.0717 |
| Blue | 1.65% | 2.48% | 4.96% | 5.79% | 0.0399 | 0.0452 |
| Purple | 0.83% | 0.83% | 0.83% | 0.83% | 0.0103 | 0.0106 |

The gated branch yields a net six additional successful first choices out of
121 validation boards. Yellow accounts for a net four, Green one, and Blue one;
Purple contributes no top1 gain. This architecture uses the supplied evidence
more effectively overall while preserving the stronger semantic ranking, but
does not yet recognize more Purple groups near the top. Purple mean rank moves
from roughly 964 to 910, which is insufficient for practical guessing.

Next priorities: confirm across matched seeds, and isolate which edge families
drive the correction before claiming a linguistic-feature benefit. The branch
also receives semantic cosine and WordNet features, so current gains cannot be
attributed specifically to the Purple-oriented channels. Richer phrase/fragment
evidence that preserves a common explanation remains a separate hypothesis.

Tests verify exact baseline logits at initialization, unchanged base parameters
after branch optimization, nonzero edge-encoder gradients, and invariance to
candidate word order. No notebook edits or held-out test evaluation were made.

## 5. Gated branch with only semantic and WordNet edges

This isolates the three semantic channels (`semantic_cosine`, `sense_jaccard`,
`rare_concept_jaccard`) from the other twelve channels. Excluded channels are
masked to zero before the edge MLP. The 15-input MLP, all parameter shapes,
initialization seed, frozen text checkpoint, state mixture, and 1,500-update
budget match experiment 4. The channel mask is stored in model configuration
as `residual_edge_channels`, so checkpoint reconstruction retains the ablation.
A targeted test verifies that arbitrary changes in masked channels cannot
change predictions.

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-semantic-residual --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt
```

Result: `graph-features/20261023T153427Z-gated-semantic-residual.json`.
Checkpoint: `weights/experiments/20261023T153427Z-gated-semantic-residual.pt`.
The best checkpoint was update 1,500; no held-out test evaluation was performed.

| Model | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text only | 14.05% | 0.2177 | 28.10% | 17.36% | 22.18% |
| Gated semantic/WordNet only | 17.36% | 0.2464 | 28.93% | 19.42% | 23.83% |
| Gated all channels | 19.01% | 0.2549 | 30.58% | 19.63% | 23.42% |

Full-board validation by color:

| Color | Semantic-only top1 | All-edge top1 | Semantic-only top4 | All-edge top4 |
| --- | ---: | ---: | ---: | ---: |
| Yellow | 9.09% | 10.74% | 15.70% | 15.70% |
| Green | 4.13% | 4.96% | 6.61% | 8.26% |
| Blue | 3.31% | 2.48% | 5.79% | 5.79% |
| Purple | 0.83% | 0.83% | 0.83% | 0.83% |

Semantic/WordNet edges recover four of the all-edge branch's net six additional
successful full-board first choices over text-only. The remaining twelve
channels provide a modest aggregate benefit in this seed: net two more successes,
higher MRR and Hit@4, with gains concentrated in Yellow/Green rather than Purple.
This comparison does not assign credit to individual channels or establish a
statistically reliable difference. Both branches end at their best checkpoint
at the budget limit, so the comparison is conditional on this training budget.

Purple mean rank is about 973 for semantic-only versus 910 for all-edge, but
neither brings additional Purple groups into the top four. That supports
investigating richer linguistic evidence while keeping the semantic score path.
The next isolation would train the gated branch on the twelve nonsemantic
channels only; repeated branch seeds would then check robustness before making
architecture or feature conclusions.

## 6. Gated branch with the twelve linguistic channels only

Semantic cosine, synset Jaccard, and rare-concept Jaccard are zero-masked.
Channels 3 through 14 remain active: character overlaps, prefix/suffix fragment
similarities, phrase completions, anagrams, and spelling structure. "Linguistic"
here distinguishes this feature family; fragment similarity still uses frozen
text embeddings. Node spelling stays disabled. All branch parameter shapes,
seed, base checkpoint, training settings, and 1,500-update budget match runs 4/5.

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-linguistic-residual --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt
```

Result: `graph-features/20261023T153713Z-gated-linguistic-residual.json`.
Checkpoint: `weights/experiments/20261023T153713Z-gated-linguistic-residual.pt`.
Best checkpoint: update 1,500 (MRR breaks the Hit@1 tie with update 1,400).

| Branch inputs | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original text scorer, no branch | 14.05% | 0.2177 | 28.10% | 17.36% | 22.18% |
| Three semantic/WordNet channels | 17.36% | 0.2464 | 28.93% | 19.42% | 23.83% |
| Twelve linguistic channels | 18.18% | 0.2447 | 28.93% | 19.42% | 23.14% |
| All fifteen channels | 19.01% | 0.2549 | 30.58% | 19.63% | 23.42% |

Full-board color diagnostics for the twelve-channel branch:

| Color | Top1 | Top4 | MRR | Mean rank |
| --- | ---: | ---: | ---: | ---: |
| Yellow | 9.92% | 14.88% | 0.1284 | 520.0 |
| Green | 4.96% | 7.44% | 0.0691 | 585.1 |
| Blue | 2.48% | 5.79% | 0.0448 | 722.0 |
| Purple | 0.83% | 0.83% | 0.0105 | 923.2 |

The linguistic-only branch has 22 successful full-board first choices versus
23 for all-edge, 21 for semantic-only, and 17 for the original text model.
The two isolated branches are close; all-edge remains the best aggregate
configuration in this seed. None adds Purple coverage near the top.

Interpretation caveat for runs 4 through 6: the correction and gate also consume
the frozen semantic candidate representation. Consequently, improvements over
the original text model may come from fitting an additional semantic scoring
head, even with uninformative edges. Earlier attribution of improvements to
particular edge families should be treated as provisional. The required next
control is an otherwise identical branch with all edge channels zero-masked.
This preserves the text-conditioned correction, parameter count, initialization,
and extra training budget, and measures the incremental value of edge information.
Single-seed differences of one or two validation boards remain inconclusive.

## 7. Zero-edge gated-branch control

The identical gated branch receives the frozen semantic candidate representation,
but every edge channel is multiplied by zero. Parameter shapes, initialization,
seed, frozen text checkpoint, state mixture, optimizer, and batch order match
the other branch runs. An empty `residual_edge_channels` list represents this
control and is saved in model configuration. A targeted test confirms that
changing every supplied edge value leaves predictions exactly unchanged.

The first control run stopped at update 1,000 and restored update 900, yielding
s4 Hit@1 14.88% and MRR 0.2212. Its result is retained as
`graph-features/20261023T153922Z-gated-zero-edge-control.json`. To remove unequal
training exposure as a concern, the same seed/configuration was rerun with
early stopping deferred until the full 1,500-update budget. Its shared training
prefix reproduced the first control run.

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-zero-edge-control --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --min-updates-before-stop 1500
```

Full-budget result: `graph-features/20261023T154009Z-gated-zero-edge-control.json`.
Checkpoint: `weights/experiments/20261023T154009Z-gated-zero-edge-control.pt`.
Selected update: 1,500 (MRR tie-breaker). No held-out test evaluation.

| Branch inputs | s4 successes / 121 | s4 Hit@1 | s4 MRR | s4 Hit@4 |
| --- | ---: | ---: | ---: | ---: |
| Original text model, no added branch | 17 | 14.05% | 0.2177 | 28.10% |
| Zero-edge control | 18 | 14.88% | 0.2214 | 28.10% |
| Semantic/WordNet only | 21 | 17.36% | 0.2464 | 28.93% |
| Twelve linguistic channels only | 22 | 18.18% | 0.2447 | 28.93% |
| All fifteen channels | 23 | 19.01% | 0.2549 | 30.58% |

Relative to the zero-edge control, all-edge supplies a net five more successful
first choices (+4.13 percentage points), +0.0334 MRR, and +2.48 points Hit@4.
The added scoring head alone accounts for only a small gain over the original
text model in this seed. This supports an incremental benefit from edge inputs,
while seed replication remains necessary to establish robustness.

Full-board colors for the control:

| Color | Zero-edge top1 | All-edge top1 | Zero-edge top4 | All-edge top4 |
| --- | ---: | ---: | ---: | ---: |
| Yellow | 8.26% | 10.74% | 14.88% | 15.70% |
| Green | 3.31% | 4.96% | 6.61% | 8.26% |
| Blue | 2.48% | 2.48% | 5.79% | 5.79% |
| Purple | 0.83% | 0.83% | 0.83% | 0.83% |

The five net gains over the control are three Yellow and two Green, with no
net Blue/Purple top1 gains. Purple mean rank remains about 967 for the control
versus 910 for all-edge, without any additional top-four coverage. The gated
architecture makes the current features useful primarily for easier groups;
the central Purple representation/evidence problem remains open.

## 8. Candidate-local GINE with gated residual scoring

Replaced the six-independent-pair MLP with two dense GINE layers on each
four-word candidate. Inputs are its four frozen 256-dimensional word
representations from the candidate subset encoder, projected to width 32.
Each layer uses:

`h'_i = MLP((1 + eps) * h_i + sum_{j != i} ReLU(h_j + W_e * e_ij))`

Each layer has an independent bias-free edge projection (15 -> 32), trainable
scalar epsilon initialized to zero, and 32 -> 32 -> 32 ReLU MLP. The complete
four-node graph excludes self-messages. Mean/max pooling of updated nodes gives
64 features. These concatenate with the original frozen semantic candidate
summary and feed the existing sigmoid gate and signed correction heads.
The correction initializes to zero and the text scorer remains frozen. This is
implemented directly in PyTorch; no graph-library dependency was added.

The pooled-edge branch has about 2,210 trainable parameters; GINE has 14,052.
GINE also sees unpooled word representations. Thus an improvement over pooling
cannot be attributed solely to the message-passing rule. The matching next
control should use the same GINE architecture with all edge inputs zeroed.
The previous zero-edge pooled-branch control does not control for this increase
in capacity or node input information.

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-gine-residual --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --min-updates-before-stop 1500
```

Same base checkpoint, date split, seed, batch size, optimizer, LR, and weight
decay as prior branch runs. Best checkpoint: update 1,500, selected by validation
s4 Hit@1 with MRR tie-breaker. Update 0 remained eligible for selection. The run
was still improving at the budget limit; no held-out test evaluation was used.

Result: `graph-features/20261023T154356Z-gated-gine-residual.json`.
Checkpoint: `weights/experiments/20261023T154356Z-gated-gine-residual.pt`.
The experiment CLI can reload it with `--evaluate-only`; Streamlit's existing
checkpoint loader does not yet support this experimental model class.

| Model | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original text only | 14.05% | 0.2177 | 28.10% | 17.36% | 22.18% |
| Pooled gated branch, all edges | 19.01% | 0.2549 | 30.58% | 19.63% | 23.42% |
| GINE gated branch, all edges | 21.49% | 0.2774 | 30.58% | 21.69% | 24.79% |

GINE selects a correct first group on 26/121 full boards versus 23/121 for the
pooled branch: a net gain of three. Full-board color diagnostics:

| Color | Pooled top1 | GINE top1 | Pooled top4 | GINE top4 | Pooled MRR | GINE MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Yellow | 10.74% | 9.92% | 15.70% | 15.70% | 0.1360 | 0.1369 |
| Green | 4.96% | 6.61% | 8.26% | 8.26% | 0.0717 | 0.0862 |
| Blue | 2.48% | 4.13% | 5.79% | 4.96% | 0.0452 | 0.0536 |
| Purple | 0.83% | 0.83% | 0.83% | 1.65% | 0.0106 | 0.0141 |

The net top1 gain is two Green and two Blue, offset by one Yellow. Purple top1
remains one puzzle; its top4 count rises from one to two. In s2 states Purple
top4 improves from 6.89% to 10.47%, and in s3 from 1.10% to 3.31%. This is
some evidence of improved ranking for Purple, but full-board practical coverage
remains poor and the result is a single seed.

Five targeted tests pass across the pooled and GINE branches. They check the
GINE message formula against a hand calculation, excluded diagonal messages,
baseline-equivalent initialization, edge-gradient flow, unchanged frozen
parameters, candidate permutation invariance, masking, and checkpoint roundtrip.
No notebook changes were made.

## 9. Matching GINE zero-edge control and reporting across state sizes

The same two-layer GINE is trained with all fifteen edge channels zero-masked.
All node representations, branch parameter shapes, seed, base weights, batch
order, optimizer, and 1,500-update budget match run 8. GINE still aggregates
neighboring word representations, so this controls for its additional capacity
and per-word semantic inputs. A new test confirms independence from supplied
edge values while retaining dependence on the node inputs.

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-gine-zero-edge-control --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --min-updates-before-stop 1500
```

Result: `graph-features/20261023T154802Z-gated-gine-zero-edge-control.json`.
Checkpoint: `weights/experiments/20261023T154802Z-gated-gine-zero-edge-control.pt`.
Best checkpoint: update 1,500. Checkpoint reload reproduces the validation
metrics. Three GINE tests pass. No notebook or held-out test evaluation changes.

### Metric definitions and downstream states

The recent compact comparison tables report **s4 for all three columns**:
Hit@1, MRR, and Hit@4. These mean any correct remaining group is first, reciprocal
rank of the best-ranked correct group, and any correct group in the first four
candidates. They are not calibrated probabilities of winning a game.

Every training run already evaluates all three state sizes and saves all metrics
and color breakdowns. Each row below uses the same s4-selected checkpoint:

- s4: 16 words, four remaining groups; 121 validation states.
- s3: 12 words, three remaining groups; all four possible removals of a true
  group from each puzzle, giving 484 validation states.
- s2: eight words, two remaining groups; all six true-group pairs per puzzle,
  giving 726 validation states.
- s1: four words left; the only candidate is forced, so no ranking diagnostic.

These are oracle residual states, not trajectories actually visited by the
feedback search harness. They measure ranking after correct removals but do not
capture retries, feedback pruning, beam competition, or the distribution of
states reached by the policy. A puzzle contributes correlated residual states;
do not multiply these rates to estimate whole-puzzle solve probability. Full
feedback-harness validation remains necessary for a global policy comparison.

| Level | Zero-edge Hit@1 | GINE + edges Hit@1 | Zero-edge MRR | GINE + edges MRR | Zero-edge Hit@4 | GINE + edges Hit@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s4 | 15.70% | 21.49% | 0.2252 | 0.2774 | 28.93% | 30.58% |
| s3 | 17.98% | 21.69% | 0.2560 | 0.3025 | 29.55% | 34.50% |
| s2 | 22.59% | 24.79% | 0.3414 | 0.3767 | 40.77% | 45.45% |
| Equal-weight macro | 18.76% | 22.66% | 0.2742 | 0.3189 | 33.08% | 36.85% |

The all-edge GINE improves every decision metric at each state size relative to
its matching zero-edge control. On full boards it adds a net seven first-choice
successes (26 versus 19 out of 121). This supports incremental value from edge
information with no observed loss at later levels in this seed. Future results
also store an equal-weight `validation_macro` across s2/s3/s4, while checkpoint
selection remains s4-first with an s4 MRR tie-breaker. Training continues mixing
all three levels equally by update.

Purple-specific top4 coverage:

| Level | Zero-edge GINE | GINE + edges |
| --- | ---: | ---: |
| s4 | 0.83% | 1.65% |
| s3 | 1.38% | 3.31% |
| s2 | 7.44% | 10.47% |

Purple rankings improve at all levels, but full-board coverage remains only
two of 121 puzzles. The single-seed validation results warrant confirmation
before claiming robust gains or improved live puzzle-solving performance.

## 10. GINE branch stability: matched seeds 0, 1, 2, 3

Completed six new sequential training runs: zero-edge control and all-edge GINE
for each of seeds 1, 2, and 3. Combined with the existing seed-0 pair, every run
used the same frozen text checkpoint and validation dates, 1,500 updates, LR
1e-4, weight decay 1e-6, and batch size 16. Each pair shares initialization seed
and batch/state sampling order. Model architecture and checkpoint selection
remain fixed. No notebooks or held-out test evaluations were changed.

The detailed report, including raw artifact paths and all color/state-size
breakdowns, is [gine-stability.md](gine-stability.md). Regenerate it with:

```bash
nix develop -c uv run python scripts/summarize_gine_stability.py
```

Full-board Hit@1 by seed:

| Seed | Zero edges | With edges | Paired delta |
| --- | ---: | ---: | ---: |
| 0 | 15.70% | 21.49% | +5.79 pp |
| 1 | 14.88% | 16.53% | +1.65 pp |
| 2 | 15.70% | 19.01% | +3.31 pp |
| 3 | 16.53% | 17.36% | +0.83 pp |

Mean ± sample standard deviation across four seeds:

| Level | Control Hit@1 | Edge Hit@1 | Control MRR | Edge MRR |
| --- | ---: | ---: | ---: | ---: |
| s4 | 15.70 ± 0.67% | 18.60 ± 2.19% | 0.2263 ± 0.0033 | 0.2566 ± 0.0142 |
| s3 | 17.87 ± 0.43% | 19.58 ± 1.50% | 0.2563 ± 0.0030 | 0.2853 ± 0.0126 |
| s2 | 22.42 ± 0.17% | 24.28 ± 0.40% | 0.3397 ± 0.0019 | 0.3711 ± 0.0073 |

Edges improve Hit@1, MRR, and Hit@4 at every level in all four paired seeds.
The direction of benefit repeats, but s4 performance varies materially with the
seed. The initial seed-0 result was optimistic: the three fresh edge seeds
average 17.63% s4 Hit@1, rather than 21.49%. The four-seed mean paired s4 gain
is +2.89 percentage points; for the three fresh seeds it is +1.93 points.
Do not present seed 0 alone as typical performance.

Full-board Purple top4 gains do not repeat: seed 0 finds two Purple groups in
the top four, while the other three find only one each, matching their controls.
Purple s3 and s2 top4 improves in every pair: across seeds the averages are
2.41% vs 1.24% at s3, and 9.92% vs 7.37% at s2. Practical full-board Purple
recognition remains weak.

Conclusion: useful, consistently positive edge contribution conditional on this
backbone, split, and budget; less stable absolute s4 performance, and no repeated
full-board Purple coverage improvement. Several edge models are still improving
at the budget endpoint. These runs assess branch optimization stability, not
backbone retraining, dataset-split uncertainty, or live search solve rate.
GINE is a reasonable baseline for the next controlled architecture comparison;
use matched seeds rather than comparing against the best seed-0 checkpoint.

## 11. Longer budget followed by nonlinear edge encoding

Completed sixteen new sequential runs: four matched linear-edge/control seed
pairs with up to 3,000 updates, followed by four nonlinear-edge/control pairs
using the same stopping policy. The longer budget increased mean s4 Hit@1 from
18.60% to 21.07%, with gains at s3/s2 as well. Adding a two-layer nonlinear edge
encoder produced 20.04% mean s4 Hit@1 and greater seed variability; it did not
improve the aggregate ranking comparison. Keep the longer-trained linear-edge
GINE as the current baseline.

Full report, controls, all state sizes, Purple coverage, caveats, and the deferred
GINE/set-attention ordering idea: [gine-budget-and-edge-encoder.md](gine-budget-and-edge-encoder.md).

## 12. GINE input-stage ablation and Purple coverage audit

Completed sixteen new projected/global input-stage runs with matching zero-edge
controls across four seeds, plus coverage audits of existing candidate-input and
new projected-input models. Projected word inputs to the correction branch yield
22.93% mean s4 Hit@1 versus 21.07% for candidate inputs and 19.21% for global
inputs, with improved mean s3/s2 ranking but greater seed variability. The frozen
scorer's full attention path is preserved; this does not establish that global
encoding can be removed from the whole model.

The audit finds both coverage gaps and underuse of complete phrase evidence.
Only six of 39 explicit Purple blank-template groups have common four-word
completions in the current index. Six more have attested closed compounds in
the same WordNet vocabulary that the phrase extractor currently omits. Even the
six fully supported current phrase groups are usually ranked poorly.

Full report, examples, controls, and reproducible scripts:
[gine-input-and-coverage.md](gine-input-and-coverage.md).
