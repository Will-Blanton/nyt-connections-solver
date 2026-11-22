# GINE input-stage ablation and edge coverage audit

Both experiments use the existing frozen text scorer and original validation
split. Notebooks and held-out test evaluation are outside scope.

## Experiment 3: GINE input stage

Compare projected text (before either attention encoder), global word encoding,
and current candidate word encoding as input to the GINE correction branch.
The original text scorer retains its complete global/candidate attention path.
Thus this measures context within the new branch, not the benefit of removing
the global encoder from the whole model.

For projected and global stages, train four matched edge/control pairs (seeds
0–3), one run at a time, using the linear edge encoder, 3,000-update maximum,
minimum 1,500 updates, eight stale checks, and existing s4 Hit@1/MRR selection.
Reuse the previously completed four candidate-stage pairs with these settings.
All branch shapes and shared initializations match; only branch inputs differ.

## Experiment 4: Purple feature coverage

Audit whether existing channels fire on true groups, whether support extends
across all six pairs, and whether mean edge evidence separates true groups from
incorrect candidates. Also compare against high-ranked model distractors and
measure true four-word shared phrase completions versus pairwise existence.
Nonzero dense semantic channels do not establish explanatory coverage. This is
a diagnostic of the existing features, not a new training signal.

## Completed input-stage results

Completed sixteen new sequential runs: four projected-input edge/control pairs
and four global-input edge/control pairs. Reused the existing four candidate-input
pairs with matching training settings. Each variant has the same branch width,
parameter count, initialization of shared parameters, frozen scorer, and budget.
All metrics at every level use the run's s4-selected checkpoint.

Detailed seed-level reports and artifact paths:

- [Projected inputs](gine-projected-input.md)
- [Global inputs](gine-global-input.md)
- [Candidate inputs, previous matched baseline](gine-long-budget.md)

Full-board mean ± sample standard deviation over seeds 0–3:

| Input to GINE branch | Edge Hit@1 | Edge MRR | Edge Hit@4 | Matching zero-edge Hit@1 |
| --- | ---: | ---: | ---: | ---: |
| Projected text | 22.93 ± 3.84% | 0.3153 ± 0.0335 | 36.98 ± 3.12% | 15.91 ± 0.41% |
| After global encoder | 19.21 ± 2.28% | 0.2716 ± 0.0248 | 31.82 ± 2.90% | 16.32 ± 0.79% |
| After candidate encoder | 21.07 ± 2.19% | 0.2858 ± 0.0273 | 33.68 ± 4.50% | 15.91 ± 0.79% |

Subsequent-state means, with edges:

| GINE input | s3 Hit@1 | s3 MRR | s3 Hit@4 | s2 Hit@1 | s2 MRR | s2 Hit@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Projected text | 23.45% | 0.3329 | 40.08% | 28.65% | 0.4194 | 50.93% |
| Global | 19.78% | 0.2948 | 35.85% | 25.69% | 0.3886 | 48.07% |
| Candidate | 21.44% | 0.3099 | 36.98% | 26.58% | 0.3956 | 47.97% |

Projected input beats candidate input on s4 Hit@1 in three of four seeds, with
paired differences -5.79, +6.61, +2.48, and +4.13 percentage points. It improves
the mean by 1.86 points but is more variable. Global input loses to candidate
input in all four seeds on s4 Hit@1. Projected input improves every tracked
ranking metric at every level relative to its own zero-edge control in all seeds.

This supports projected word inputs as a promising GINE-branch configuration,
not a conclusion that the global encoder is unnecessary throughout the system.
The scorer still uses global and candidate attention. Input stages also differ
in numerical distributions/scales as well as contextual content; this experiment
does not isolate those two effects. No input normalization or optimizer retuning
was introduced for any stage. Fixed-backbone/split and small-seed limitations
remain. A swap to GINE-before-candidate-attention in the whole scoring path is
still a separate experiment.

## Completed feature audit

Reports and raw per-group/channel data:

- [Current candidate-input models](edge-coverage/coverage.md)
- [Projected-input models](edge-coverage-projected/coverage.md)
- Each directory includes `coverage.json` with feature metadata, checkpoint
  provenance, words, category descriptions, and per-seed true-group ranks.

The audit uses 121 validation puzzles. Static coverage is counted once per true
group; model misses are correlated appearances across four seeds. It examines
the specific true group's rank, not whether some other correct group is selected.
The probability-mass objective intentionally allows easier correct choices;
poor Purple rank does not automatically imply a failed first decision.

### Coverage is sparse and often nondiscriminative

- Current phrase channels support all six pairs in only 2/121 Purple groups for
  shared preceding words and 4/121 for following words (six distinct groups).
- Among 39 Purple labels with explicit underscore templates, only six have a
  completion shared across all four words in the current phrase index. This is
  a narrow label-based diagnostic subset, not a comprehensive category taxonomy.
- Common-prefix support occurs on any pair in 6.6% of Purple groups, common
  suffix in 15.7%, and exact anagrams in 0.8%. The three explicit spelling-pattern
  flags never support a true Purple pair in this validation split.
- Prefix-fragment semantic similarity has support on all pairs in 22.3% of
  Purple groups, but its mean-evidence percentile versus incorrect quads is only
  0.553 (tie-aware, 0.5 = no separation). Other individual channels' mean-evidence
  percentiles range roughly 0.448–0.533. This does not bound the ability of a
  multichannel nonlinear model; it cautions against equating nonzero values with
  a useful explanation.

### Available evidence is also underused

For the six groups with complete current phrase evidence, the projected-input
models rank the true group as follows (seed order 0, 1, 2, 3):

| Category | Words | Ranks |
| --- | --- | --- |
| ___ pit | barbecue, orchestra, snake, tar | 203, 315, 537, 355 |
| ___ control | birth, cruise, quality, remote | 189, 159, 245, 119 |
| double ___ | agent, dribble, jeopardy, standard | 21, 31, 26, 40 |
| ___ cleaner | bathroom, dry, pipe, vacuum | 72, 168, 83, 101 |
| ___ zone | buffer, comfort, time, twilight | 340, 356, 607, 558 |
| ski ___ | jump, lift, lodge, slope | 4, 7, 5, 5 |

Thus broader dictionaries alone will not fix this. The model rarely promotes
even fully supported phrase groups near the top. Both the original and projected
model audits show this issue. Nor did the audit find conflicting pairwise
completion identities among these complete true-group cases: all six-pair
cliques had an actual shared four-word completion. That specific explanation-loss
hypothesis is not demonstrated by these true-group examples; incorrect candidates
may still have different behavior.

### A concrete extraction gap within the existing resource

The phrase index splits multiword lemmas at spaces, but misses attested closed
compounds. A conservative diagnostic requires all four concatenated forms and
the completion itself to exist in the same WordNet English vocabulary. It finds
six additional explicit-template groups, including:

- cray/jelly/silver/star + fish
- bull/fog/matter/shoe + horn
- long + bow/fellow/horn/legs
- blow/cat/gold/sword + fish
- chatter/juke/shadow/soap + box
- after/low/night/wild + life

The current six phrase-template cases and these six compound-template cases are
distinct. This could raise attested completion coverage from 6/39 to 12/39 for
this narrow subset without adding a new lexical source. The checker also finds
other attested constructions (boxing weights and -ism art movements) and an
alternative completion for a heteronym group. Attestation alone is therefore
evidence, not a guarantee of the intended explanation. No new compound features
were fed to any model in this experiment.

## Interpretation and next priorities

Projected-input GINE is a promising next baseline with stronger average ranking,
but no global encoder has been removed from the frozen scoring path. The audit
supports two targeted follow-ups: extend the phrase extractor to use existing
closed compounds, and investigate why complete phrase evidence is not reliably
used. Completion-identity representations remain a possible improvement, not
an established cure. Keep both coverage and use-of-known-evidence diagnostics
alongside overall/puzzle-level metrics.

Eleven targeted tests pass across the GINE branches and audit helpers. They check
stage extraction, frozen-score preservation, GINE updates, masking, gradients,
permutation/reload behavior, tie-aware percentiles, excluded true-group distractors,
inconsistent pairwise witnesses, and closed-compound attestation requirements.
An actual projected-input checkpoint reload reproduces its recorded metrics.

Reproduction examples:

```bash
nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant gated-gine-projected --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --updates 3000 --min-updates-before-stop 1500 --patience 8 --seed 0
nix develop -c uv run python scripts/summarize_gine_stability.py --updates 3000 --allow-early-stop --variants gated-gine-projected-control gated-gine-projected --output experiments/gine-projected-input.md
nix develop -c uv run python scripts/summarize_gine_stability.py --updates 3000 --allow-early-stop --variants gated-gine-global-control gated-gine-global --output experiments/gine-global-input.md
nix develop -c uv run python scripts/audit_graph_edge_coverage.py --variant gated-gine-projected --output-dir experiments/edge-coverage-projected
```
