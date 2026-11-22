# Experiment record and deployment handoff

This is the portable research record. Start here when working on another machine.
`experiments/` is ignored and stays on the research machine; it contains raw logs,
trial databases, histories, plots and intermediate outputs. The reports under
`reports/` are small historical snapshots, retaining their original paths beneath
`experiments/`. [manifest.json](manifest.json) indexes every snapshot with its
source path and SHA-256, and records the selected model's provenance and checksum.
The manifest indexes reports, not every individual run or optimizer trial.

## Selected baseline

The November 7 refit selected **ordinary candidate-local GINE, CV recipe, seed 10**:

- Checkpoint: `weights/experiments/20261107T055124Z-v1-refit-cv-seed-10.pt`.
- SHA-256: `fc43e5af84a35f6d032495b5d224b0f4cdf4647bd11c7d91bdc86c62d64b8898`.
- Frozen original Qwen 1024-dimensional word vectors; all active LayerNorm,
  text projection, GINE and readout parameters learned from scratch.
- Learned text projection width 256, GINE width 64 with three layers, readout
  width 64; dropout 0, AdamW LR 0.0009802617130802425, weight decay
  0.005032124249909019. Original 15 ordered edge channels; no coherence or
  direct-summary branch. No phonetic, transformation, or Wiktionary additions.
- Final refit: 1,089 original train+validation puzzles, 1,878 fixed updates,
  batch 16, balanced s2/s3/s4, probability-mass plus floor loss (weight .1,
  beta .1). No validation scheduler or early stopping during refit.
- Harness: beam 20, `score_mode=full_board`, four allowed mistakes, shuffle seed 0.

The previous and CV recipes both solved **75/363** test appearances over three
seeds. These are the same 121 puzzles repeated, not 363 independent puzzles.
CV won the predeclared first-guess tie-break (142 versus 136 correct). It did not
demonstrate a solve-rate improvement; the previous recipe directly recovered more
Purple groups (73 versus 51). The deployment seed was fixed before test evaluation.
The actual selected checkpoint solved **22/121 (18.18%)**, with original-board
Hit@1 **33.06%**; do not assign it the three-seed aggregate solve rate.

[Final comparison and refit protocol](reports/gine-v1-refit/report.md) ·
[Five-fold tuning](reports/gine-coherence-cv/report.md)

The supported production script completed a full verification run on November 20:
1,878 updates, checkpoint reload, all 121 feedback games, and s2/s3/s4 rankings.
It solved 26/121 with s4 Hit@1 37.19%. Recipe/features/splits match the selected
baseline, but the weights differ numerically from the older run; this was pipeline
verification, not new model selection. The selected checkpoint above is unchanged.
See the [verification report](reports/baseline-verification-20261120/report.md).

## Main findings and decisions

These are exploratory findings under the particular controls in each report,
not a leaderboard across different splits, initializations and training budgets.
Older reports describe older baselines; the selected baseline above takes precedence.

| Area | Finding / decision | Evidence |
|---|---|---|
| Search scoring | Sum the four groups' original-board scores, including the forced last group. Wider beams, simple voting, information bonuses and a weakest-group penalty did not provide a reliable gain in the tested older scorer. | [Harness synthesis](reports/harness-findings.md) |
| GINE versus residual scoring | Standalone candidate-local GINE improved validation solves over the older frozen-scorer-plus-correction setup. Active projections and LayerNorm should be trainable; only original Qwen vectors stay frozen. | [Standalone](reports/standalone-gine.md), [trainable projection](reports/standalone-gine-trainable-projection.md), [tuning](reports/gine-hyperparameter-tuning.md) |
| Optimization / representation | Investigated larger projections, smaller adapters, gradients, longer runs, schedules and auxiliary supervision. Treat these as historical ablations, not extra requirements for the final recipe. | [Gradient diagnosis](reports/gine-gradient-diagnosis.md), [long training](reports/gine-long-training.md), [cosine schedule](reports/gine-long-cosine.md) |
| Edge coherence | Five-fold tuning: regular versus coherence Hit@1 40.04% / 39.48%; OOF solves 257/1,089 / 259/1,089. No material solve separation; retain ordinary GINE for simplicity. | [CV report](reports/gine-coherence-cv/report.md) |
| Direct edge-summary channel | Added per-channel clique statistics. Core solves fell 101→94/363; concept arm improved 87→97; edit-target arm fell 99→81. Mixed interactions, no general promotion. | [Matched comparison](reports/gine-edge-summary/edge-summary-comparison.md) |
| Shared concept / edit witnesses | Exact quartet-wide evidence is representable and reusable. Edits have very little positive supervision; the wind/wine/wing/wink target remained last-ranked. Keep optional. | [Witness experiments](reports/gine-relation-witnesses/report.md), [regular versus coherence](reports/gine-relation-witnesses-regular/coherence-comparison.md) |
| Phonetic features | Node n-gram linear fusion effectively tied edge-only; gating and CNNs did not establish an intended sound-group retrieval gain. Keep optional. | [Sound edges](reports/gine-phonetics.md), [node fusion](reports/gine-phonetic-fusion.md) |
| Letter transformations | Pairwise substitution/deletion semantics and target-specific Purple recovery were audited. Aggregate changes alone did not establish recovery of the intended Purple mechanisms. | [Letter edits](reports/gine-letter-edits.md), [Purple audit](reports/gine-letter-edits-purple.md) |
| Learned transformations | Explored node/edge transformation encodings, tuning and smaller networks; not selected for baseline. The capacity follow-up was interrupted and its progress report is not a completed result. | [Transformations](reports/gine-transformations.md), [tuning](reports/gine-transformation-tuning.md), [incomplete capacity study](reports/gine-transformation-capacity.md) |
| Curriculum / objectives | Board-size, candidate-size, fragment states, contrastive, staged residual and probability-floor variants were investigated. Ranking and complete-solve improvements can diverge; avoid combining interventions without matched controls. | [Board curriculum](reports/gine-curriculum.md), [candidate curriculum](reports/gine-candidate-curriculum.md), [fragments](reports/gine-fragment-states.md), [contrastive](reports/gine-group-contrastive.md), [two-stage](reports/gine-two-stage.md), [floor](reports/gine-probability-floor.md), [floor tuning](reports/gine-floor-tuning.md) |
| Sparse connectivity | Hard pruning hurt: complete 97/363 solves, moderate sparsity 59, strict sparsity 9. Existing edge evidence is not a reliable hard filter. | [Sparse GINE](reports/gine-sparse.md) |
| Coverage / mechanism audit | 4,356 train+validation groups inventoried; all 484 validation groups reviewed. Both missing evidence and failure to exploit present evidence remain. Labels are diagnostic only. | [Mechanism audit](reports/group-mechanism-audit/report.md), [gaps](reports/connection-mechanism-gaps.md) |
| Wiktionary phrases | Resource and filtering audits improved intended phrase coverage. Quartet-wide common witnesses reduce incidental pairwise matches. Optional implementation, not trained into selected baseline. | [Resource audit](reports/wiktionary-phrases/report.md), [filters](reports/wiktionary-filters/report.md), [feature guide](../wiktionary-phrase-features.md) |

The manifest lists additional reports, including controls, isolation runs, tuning,
feature coverage and earlier harness evaluations omitted from this abridged table.
See individual reports for scripts, seeds, budgets, metrics and limitations.

## Evaluation boundaries for future agents

- The original test split has been inspected repeatedly, most recently for the
  explicitly authorized final two-recipe selection. It is not a pristine holdout.
  Use newly ingested chronological puzzles / future live games for prospective
  evaluation. Do not silently resume tuning against the old test set.
- s4/s3/s2 mean boards with four/three/two remaining official groups. s3/s2 ranking
  uses oracle reduced states; it is not actual feedback-trajectory performance.
- Hit@1 accepts any correct group. Report complete harness solves separately.
  Distinguish directly selected groups from the forced final group, especially Purple.
- Multiple seeds reuse the same puzzles; their appearances are correlated. CV
  folds were also used to choose hyperparameters and updates, so CV results are
  selection-affected, not an independent final test.
- Source CSV color order is Purple, Blue, Green, Yellow. Do not infer colors from
  legacy reversed label metadata. Preserve edge channel order and checkpoint
  feature metadata exactly.
- Historical reports may say "current", "running", or "next". Those describe
  their original context. The transformation-capacity study was interrupted;
  the five-fold coherence study and final refit completed.

## Transfer to the deployment machine

Git carries these notes, reports, code and tests. It does **not** carry model
weights, datasets, embedding caches, WordNet resources or generated histories.
Transfer the selected checkpoint separately and verify its checksum. Inspect its
feature/resource metadata to transfer or regenerate the matching Qwen cache and
WordNet resources; never substitute embeddings or reorder feature channels.
Historical diagnostics also require the dataset. Wiktionary/phonetic resources
are only needed for checkpoints that actually enable those optional features.

The checkpoint was reconstructed and its Hit@1 reproduced through the production
loader in the final refit audit. That is historical verification, not a fresh
deployment smoke test. Run a fresh load and one historical game after transfer.
The existing Streamlit app is historical diagnostics, not a live-board web adapter.

Use `nix develop -c uv run python ...` for Python scripts. See
[model diagnostics](../model-diagnostics.md) for the Streamlit entry point.
Website input, manual feedback, persistence, hosting and operational logging are
the next branch's scope. A further training run is optional, not required to use
the selected refitted model.

The supported fresh-training entry point is now `scripts/train.py`, with its
complete recipe and date splits in `config/gine-baseline.json`. See
the [training guide](../training.md). Historical experiment runners and tests have
been preserved in [`archive/research/`](../../archive/research/README.md); report
commands retain their original paths and are historical instructions.

## Maintaining this record

On the research machine, after completing experiments:

```bash
nix develop -c uv run python scripts/export_experiment_notes.py
```

This copies only Markdown reports, refreshes their inventory and selected model
provenance, and leaves existing snapshots intact. It does not run experiments,
copy logs or weights, stage files, or upload anything. Update this overview when
conclusions change. Review the resulting diff before committing. The exporter
requires the local final-refit/CV selection records; do not run it on a deployment
checkout without those research artifacts.

Report text is preserved verbatim, including historical links to raw files that
are intentionally absent from Git. Use each manifest entry's `local_source` to
locate those artifacts on the research machine. Keep a separate backup of that
directory if detailed replay or re-analysis is needed; these summaries do not
replace the raw evidence.
