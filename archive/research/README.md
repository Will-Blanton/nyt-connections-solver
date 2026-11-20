# Research source archive

Historical experiment runners and their tests are retained here for auditability.
They are tracked source, not ignored generated output. `manifest.json` records
their original paths and the mixed test files preserved before cleanup.

This is a snapshot, not a supported executable package. Scripts retain historical
imports, relative paths, resource assumptions and references to ignored outputs.
To reproduce an old study, reconstruct its original `scripts/` and `tests/`
layout in a separate research checkout and follow its report. Do not add this
archive to the production Python path. Moving the runners does not establish
that their optional models or features are part of the deployed baseline.

Production entry points are in the top-level `scripts/`; default tests are in
the top-level `tests/`. Neither should import this directory. Archived tests
are excluded from default discovery; run them only in a reconstructed research
workspace. Optional/legacy model code remains in `src` for checkpoint
compatibility, without being selected by the production training recipe.

The portable findings and report inventory live in
[`notes/experiments/`](../../notes/experiments/README.md). Raw experiment outputs,
datasets, resources and weights are still ignored and must be backed up separately.
