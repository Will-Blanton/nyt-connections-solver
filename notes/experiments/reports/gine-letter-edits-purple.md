# Purple recovery after adding letter-edit features

Validation only: 121 puzzles evaluated with each of three matched training seeds (363 puzzle-seed attempts per arm). These are repeated puzzles, not 363 independent puzzles. Read saved per-guess harness histories; no retraining or new model evaluation.

A recovered Purple group counts whenever the harness accepts it. The earlier-selection column excludes guesses after three other groups have been accepted, when the last four words are forced. This distinction measures direct selection versus endgame recovery; it does not establish which features caused an individual decision.

| Features | Purple recovered | Selected before final four | Forced final four | Purple on first guess |
|---|---:|---:|---:|---:|
| control | 98/363 (27.00%) | 63/363 (17.36%) | 35/363 (9.64%) | 11/363 (3.03%) |
| substitution | 104/363 (28.65%) | 64/363 (17.63%) | 40/363 (11.02%) | 12/363 (3.31%) |
| deletion | 105/363 (28.93%) | 64/363 (17.63%) | 41/363 (11.29%) | 12/363 (3.31%) |
| both | 107/363 (29.48%) | 67/363 (18.46%) | 40/363 (11.02%) | 16/363 (4.41%) |

Combined features add nine Purple recoveries versus control, but five of those net additions are forced last groups. Earlier selections increase only four, from 63 to 67. Per-seed earlier-selection counts are control 22/22/19 versus combined 24/17/26: the direction is inconsistent. Substitution alone and deletion alone each add only one net earlier selection.

On the eight curated Purple groups directly supported by these mechanics (24 puzzle-seed attempts), control recovers 3, substitution 2, deletion 3, and combined 5. Excluding forced final groups, those counts are 3, 1, 3, and 5. Control and deletion recover only the fruit substitution puzzle (2025-03-07), once per seed. Combined retains all three fruit recoveries and additionally recovers the astronomical-terms deletion puzzle (2026-02-28) in seeds 10 and 12. The other six targeted puzzles are never recovered in any arm. None of the eight groups ranks in the initial full-board top four for any model.

Conclusion: a small positive signal for combined features on one extra targeted puzzle, but insufficient evidence of a reliable Purple-solving improvement. Keep the feature families opt-in and retain the baseline. Most aggregate recovery gains are endgame effects; broad gains in direct Purple selection are weak and vary by seed.

Source: [per-seed, per-date recovery details](gine-letter-edits/purple-recovery-detail.json), derived from the twelve saved validation harness histories listed in `gine-letter-edits/runs.json`.
