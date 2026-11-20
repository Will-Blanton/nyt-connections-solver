# Wiktionary witness filtering

Validation only: 121 boards. All entries remain available as spaced/closed compounds and substrings; no boundary-only restriction. No training or test evaluation.

A pair clique means all six pairs share some completion in the same direction. A common witness means all four share the exact same completion. Counts below are quartet-direction occurrences, not unique quartets, guesses, or solves.

Unfiltered pair cliques: **74,240** incorrect occurrences. Common-witness filter: **47,629**, removing **26,611 (35.8%)**. The previous 47,629 figure already required a common witness; it is not a pair-clique baseline.

Specificity is measured by the number of distinct compact lexical dictionary titles with a completion at that end, excluding an empty source. Frequencies are global dictionary statistics, computed without puzzle labels. Smaller counts are more specific. IDF weighting alone removes nothing; the following grid measures hard maximum-frequency cutoffs, equivalent to thresholding a monotonic IDF score. Cutoffs are exploratory, not independently validated operating points.

Intended coverage uses explicit, model-reviewed witness annotations: 38 groups, 32 Purple, 15 requiring closed or mixed spellings. Unrelated witnesses do not count as retained intended coverage. Annotation is not human-verified.

| Max dictionary frequency | Incorrect after common witness | Removed vs common witness | Intended retained | Purple retained | Closed/mixed retained |
|---|---:|---:|---:|---:|---:|
| Unlimited | 47,629 | 0.0% | 38/38 | 32/32 | 15/15 |
| 10000 | 31,800 | 33.2% | 38/38 | 32/32 | 15/15 |
| 5000 | 29,915 | 37.2% | 37/38 | 31/32 | 14/15 |
| 2500 | 6,555 | 86.2% | 37/38 | 31/32 | 14/15 |
| 1000 | 2,138 | 95.5% | 37/38 | 31/32 | 14/15 |
| 500 | 586 | 98.8% | 24/38 | 20/32 | 6/15 |
| 250 | 191 | 99.6% | 18/38 | 14/32 | 4/15 |
| 100 | 27 | 99.9% | 5/38 | 5/32 | 1/15 |
| 50 | 13 | 100.0% | 1/38 | 1/32 | 0/15 |
| 25 | 7 | 100.0% | 0/38 | 0/32 | 0/15 |

## Intended completions and their dictionary frequencies

- 2024-02-26 art movements, with -ism: `ism` (following), 8,170 titles; closed/mixed=True
- 2023-09-10 sea ___: `sea` (preceding), 951 titles; closed/mixed=False
- 2024-03-25 tele___: `tele` (preceding), 864 titles; closed/mixed=True
- 2023-06-18 ___ fish that aren't fish: `fish` (following), 716 titles; closed/mixed=True
- 2025-02-17 ___fish: `fish` (following), 716 titles; closed/mixed=True
- 2024-12-07 long___: `long` (preceding), 689 titles; closed/mixed=True
- 2026-05-12 "long" things: `long` (preceding), 689 titles; closed/mixed=False
- 2025-01-03 pot___: `pot` (preceding), 608 titles; closed/mixed=False
- 2024-01-06 ___ ball: `ball` (following), 587 titles; closed/mixed=False
- 2026-08-19 words before "ball" in sport names: `ball` (following), 587 titles; closed/mixed=True
- 2026-05-28 ski ___: `ski` (preceding), 550 titles; closed/mixed=False
- 2025-08-13 star___: `star` (preceding), 529 titles; closed/mixed=True
- 2025-03-08 ___box: `box` (following), 527 titles; closed/mixed=True
- 2026-09-20 ___boards you ride on: `board` (following), 515 titles; closed/mixed=True
- 2024-11-19 double ___: `double` (preceding), 459 titles; closed/mixed=False
- 2025-05-20 words before “bed”: `bed` (following), 426 titles; closed/mixed=False
- 2026-01-23 words before "light": `light` (following), 384 titles; closed/mixed=True
- 2025-11-15 ___ zone: `zone` (following), 370 titles; closed/mixed=False
- 2025-02-22 words after “golden”: `golden` (preceding), 263 titles; closed/mixed=True
- 2024-08-05 words after “eye”: `eye` (preceding), 252 titles; closed/mixed=False
- 2024-08-06 ___horn: `horn` (following), 232 titles; closed/mixed=True
- 2026-04-07 ___life: `life` (following), 175 titles; closed/mixed=True
- 2026-02-09 draw ___: `draw` (preceding), 154 titles; closed/mixed=False
- 2023-11-24 ___ pad: `pad` (following), 149 titles; closed/mixed=False
- 2025-06-21 apple products: `apple` (preceding), 144 titles; closed/mixed=False
- 2023-09-04 ___ chair: `chair` (following), 139 titles; closed/mixed=False
- 2025-03-17 kinds of trucks: `truck` (following), 133 titles; closed/mixed=False
- 2026-03-24 kinds of trucks: `truck` (following), 133 titles; closed/mixed=False
- 2024-08-12 kinds of knives: `knife` (following), 130 titles; closed/mixed=False
- 2025-09-26 kinds of bears: `bear` (following), 125 titles; closed/mixed=False
- 2025-03-04 ___ chain: `chain` (following), 124 titles; closed/mixed=False
- 2024-09-28 ___ control: `control` (following), 109 titles; closed/mixed=False
- 2024-02-24 weights in boxing: `weight` (following), 104 titles; closed/mixed=True
- 2023-08-20 happy ___: `happy` (preceding), 86 titles; closed/mixed=False
- 2023-06-28 ___ triangle: `triangle` (following), 85 titles; closed/mixed=False
- 2024-01-29 words with “fruit”: `fruit` (following), 85 titles; closed/mixed=True
- 2025-09-26 dirty ___: `dirty` (preceding), 71 titles; closed/mixed=False
- 2024-11-15 ___ of time: `oftime` (following), 41 titles; closed/mixed=False

These filters suppress phrase evidence only. They must not hard-prune entire candidate groups or harness paths, since valid groups need not have phrase evidence. Exact witness identity is also essential: pairwise vectors alone cannot establish quartet-wide consistency.

Reproduce: `nix develop -c uv run python scripts/audit_wiktionary_filters.py`
