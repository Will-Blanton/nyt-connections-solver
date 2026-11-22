# Closed-compound edge experiment

Four paired seeds (0–3), validation only. Frozen text scorer with projected-input two-layer GINE correction; original 15 edges active in both arms. The feature arm adds two channels for common WordNet-attested closed-compound completions (word before or after); the matched control masks only those two channels. Same split, training schedule, LR 1e-4, weight decay 1e-6, batch 16, budget 3000, selection by s4 Hit@1 then MRR. Each arm selects its own checkpoint. No held-out test evaluation.

| State | Metric | Control | Compound | Difference |
|---|---|---:|---:|---:|
| s4 | Hit@1 | 0.2417 | 0.2314 | -0.0103 |
| s4 | MRR | 0.3469 | 0.3250 | -0.0220 |
| s4 | Hit@4 | 0.4153 | 0.3843 | -0.0310 |
| s4 | Purple top1 | 0.0062 | 0.0083 | +0.0021 |
| s4 | Purple MRR | 0.0161 | 0.0155 | -0.0006 |
| s4 | Purple top4 | 0.0186 | 0.0165 | -0.0021 |
| s3 | Hit@1 | 0.2583 | 0.2459 | -0.0124 |
| s3 | MRR | 0.3683 | 0.3505 | -0.0178 |
| s3 | Hit@4 | 0.4447 | 0.4246 | -0.0201 |
| s3 | Purple top1 | 0.0124 | 0.0131 | +0.0007 |
| s3 | Purple MRR | 0.0330 | 0.0323 | -0.0007 |
| s3 | Purple top4 | 0.0358 | 0.0344 | -0.0014 |
| s2 | Hit@1 | 0.3151 | 0.3054 | -0.0096 |
| s2 | MRR | 0.4514 | 0.4400 | -0.0114 |
| s2 | Hit@4 | 0.5551 | 0.5355 | -0.0196 |
| s2 | Purple top1 | 0.0393 | 0.0427 | +0.0034 |
| s2 | Purple MRR | 0.1057 | 0.1086 | +0.0029 |
| s2 | Purple top4 | 0.1178 | 0.1157 | -0.0021 |

| Seed | Control s4 Hit@1 | Compound s4 Hit@1 | Control best update | Compound best update |
|---:|---:|---:|---:|---:|
| 0 | 0.2397 | 0.2562 | 2900 | 2800 |
| 1 | 0.2314 | 0.1983 | 2700 | 1500 |
| 2 | 0.2645 | 0.2314 | 2900 | 2300 |
| 3 | 0.2314 | 0.2397 | 2500 | 2800 |

The added channels did not help overall: compound s4 Hit@1 fell by 1.03 percentage points, MRR by 0.0220, and Hit@4 by 3.10 points. Purple s4 top1 rose by 0.21 points, but Purple MRR and top4 fell slightly; that is not compelling evidence of improved Purple ranking. s3 and s2 aggregate metrics also fell. The previous 15-edge projected baseline was 22.93% s4 Hit@1; the matched 17-dimensional masked control here reached 24.17%, so use the paired control rather than comparing directly with the prior run.

Feature coverage check on six audited Purple blank templates: the six pair edges supported by the new feature were 3 for fish, 6 for horn, 3 for long, 6 for fish (blow/cat/gold/sword), 6 for box, and 6 for life. Some expected WordNet compounds are absent under the strict requirement that both the full compound and completion be standalone lexicon words. The feature fires on pairwise shared completions; a candidate can still have several unrelated witnesses rather than one completion shared by all four. This and weak Purple MRR suggest the pairwise channel is insufficient by itself.

Raw results:
- Seed 0, control: `experiments/graph-features/20261023T172018Z-gated-gine-projected-compounds-control.json`
- Seed 0, compound: `experiments/graph-features/20261023T172124Z-gated-gine-projected-compounds.json`
- Seed 1, control: `experiments/graph-features/20261023T172231Z-gated-gine-projected-compounds-control.json`
- Seed 1, compound: `experiments/graph-features/20261023T172314Z-gated-gine-projected-compounds.json`
- Seed 2, control: `experiments/graph-features/20261023T172423Z-gated-gine-projected-compounds-control.json`
- Seed 2, compound: `experiments/graph-features/20261023T172534Z-gated-gine-projected-compounds.json`
- Seed 3, control: `experiments/graph-features/20261023T172640Z-gated-gine-projected-compounds-control.json`
- Seed 3, compound: `experiments/graph-features/20261023T172749Z-gated-gine-projected-compounds.json`
