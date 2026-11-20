# Candidate-local edge coherence amplification

Validation solves: control 84/363; amplified 94/363. Direct Purple selections: 64 versus 74.

## Design

Three matched seeds (10/11/12), 121 validation puzzles repeated across seeds. Fresh controls; 15,000 updates per arm, exactly 5,000 at each state size, batch 16. No notebooks or test evaluation. These are correlated appearances of the same validation puzzles.

For each candidate quartet and each raw edge channel, compute the mean connection from each word to the other three, then take the minimum across the four words. Multiply every edge in that channel by (1 + coherence). Ignore diagonals, preserve all channels and connectivity, and calculate once before message passing. No trainable parameters, additional providers, global encoder, or separate score channel. Raw 0–1 inputs can become 0–2 after amplification. This tests coherence-dependent amplification, not hard selection of a winning relation or shared witness identity.

Original 15 edge channels, frozen Qwen embeddings, trainable active LayerNorm/text projection, two GINE layers of width 128 and readout 128. AdamW LR 0.0003016684667937828, weight decay 9.500890359387789e-05, dropout zero. Probability-mass loss plus existing floor regularizer (weight .1, beta .1) unchanged. Evaluate every 250 updates, restore s4 Hit@1 with MRR tie-break. No early stopping, scheduler, tuning, or pruning.

Harness: beam 20, original-board scores for all four groups, four mistakes, shuffle seed 0, normal public-feedback pruning. Normal checkpoint reconstruction retains the amplification setting, including Streamlit. Independent saved-checkpoint audit reproduces all six full-board Hit@1 values across 726 board/model evaluations. Reduced-state s3/s2 metrics use oracle remaining groups, not actual search trajectories.

## Results

| Metric | Control | Amplified |
|---|---:|---:|
| s4 hit1 | 0.3471 | 0.3636 |
| s4 mrr | 0.4492 | 0.4527 |
| s4 hit4 | 0.5262 | 0.5152 |
| s3 hit1 | 0.3657 | 0.3678 |
| s3 mrr | 0.4784 | 0.4789 |
| s3 hit4 | 0.5682 | 0.5668 |
| s2 hit1 | 0.4302 | 0.4389 |
| s2 mrr | 0.5554 | 0.5606 |
| s2 hit4 | 0.6538 | 0.6731 |
| Harness solves | 84/363 (23.14%) | 94/363 (25.90%) |
| Harness first_correct | 130/363 (35.81%) | 132/363 (36.36%) |
| yellow recovered | 171 | 184 |
| green recovered | 154 | 158 |
| blue recovered | 118 | 124 |
| purple recovered | 97 | 107 |
| Purple selected | 64 | 74 |
| Purple forced | 33 | 33 |

## Per seed

| Seed | Arm | s4 Hit@1 | Solves | Purple direct / forced | Selected update |
|---|---|---:|---:|---:|---:|
| 10 | control | 34.71% | 29/121 | 22 / 11 | 11000 |
| 10 | amplified | 35.54% | 32/121 | 27 / 11 | 6000 |
| 11 | control | 35.54% | 19/121 | 20 / 5 | 9250 |
| 11 | amplified | 37.19% | 30/121 | 24 / 10 | 4250 |
| 12 | control | 33.88% | 36/121 | 22 / 17 | 1000 |
| 12 | amplified | 36.36% | 32/121 | 23 / 12 | 2000 |

## Purple mechanisms

Existing coarse mechanism labels are diagnostic only; they never enter training. Each group appears three times across seeds. Direct / forced distinguishes model-selected groups from the last four words being accepted.

| Mechanism | Appearances | Control direct / forced | Amplified direct / forced | Control mean rank | Amplified mean rank |
|---|---:|---:|---:|---:|---:|
| phrase_template | 135 | 31 / 9 | 31 / 9 | 804.8 | 811.1 |
| sound | 36 | 1 / 1 | 3 / 1 | 1047.8 | 934.6 |
| named_or_cultural_association | 45 | 5 / 9 | 6 / 6 | 918.2 | 728.6 |
| polysemy | 6 | 0 / 1 | 1 / 0 | 743.2 | 271.0 |
| other_semantic_or_associative | 60 | 14 / 3 | 18 / 3 | 700.4 | 612.6 |
| embedded_fragment | 45 | 5 / 7 | 7 / 10 | 1069.9 | 976.3 |
| letter_edit | 30 | 4 / 3 | 2 / 4 | 985.7 | 1032.6 |
| anagram | 6 | 4 / 0 | 6 / 0 | 449.0 | 407.0 |

## Paired outcomes

34 new solves and 24 lost solves; 28 gained direct Purple selections and 18 lost direct selections.

## Interpretation

Promising matched validation signal, not an established replacement baseline. Selected s4 Hit@1 improves in all three seeds; complete solves improve in two and decline in one. Direct Purple improves in all three, while total forced Purple recoveries are unchanged. There is substantial per-puzzle turnover.

The intended shared-phrase mechanism does not improve here: phrase-template Purple direct recovery stays at 31/135 appearances, and mean full-board rank slightly worsens. Direct gains instead occur in other semantic, sound, fragment, anagram, cultural, and polysemy buckets, with letter-edit recovery declining. These coarse labels cannot identify the causal feature channel. No sound or letter-edit providers were added in this study.

Learning curves show later overfitting rather than an unfinished learning plateau: training loss continues down while validation ranking declines. Amplified checkpoints are selected at 6,000 / 4,250 / 2,000 updates, after training every arm for 15,000. More updates alone are not indicated by these trajectories.

An earlier floor-0.1 experiment recorded 97/363 solves, versus 94/363 here. That is a historical comparison, not a matched control; this study supports a gain over its fresh 84/363 control but does not establish a new best solve rate. Keep the mechanism opt-in. The separate coherence-vector scoring channel is a reasonable next matched comparison, and a uniform-scale control would help determine whether the benefit comes from coherence specifically.

## Limits and next comparison

Channel means can differ despite their common numeric range. This experiment does not calibrate those distributions or verify that pairwise phrase matches share one actual completion. Amplification also changes message magnitude; a uniform amplification control would be needed to isolate coherence from a general change in edge scale. The separate coherence-vector scoring channel has not been tested.

31 targeted and related regression tests passed: coherence values, an excluded outsider, diagonal exclusion, unchanged input, candidate locality, permutation equivariance, gradients, zero-scale equivalence, checkpoint reconstruction, existing sparse/GINE behavior, probability-floor loss, training probes, and transformation models.

Reproduce with `nix develop -c uv run python scripts/run_gine_edge_coherence.py`, then `scripts/audit_gine_probability_floor.py --results experiments/gine-edge-coherence` and `scripts/summarize_gine_edge_coherence.py` in the same environment. Curves: `experiments/gine-edge-coherence/learning-curves.html`. Per-channel gold/near-miss/other means: `coherence-diagnostics.json`. Detailed paired Purple outcomes: `summary.json`.
