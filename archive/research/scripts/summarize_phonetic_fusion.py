"""Summarize the fixed phonetic embedding/fusion sweep and node-only ablation."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev

import altair as alt


def main():
    output = Path('experiments/gine-phonetic-fusion')
    manifest = json.loads((output / 'manifest.json').read_text())
    selection = json.loads((output / 'selected.json').read_text())
    rows, curves = [], []
    for run in json.loads((output / 'runs.json').read_text()):
        result = json.loads(Path(run['result']).read_text())
        harness = json.loads(Path(run['harness']).read_text())
        rows.append(dict(variant=run['variant'], seed=run['seed'], checkpoint=result['checkpoint'],
                         ranking=result['training']['validation'], best_update=result['training']['best_update'],
                         updates=result['training']['updates_ran'], harness=harness['summary'],
                         mean_known_word_gate=result['mean_known_word_gate'],
                         trainable_parameters=result['trainable_parameters'],
                         recovered_colors=dict(Counter(c for game in harness['games'] for c in game['solved_colors']))))
        for item in result['training']['history']:
            curves.append(dict(variant=run['variant'], seed=run['seed'], update=item['update'],
                               hit1=item['val_s4_hit1'], loss=item['train_loss']))
    variants = [*manifest['variants'], 'selected_nodes_only']
    assert len(rows) == len(variants) * len(manifest['seeds'])
    assert all(r['updates'] == 15000 and r['harness']['puzzles'] == 121 for r in rows)
    aggregate = {}
    for variant in variants:
        selected = [r for r in rows if r['variant'] == variant]
        aggregate[variant] = dict(
            ranking={size: {m: mean(r['ranking'][size][m] for r in selected) for m in ('hit1', 'mrr', 'hit4')}
                     for size in ('s4', 's3', 's2')},
            s4_hit1_sd=stdev(r['ranking']['s4']['hit1'] for r in selected),
            harness={m: mean(r['harness'][m] for r in selected) for m in
                     ('solved', 'solve_rate', 'first_guess_accuracy', 'mean_correct_groups', 'mean_seconds')},
            mean_known_word_gate=mean(r['mean_known_word_gate'] for r in selected)
                                 if selected[0]['mean_known_word_gate'] is not None else None,
            trainable_parameters=selected[0]['trainable_parameters'],
            color_top4={color: mean(r['ranking']['s4']['by_color'][color]['top4'] for r in selected)
                        for color in ('yellow', 'green', 'blue', 'purple')},
            recovered_colors={color: sum(r['recovered_colors'].get(color, 0) for r in selected)
                              for color in ('yellow', 'green', 'blue', 'purple')})
    coverage = json.loads((output / 'feature-coverage.json').read_text())
    group_ranks = json.loads((output / 'group-ranks.json').read_text()) if (output / 'group-ranks.json').exists() else None
    summary = dict(manifest=manifest, selection=selection, aggregate=aggregate, runs=rows,
                   feature_coverage=coverage['splits'])
    if group_ranks:
        summary['by_mechanism'] = group_ranks['aggregate']
        diagnostics = group_ranks.get('fusion_diagnostics', [])
        for variant in variants:
            samples = [r['contribution_norm_ratio'] for r in diagnostics if r['variant'] == variant]
            aggregate[variant]['phonetic_contribution_norm_ratio'] = mean(samples) if samples else None
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    chart = alt.Chart(alt.Data(values=curves)).mark_line().encode(
        x='update:Q', y=alt.Y('hit1:Q', title='Validation s4 Hit@1'), color='variant:N',
        tooltip=['variant:N', 'seed:N', 'update:Q', 'hit1:Q', 'loss:Q'])
    chart.properties(width=680, height=220).facet('seed:N', columns=1).save(output / 'learning-curves.html')
    lines = ['# Phonetic node encodings and feature fusion', '',
             'Three matched seeds (10–12) per configuration; 15,000 updates per run; validation-only '
             'selection and evaluation. Qwen stays frozen. The original learned text LayerNorm and '
             '1024→256 projection, candidate-local GINE, readout, phonetic projection/gate and CNN '
             '(where present) all train jointly. No auxiliary loss, LR schedule, backbone freeze, '
             'test evaluation, notebook edits, or extra hyperparameter tuning.', '',
             '## Representations and fusion', '',
             '- **Phoneme n-grams:** 128-dimensional signed-hashed 1/2/3-gram vectors, including '
             'word-boundary markers. Normalize each pronunciation, average alternatives, normalize again. '
             'The vectors are fixed features; their projection is learned.',
             '- **Learned sequence CNN:** 24-dimensional learned phoneme embeddings, three 32-channel '
             'convolutions (kernels 3/5/7), masked max pooling per pronunciation, learned 96→128 '
             'readout, averaging across valid alternatives, then L2 normalization. This encoder is '
             'trained from scratch with puzzle supervision; it is not a pretrained speech encoder.',
             '- **Linear fusion:** `z_text + P(h_phone)` before candidate-local GINE.',
             '- **Gated fusion:** `z_text + sigmoid(w [z_text, P(h_phone)] + b) * P(h_phone)`, '
             'with one learned scalar gate per word.', '',
             'The extra projection starts at zero in every arm; the gate starts at 0.5. All '
             'shared model tensors are copied from identical per-seed initializations, and new '
             'module construction preserves the baseline RNG state. Initial predictions match '
             'the semantic model; new branches learn their contribution. Different encoders have '
             'different parameter counts, so this does not isolate representation from capacity.', '',
             'Use at most eight distinct stress-free CMUdict pronunciations and 32 phonemes per '
             'pronunciation. Unknown words have zero phonetic contribution and preserve their text '
             'branch. The 1408-dimensional cached input contains 1024 text values, 128 n-gram '
             'values, and 8×32 integer phone IDs stored losslessly as float32. IDs are consumed '
             'only by the CNN, never by the text projection or a numeric similarity calculation. '
             'Resources, vocabulary order and layout are stored in checkpoint feature metadata. '
             '[Conv1d reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html).', '',
             '## Controlled training setup', '',
             f"Original tuned parameters: `{json.dumps(manifest['params'], sort_keys=True)}`. "
             'Batch 16, mixed s2/s3/s4 probability-mass loss, evaluation every 250 updates, no '
             'early stopping or pruning. Select each checkpoint by validation s4 Hit@1, MRR '
             'only breaking checkpoint ties.', '',
             'The sweep holds edges fixed: original 15 plus the four whole-word sound channels. '
             'The two substring-sound channels remain masked because the prior experiment did '
             'not show a consistent benefit. The edge-only control has the same node payload '
             'and text/GINE initialization, with phonetic-node fusion disabled.', '',
             f"Selected node-fusion family: **{selection['variant']}**, by mean validation s4 Hit@1 "
             'and mean MRR for ties, before the ablation. The same family is then retrained '
             'with only the added sound edges masked. “Selected nodes only” therefore means '
             '**phonetic nodes plus the original 15 edges**, not an edgeless graph. All three '
             'seed results are retained; no favorable seed is selected for the aggregate.', '',
             '## Validation results', '',
             '| Variant | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 | Solve rate |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for variant in variants:
        r = aggregate[variant]
        lines.append(f"| {variant} | {r['ranking']['s4']['hit1']:.2%} | {r['ranking']['s4']['mrr']:.4f} | "
                     f"{r['ranking']['s4']['hit4']:.2%} | {r['ranking']['s3']['hit1']:.2%} | "
                     f"{r['ranking']['s2']['hit1']:.2%} | {r['harness']['solve_rate']:.2%} |")
    lines += ['', 's3/s2 ranking uses oracle reduced states, not feedback trajectories. Harness '
              'evaluations use the same 121 validation puzzles, beam 20, full-board scoring, '
              'four mistakes, shuffle seed 0 and existing public-feedback pruning. The normal '
              'checkpoint loader reconstructed each model and its original features for replay.', '',
              '| Seed | Variant | s4 Hit@1 | Solves /121 | Selected update |', '|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | "
                     f"{r['harness']['solved']} | {r['best_update']} |")
    lines += ['', '## Group diagnostics', '',
              '| Variant | Yellow group Top4 | Green | Blue | Purple | Mean known-word gate |',
              '|---|---:|---:|---:|---:|---:|']
    for variant in variants:
        r = aggregate[variant]
        gate = 'disabled' if r['mean_known_word_gate'] is None else f"{r['mean_known_word_gate']:.3f}"
        lines.append(f"| {variant} | " + ' | '.join(f"{r['color_top4'][c]:.2%}" for c in
                    ('yellow', 'green', 'blue', 'purple')) + f' | {gate} |')
    lines += ['', 'Group Top4 concerns each specific true group; state Hit@1 accepts any true '
              'group. A gate average near its initial value does not prove individual gates are '
              'constant or that the projected contribution is zero.', '',
              '| Variant | Yellow recovered /363 | Green | Blue | Purple |', '|---|---:|---:|---:|---:|']
    for variant in variants:
        lines.append(f'| {variant} | ' + ' | '.join(str(aggregate[variant]['recovered_colors'][c])
                     for c in ('yellow', 'green', 'blue', 'purple')) + ' |')
    if group_ranks:
        lines += ['', '| Variant | Sound-labelled group Top4 | Sound-labelled MRR |', '|---|---:|---:|']
        for variant in variants:
            r = group_ranks['aggregate'][variant]['sound']
            lines.append(f"| {variant} | {r['top4']:.2%} | {r['mrr']:.4f} |")
        lines += ['', 'There are 16 label-audited sound groups per seed. Repeated measurements '
                  'are correlated; the labels classify diagnostic rows only and never enter training. '
                  'Full per-group ranks are in `group-ranks.json`.']
        lines += ['', '| Variant | Mean phonetic contribution norm / text norm |', '|---|---:|']
        for variant in variants:
            value = aggregate[variant]['phonetic_contribution_norm_ratio']
            lines.append(f"| {variant} | {value:.4f} |" if value is not None else f'| {variant} | — |')
        lines += ['', 'Contribution ratios are averaged over known words within each original '
                  'validation board, then across boards/seeds. They describe use of the branch, '
                  'not causal feature importance or an accuracy gain.']
    lines += ['', '## Coverage and cost', '',
              '| Split | Distinct words | Known pronunciation | Variant cap hit | Phone-length truncation |',
              '|---|---:|---:|---:|---:|']
    for split, r in coverage['splits'].items():
        lines.append(f"| {split} | {r['unique_words']} | {r['known']} | {r['capped_variants']} | {r['truncated_words']} |")
    lines += ['', '| Variant | Trainable parameters | Warm-cache mean game seconds |', '|---|---:|---:|']
    for variant in variants:
        r = aggregate[variant]
        lines.append(f"| {variant} | {r['trainable_parameters']:,} | {r['harness']['mean_seconds']:.3f} |")
    lines += ['', 'The CNN runs once per word before candidates are enumerated; its output is '
              'reused within each forward. Resource and node payload caching is shared with the '
              'harness. Timings depend on guesses/states, cache warmth and system load; they are '
              'not a controlled inference benchmark or a claim about cold-start latency. '
              'Parameter counts use `requires_grad`; the edge-only row includes the disabled '
              'phonetic projection, which receives no gradients.', '',
              '## Findings', '',
              'The selected fixed n-gram + linear fusion is essentially tied with edge-only: '
              '35.54% versus 35.26% s4 Hit@1 and 25.62% versus 25.34% solves. That amounts '
              'to one additional correct original-board top choice and one additional solve '
              'across 363 repeated seed/puzzle evaluations, not a compelling general gain.', '',
              'CNN + linear fusion has the highest mean harness solve rate, 26.72%, with '
              'paired solve counts 31/35/31 versus 28/32/32 for edge-only. The mean gain is '
              '1.38 percentage points and the third seed worsens. Its Hit@1 drops to 33.33%; '
              'MRR/Hit@4 also decline. This is a possible partition-search trade-off, not '
              'evidence of uniformly better subset decisions. Gating did not improve solve '
              'rate for either representation.', '',
              'The preselected n-gram-linear node-only ablation gets 34.71% Hit@1 and '
              '25.34% solves, versus 35.54%/25.62% with sound edges. This gives little evidence '
              'for a material combined-modality benefit. The ablation is for the family '
              'selected on Hit@1; CNN-linear did not receive its own node-only comparison, '
              'so its harness gain cannot establish node/edge complementarity.', '',
              'Sound-labelled group Top4 remains 6.25% for edge-only and both n-gram '
              'fusions; it is 2.08% for CNN-linear, 4.17% for CNN-gated and zero for the '
              'selected node-only arm. Thus the modest overall differences are not a '
              'demonstrated improvement in the intended sound-group retrieval task.', '',
              'The branches are trainable and used: mean known-word gates are approximately '
              '0.56 for gated n-grams and 0.79 for gated CNNs. The phonetic contribution '
              'norm averages about 2.7% of the text norm for linear n-grams and 6.3% for '
              'linear CNNs, versus 1.9%/1.7% with gates. These are descriptive measurements; '
              'they do not prove why a variant succeeds or fails. Zero-start fusion, relative '
              'scaling, mean pooling of pronunciation alternatives and fixed baseline '
              'optimization settings remain limitations, not isolated causes.', '',
              'Keep the existing notebook baseline. Retain the new variants as reproducible '
              'experiments; the CNN-linear harness result is worth confirming before '
              'promoting it. No test evaluation or further tuning was used to resolve these '
              'small validation differences.', '',
              '## Limits and reproducibility', '',
              'The same validation puzzles select checkpoints and the fusion family. Fresh seeds '
              'measure training variability, not new held-out data; selected-family results are '
              'exploratory and may be optimistic. No test puzzles were evaluated. The notebook '
              'baseline and default feature configuration remain unchanged; phonetic nodes are '
              'opt-in with `phonetic_nodes=True`. Saved models use the shared registry and can '
              'be loaded by Streamlit without architecture-specific UI changes.', '',
              'Run: `nix develop -c uv run python scripts/run_phonetic_fusion_experiment.py`. '
              'Report: `nix develop -c uv run python scripts/summarize_phonetic_fusion.py`. '
              'Manifest, selection, progress, all per-seed metrics, checkpoint paths and replay '
              'histories: [summary.json](gine-phonetic-fusion/summary.json). '
              'Full-budget validation curves: [learning-curves.html](gine-phonetic-fusion/learning-curves.html).']
    Path('experiments/gine-phonetic-fusion.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
