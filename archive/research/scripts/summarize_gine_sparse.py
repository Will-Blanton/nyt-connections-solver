"""Summarize matched sparse connectivity experiments on validation puzzles."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from summarize_gine_group_contrastive import purple_status

ROOT = Path('experiments/gine-sparse')
VARIANTS = ('complete', 'sparse-q90', 'sparse-q99')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    connectivity = json.loads((ROOT / 'connectivity.json').read_text())
    entries = json.loads((ROOT / 'runs.json').read_text())
    assert len(entries) == 9
    runs, games_by_key = [], {}
    for entry in entries:
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        assert result['status'] == 'complete' and harness['split'] == 'val'
        assert result['training']['updates_ran'] == 15000
        games = harness['games']
        assert len(games) == len({g['date'] for g in games}) == 121
        games_by_key[entry['variant'], entry['seed']] = {g['date']: g for g in games}
        runs.append(dict(**entry, checkpoint=result['checkpoint'], ranking=result['training']['validation'],
                         best_update=result['training']['best_update'],
                         solves=sum(g['won'] for g in games), first_correct=sum(g['first_guess_correct'] for g in games),
                         seconds=mean(g['seconds'] for g in games),
                         colors=dict(Counter(c for g in games for c in g['solved_colors'])),
                         purple=dict(Counter(purple_status(g) for g in games))))
    aggregate, paired = {}, {}
    for variant in VARIANTS:
        rows = [r for r in runs if r['variant'] == variant]
        assert len(rows) == 3
        aggregate[variant] = dict(
            ranking={s: {k: mean(r['ranking'][s][k] for r in rows) for k in ('hit1', 'mrr', 'hit4')}
                     for s in ('s4', 's3', 's2')},
            solves=sum(r['solves'] for r in rows), first_correct=sum(r['first_correct'] for r in rows),
            seconds=mean(r['seconds'] for r in rows),
            color_recovery={c: sum(r['colors'].get(c, 0) for r in rows) for c in ('yellow','green','blue','purple')},
            purple={k: sum(r['purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')})
        if variant != 'complete':
            counters = Counter()
            for seed in manifest['seeds']:
                a, b = games_by_key['complete', seed], games_by_key[variant, seed]
                assert a.keys() == b.keys()
                for date in a:
                    counters['solve_gain'] += b[date]['won'] and not a[date]['won']
                    counters['solve_loss'] += a[date]['won'] and not b[date]['won']
                    p, q = purple_status(a[date]), purple_status(b[date])
                    counters['purple_gain'] += p == 'missed' and q != 'missed'
                    counters['purple_loss'] += q == 'missed' and p != 'missed'
            paired[variant] = dict(counters)
    audit_path = ROOT / 'probability-audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for variant in VARIANTS:
            boards = [r for r in audit['records'] if r['variant'] == variant]
            assert len(boards) == 363
            assert abs(mean(1 in r['gold_ranks'] for r in boards) - aggregate[variant]['ranking']['s4']['hit1']) < 1e-8
            aggregate[variant]['color_top4'] = {c: mean(r['gold_ranks'][i] <= 4 for r in boards) for i,c in enumerate(audit['colors'])}
    summary = dict(manifest=manifest, connectivity=connectivity, aggregate=aggregate, runs=runs, paired=paired)
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Sparse candidate-local GINE', '',
        'Hard edge pruning hurt this baseline. Keep complete connectivity: '
        f"{aggregate['complete']['solves']}/363 solves versus "
        f"{aggregate['sparse-q90']['solves']}/363 with moderate sparsity and "
        f"{aggregate['sparse-q99']['solves']}/363 with strict sparsity. "
        'Both sparse variants lose harness solves in every seed.', '',
        'Validation only: 121 puzzles, three matched initialization seeds (10/11/12). '
        'Six new runs, each with 15,000 updates; completed complete-graph controls reused. '
        'The same validation puzzles recur across seeds and prior experiments. '
        '363 appearances are not 363 independent puzzles. No notebooks or test-set evaluation.', '',
        '## Controlled change', '',
        'At every GINE layer, remove the whole incoming message for excluded pairs, including '
        'its node contribution. Zeroing edge features alone would not do this. Preserve the '
        'usual self term, learned epsilon, sum aggregation, active projections and readout. '
        'Do not renormalize degrees or add a nearest-neighbor fallback.', '',
        'Fit separate channel thresholds at the 90th and 99th percentiles of all '
        'off-diagonal unordered training-board pairs (one copy per full board). '
        'Keep an edge when at least one channel is positive and at least its threshold. '
        'Retain discrete ties, including binary evidence. Therefore these are channel '
        'quantiles, not target graph densities. No labels or validation features enter '
        'threshold fitting. Thresholds are fixed across candidates/states, saved in model '
        'config, and loaded by the normal harness/Streamlit checkpoint abstraction.', '',
        'Architecture: candidate-local standalone GINE, hidden 128, two layers, readout 128, '
        '15 original edge channels and frozen Qwen vectors; active LayerNorm, text projection, '
        'GINE and readout trainable. There is no global message-passing encoder in this baseline. '
        'The mask is deterministic but aggregation remains learned. Same initialization, '
        'batch permutations and balanced schedules (5,000 updates each on 8/12/16-word states). '
        'Batch 16, LR 0.0003016685, weight decay 0.0000950089, dropout zero; '
        'current probability-mass plus floor loss (weight 0.1, beta 0.1) unchanged. '
        'Evaluate every 250 updates; restore best s4 Hit@1 checkpoint with MRR tie-break. '
        'No early stopping, pruning, scheduling or extra tuning.', '',
        'Harness: beam 20, full-board score sums, four mistakes, shuffle seed 0 and '
        'normal public-feedback pruning. Same selected checkpoint for ranking and harness.', '',
        '## Connectivity diagnostics', '',
        '| Split | Variant | All pairs retained | Within true groups | Across true groups | Isolated board nodes |',
        '|---|---|---:|---:|---:|---:|']
    for split, variants in connectivity['splits'].items():
        for variant, values in variants.items():
            lines.append('| '+split+' | '+variant+' | '+' | '.join(f'{values[k]:.2%}' for k in ('pair_fraction','true_group_pair_fraction','cross_group_pair_fraction','isolated_node_fraction'))+' |')
    lines += ['', 'True-group labels above are used only to diagnose fitted masks; '
              'they do not determine connectivity. Isolated-node rates above use the full board; '
              'a node can still have no neighbors within a particular candidate quartet.', '',
              '| Validation group color | q90 pair retention | q99 pair retention |', '|---|---:|---:|']
    for i,c in enumerate(('purple','blue','green','yellow')):
        lines.append('| '+c+' | '+' | '.join(f"{connectivity['splits']['val'][v]['per_color_retained'][i]:.2%}" for v in VARIANTS[1:])+' |')
    lines += ['', '## Results', '', '| Metric | Complete | Sparse q90 | Sparse q99 |', '|---|---:|---:|---:|']
    for s in ('s4','s3','s2'):
        for k in ('hit1','mrr','hit4'):
            lines.append(f'| {s} {k} | '+' | '.join(f"{aggregate[v]['ranking'][s][k]:.4f}" if k=='mrr' else f"{aggregate[v]['ranking'][s][k]:.2%}" for v in VARIANTS)+' |')
    for k in ('solves','first_correct'):
        lines.append(f'| Harness {k} | '+' | '.join(f"{aggregate[v][k]}/363 ({aggregate[v][k]/363:.2%})" for v in VARIANTS)+' |')
    for c in ('yellow','green','blue','purple'):
        lines.append(f'| {c} recovered | '+' | '.join(f"{aggregate[v]['color_recovery'][c]}/363" for v in VARIANTS)+' |')
    for k in ('selected','forced'):
        lines.append(f'| Purple {k} | '+' | '.join(str(aggregate[v]['purple'][k])+'/363' for v in VARIANTS)+' |')
    if 'color_top4' in aggregate['complete']:
        for c in ('yellow','green','blue','purple'):
            lines.append(f'| {c} raw Top4 recall | '+' | '.join(f"{aggregate[v]['color_top4'][c]:.2%}" for v in VARIANTS)+' |')
    lines += ['', '## Per seed', '', '| Seed | Variant | s4 Hit@1 | Solves | Purple selected / forced | Best update |', '|---|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | {r['solves']}/121 | {r['purple'].get('selected',0)} / {r['purple'].get('forced',0)} | {r['best_update']} |")
    lines += ['', '## Paired changes', '']
    for v,p in paired.items():
        lines.append(f"- {v}: {p['solve_gain']} new solves, {p['solve_loss']} lost solves; {p['purple_gain']} new Purple recoveries, {p['purple_loss']} lost Purple recoveries.")
    lines += ['', '## Interpretation', '',
        'The moderate mask retains 64.3% of validation true-group pairs but still '
        'retains 42.8% of cross-group pairs. The strict mask is more selective, '
        'but retains only 25.7% of true-group pairs. Purple is especially exposed: '
        'only 56.5% / 22.7% of its within-group edges survive. Existing hand-built '
        'features do not provide a reliable hard boundary for message passing.', '',
        'Purple raw Top4 recall rises slightly (3.86% to 4.96% / 4.41%), but '
        'direct Purple selections fall from 61 to 47 / 27; total Purple recovery '
        'falls from 106 to 73 / 29. This is not a useful Purple tradeoff for the '
        'current harness. The loss of easy-group ranking and recovery dominates.', '',
        'Keep sparse connectivity opt-in and retain the complete-graph baseline. '
        'This result concerns these two fixed hard masks under matched settings; '
        'it does not rule out learned or relation-specific graph structure. '
        'Removing neighbors also changes GINE message-sum magnitude; degree '
        'normalization, separate relation aggregation and soft message gates '
        'were not tested here. Missing edge evidence should not automatically '
        'be interpreted as an irrelevant connection.', '',
        'Twelve targeted regression tests passed, covering exact complete-mask '
        'equivalence, whole-message removal, surviving self-term gradients, '
        'permutation invariance, threshold union/validation, checkpoint reload, '
        'and existing GINE/training/loss behavior. All six new checkpoints ran '
        'through the normal harness. The independent nine-checkpoint audit '
        'reproduced full-board Hit@1 on all 1,089 board/model evaluations.', '',
        '## Practical limits', '',
        'This implementation masks dense tensors, rather than using a sparse scatter kernel. '
        'It tests the inductive bias and does not promise reduced arithmetic or inference latency. '
        'Feature preparation cost is unchanged. Harness game timings include variable search '
        'work and are not a controlled model microbenchmark. Defaults remain complete connectivity.', '',
        'Reproduce with `nix develop -c uv run python scripts/run_gine_sparse.py`, '
        'then `scripts/audit_gine_probability_floor.py --results experiments/gine-sparse` '
        'and `scripts/summarize_gine_sparse.py` under the same environment. '
        'Full thresholds, mask diagnostics, checkpoints, curves, and per-game histories '
        'are in `experiments/gine-sparse/`.', '']
    Path('experiments/gine-sparse.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
