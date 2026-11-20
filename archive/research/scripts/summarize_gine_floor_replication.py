"""Add the original-seed replication to the floor-tuning note, reproducibly."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev

import summarize_gine_floor_tuning
from summarize_gine_group_contrastive import purple_status


ROOT = Path('experiments/gine-floor-tuning')
REPLICATION = Path('experiments/gine-floor-replication')
VARIANTS = ('current-floor', 'tuned-floor')


def aggregate(rows):
    result = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r['variant'] == variant]
        count = 121*len(selected)
        result[variant] = dict(appearances=count, solves=sum(r['solves'] for r in selected),
            solve_rate=sum(r['solves'] for r in selected)/count,
            seed_solve_rate_std=stdev(r['solves']/121 for r in selected),
            first_guess_accuracy=sum(r['first_correct'] for r in selected)/count,
            ranking={s: {k: mean(r['ranking'][s][k] for r in selected) for k in ('hit1','mrr','hit4')}
                     for s in ('s4','s3','s2')},
            colors={c: sum(r['colors'].get(c,0) for r in selected) for c in ('yellow','green','blue','purple')},
            purple={k: sum(r['purple'].get(k,0) for r in selected) for k in ('selected','forced','missed')})
    return result


def main():
    summarize_gine_floor_tuning.main()
    summary = json.loads((ROOT/'summary.json').read_text())
    original = []
    for entry in json.loads((REPLICATION/'runs.json').read_text()):
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        assert result['status']=='complete' and result['training']['updates_ran']==15000
        assert harness['split']=='val' and len(harness['games'])==121
        games = harness['games']
        original.append(dict(**entry, checkpoint=result['checkpoint'], ranking=result['training']['validation'],
            selected_update=result['training']['best_update'], solves=sum(g['won'] for g in games),
            first_correct=sum(g['first_guess_correct'] for g in games),
            colors=dict(Counter(c for g in games for c in g['solved_colors'])),
            purple=dict(Counter(purple_status(g) for g in games))))
    assert len(original)==6
    rows = original+summary['runs']
    comparisons = dict(original_seeds=aggregate(original), six_seeds=aggregate(rows))
    by_key = {(r['variant'],r['seed']): {g['date']:g for g in json.loads(Path(r['harness']).read_text())['games']}
              for r in rows}
    paired = Counter()
    for seed in (10,11,12,20,21,22):
        a,b = by_key['current-floor',seed],by_key['tuned-floor',seed]
        assert a.keys()==b.keys()
        for date in a:
            paired['solve_gain'] += b[date]['won'] and not a[date]['won']
            paired['solve_loss'] += a[date]['won'] and not b[date]['won']
            paired['purple_gain'] += purple_status(a[date])=='missed' and purple_status(b[date])!='missed'
            paired['purple_loss'] += purple_status(b[date])=='missed' and purple_status(a[date])!='missed'
    summary['replication'] = dict(original_runs=original, **comparisons, paired=dict(paired))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (REPLICATION/'summary.json').write_text(json.dumps(summary['replication'],indent=2)+'\n')
    lines = ['', '## Original-seed replication and six-seed comparison', '',
        'Because the initial fresh-seed gain was concentrated in seed 22, also '
        'train the already selected configuration on original seeds 10/11/12. '
        'Reuse their completed current-floor controls. Three additional full '
        '15,000-update runs; no new search or parameter selection. This permits '
        'a direct comparison with the earlier 26.72% result, which used those '
        'original seeds. All evaluations still reuse the same 121 validation '
        'puzzles; six seeds are not six independently held-out datasets.', '',
        '| Seed | Current solves | Tuned solves | Current Hit@1 | Tuned Hit@1 |',
        '|---|---:|---:|---:|---:|']
    for seed in (10,11,12,20,21,22):
        a,b = [next(r for r in rows if r['seed']==seed and r['variant']==v) for v in VARIANTS]
        lines.append(f"| {seed} | {a['solves']}/121 | {b['solves']}/121 | {a['ranking']['s4']['hit1']:.2%} | {b['ranking']['s4']['hit1']:.2%} |")
    for label,title in (('original_seeds','Original seeds 10–12'),('six_seeds','All six matched seeds')):
        values = comparisons[label]
        lines += ['', f'### {title}', '', '| Metric | Current floor loss | Tuned floor loss |', '|---|---:|---:|']
        lines.append('| Harness solves | '+' | '.join(f"{values[v]['solves']}/{values[v]['appearances']} ({values[v]['solve_rate']:.2%})" for v in VARIANTS)+' |')
        for s in ('s4','s3','s2'):
            lines.append(f'| {s} Hit@1 | '+' | '.join(f"{values[v]['ranking'][s]['hit1']:.2%}" for v in VARIANTS)+' |')
        for c in ('yellow','green','blue','purple'):
            lines.append(f'| {c} recovered | '+' | '.join(str(values[v]['colors'][c]) for v in VARIANTS)+' |')
        lines.append('| Purple before final four | '+' | '.join(str(values[v]['purple']['selected']) for v in VARIANTS)+' |')
    lines += ['', f"Across all six seeds: {paired['solve_gain']} new solves and {paired['solve_loss']} lost solves; "
              f"{paired['purple_gain']} new Purple recoveries and {paired['purple_loss']} lost Purple recoveries.", '',
              'Original-seed checkpoints and histories are listed in '
              '`gine-floor-replication/runs.json`. Combined metrics and per-seed '
              'records are in `gine-floor-tuning/summary.json` under `replication`.', '']
    lines += ['## Conclusion', '',
        'No convincing harness improvement from this bounded tuning stage. '
        'The initial fresh-seed comparison was positive (83 to 92 solves), but '
        'the original-seed replication was negative (97 to 87), leaving 180 '
        'versus 179 solves across six matched seeds. Four of six seeds lose '
        'solves; the positive net result on fresh seeds was largely driven by '
        'seed 22. Retain the current floor-weight-0.1 setup as the baseline.', '',
        'The selected alternative improves standalone ranking: s4 Hit@1 rises '
        '34.85% to 36.91%, s3 37.16% to 38.57%, and s2 43.23% to 44.88%. '
        'However, s4 Hit@4 falls 52.89% to 51.79%, and harness first-guess '
        'accuracy falls 38.15% to 37.47%. The harness selects from complete '
        'partitions, so stronger top standalone choices are not sufficient.', '',
        'Purple does not benefit: total recovery falls 205 to 201, and Purple '
        'selected before the forced final four falls 127 to 112. The selected '
        'architecture is unchanged (width 128, two layers, readout 128, dropout '
        'zero, 400,259 trainable parameters). Changes are LR 5.3111e-4, '
        'weight decay 3.8343e-5 and floor weight 0.01293, with beta still 0.1. '
        'Fresh-seed mean per-game harness time is 0.278 seconds for both '
        'configurations; this includes search with prepared resources, not '
        'model-loading or unseen-word embedding startup.', '',
        'The search selected configurations by harness performance on one '
        'initialization seed, which did not give a stable gain across seeds. '
        'Multiple-seed selection or more validation puzzles could reduce that '
        'selection noise, but neither was tested here. All these evaluations '
        'reuse validation puzzles and do not establish unseen-puzzle performance. '
        'No test-set evaluation or notebook changes. Nine targeted tuning/loss/'
        'GINE regression tests passed after correcting the pruning boundary. '
        'All nine new final/replication retrains completed 15,000 updates.', '']
    report = Path('experiments/gine-floor-tuning.md')
    content = report.read_text().replace('# GINE probability-floor hyperparameter tuning\n',
        '# GINE probability-floor hyperparameter tuning\n\n'
        '**Outcome:** standalone Hit@1 improved, but the six-seed harness result '
        'was unchanged (180/726 current versus 179/726 tuned). Keep the current '
        'floor-loss baseline.\n', 1)
    report.write_text(content+'\n'.join(lines))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
