"""Validation ranking, group coverage, and public-feedback harness comparison."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

from audit_letter_edit_coverage import EXAMPLES


ROOT = Path('experiments/gine-group-contrastive')
VARIANTS = ('probability-mass', 'per-group')


def purple_status(game):
    accepted = 0
    for guess in game['history']:
        if guess['outcome'] == 'correct':
            if guess['color'] == 'purple':
                return 'forced' if accepted == 3 else 'selected'
            accepted += 1
    return 'missed'


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    ranks = json.loads((ROOT / 'group-ranks.json').read_text())
    entries = json.loads((ROOT / 'runs.json').read_text())
    assert len(entries) == 6
    runs, games_by_run = [], {}
    for entry in entries:
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        assert result['status'] == 'complete' and harness['split'] == 'val'
        assert result['training']['updates_ran'] == 15000
        games = harness['games']
        assert len(games) == len({g['date'] for g in games}) == 121
        games_by_run[entry['variant'], entry['seed']] = {g['date']: g for g in games}
        runs.append(dict(**entry, checkpoint=result['checkpoint'], ranking=result['training']['validation'],
                         best_update=result['training']['best_update'],
                         solves=sum(g['won'] for g in games), first_correct=sum(g['first_guess_correct'] for g in games),
                         correct_groups=sum(g['correct_groups'] for g in games),
                         colors=dict(Counter(c for g in games for c in g['solved_colors'])),
                         purple=dict(Counter(purple_status(g) for g in games)),
                         target_purple=dict(Counter(purple_status(g) for g in games if g['date'] in EXAMPLES))))
    aggregate, paired = {}, {}
    for variant in VARIANTS:
        rows = [r for r in runs if r['variant'] == variant]
        group_rows = [r for r in ranks['records'] if r['variant'] == variant]
        boards = defaultdict(list)
        for r in group_rows:
            boards[r['seed'], r['date']].append(r['rank'])
        assert len(group_rows) == 1452 and len(boards) == 363 and all(len(v) == 4 for v in boards.values())
        aggregate[variant] = dict(
            ranking={s: {k: mean(r['ranking'][s][k] for r in rows) for k in ('hit1', 'mrr', 'hit4')}
                     for s in ('s4', 's3', 's2')},
            solves=sum(r['solves'] for r in rows), first_correct=sum(r['first_correct'] for r in rows),
            mean_correct_groups=sum(r['correct_groups'] for r in rows)/363,
            color_recovery={c: sum(r['colors'].get(c, 0) for r in rows) for c in ('yellow', 'green', 'blue', 'purple')},
            purple={k: sum(r['purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')},
            target_purple={k: sum(r['target_purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')},
            all_group_ranking=dict(mean_rank=mean(r['rank'] for r in group_rows),
                                   mrr=mean(1/r['rank'] for r in group_rows),
                                   top4_recall=mean(r['rank'] <= 4 for r in group_rows),
                                   median_rank=median(r['rank'] for r in group_rows),
                                   mean_worst_group_rank=mean(max(v) for v in boards.values()),
                                   all_four_top4=sum(max(v) <= 4 for v in boards.values()),
                                   at_least_three_top4=sum(sum(rank <= 4 for rank in v) >= 3 for v in boards.values())),
            color_ranking={c: dict(mean_rank=mean(r['rank'] for r in group_rows if r['color'] == c),
                                   mrr=mean(1/r['rank'] for r in group_rows if r['color'] == c),
                                   top4=mean(r['rank'] <= 4 for r in group_rows if r['color'] == c))
                           for c in ('yellow', 'green', 'blue', 'purple')})
        assert abs(mean(any(rank == 1 for rank in v) for v in boards.values()) -
                   aggregate[variant]['ranking']['s4']['hit1']) < 1e-9
    counters = Counter()
    purple_changes = []
    for seed in manifest['seeds']:
        a, b = games_by_run['probability-mass', seed], games_by_run['per-group', seed]
        assert a.keys() == b.keys()
        for date in a:
            counters['solve_gain'] += b[date]['won'] and not a[date]['won']
            counters['solve_loss'] += a[date]['won'] and not b[date]['won']
            p, q = purple_status(a[date]), purple_status(b[date])
            counters['purple_gain'] += p == 'missed' and q != 'missed'
            counters['purple_loss'] += q == 'missed' and p != 'missed'
            if p != q:
                purple_changes.append(dict(seed=seed, date=date, baseline=p, per_group=q, targeted=date in EXAMPLES))
    paired = dict(counters)
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs, paired=paired,
                   purple_changes=purple_changes, by_mechanism=ranks['aggregate'])
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Per-correct-group contrastive loss on GINE', '',
             'Validation only: 121 puzzles, three matched seeds (10/11/12), 15,000 updates '
             'each. Reuse completed probability-mass controls from `gine-curriculum/mixed`. '
             'The same validation puzzles are reused across seeds and previous experiments; '
             '363 puzzle-seed appearances are not 363 independent puzzles. No notebooks '
             'or test-set evaluation.', '', '## Controlled change', '',
             'Replace `-log(sum_correct softmax(score))` with the mean, over each correct '
             'quartet g, of `softplus(logsumexp(incorrect scores) - score[g])`. '
             'Each correct group competes with all incorrect groups, excluding other '
             'correct groups from its denominator. Average groups within a state, then '
             'states within a batch. This encourages every correct group to outrank '
             'incorrect groups without imposing equal scores on correct groups. '
             'It remains candidate scoring, not joint differentiable partitioning.', '',
             'Everything else is fixed: standalone candidate-local GINE, widths 128, '
             'two GINE layers, original 15 edges, no phonetics or letter-edit additions, '
             'frozen Qwen vectors, fully trainable active LayerNorm/projection/GINE/readout. '
             'Use the same text-only projection initialization, seeds, batch permutations '
             'and balanced state-size schedules as controls (5,000 updates each at s2/s3/s4). '
             'Batch 16, LR 0.0003016685, weight decay 0.0000950089, dropout 0. '
             'Evaluate every 250 updates, select s4 Hit@1 with MRR tie-break. '
             'No pruning, early stopping, auxiliary objectives, schedules or tuning. '
             'Hyperparameters were previously selected for the probability-mass objective; '
             'this is a fixed-setting objective comparison, not a tuned upper bound.', '',
             'Harness: normal checkpoint loader, beam 20, original full-board group-score '
             'sums, four mistakes, public-feedback pruning, shuffle seed 0.', '',
             '## Aggregate', '', '| Metric | Probability mass | Per-group contrastive |', '|---|---:|---:|']
    for size in ('s4', 's3', 's2'):
        for key in ('hit1', 'mrr', 'hit4'):
            values = [aggregate[v]['ranking'][size][key] for v in VARIANTS]
            cells = [f'{x:.4f}' if key == 'mrr' else f'{x:.2%}' for x in values]
            lines.append(f'| {size} {key} | '+' | '.join(cells)+' |')
    for key in ('solves', 'first_correct'):
        lines.append(f'| Harness {key} | '+' | '.join(f"{aggregate[v][key]}/363 ({aggregate[v][key]/363:.2%})" for v in VARIANTS)+' |')
    lines += ['', '## Coverage of all true groups', '',
              'Full-board quartet ranks use the exact same selected checkpoints as the '
              'headline/harness metrics, never a separately selected checkpoint.', '',
              '| Metric | Probability mass | Per-group contrastive |', '|---|---:|---:|']
    for key in ('mean_rank', 'mrr', 'top4_recall', 'median_rank', 'mean_worst_group_rank', 'all_four_top4', 'at_least_three_top4'):
        values = [aggregate[v]['all_group_ranking'][key] for v in VARIANTS]
        cells = [f'{x:.2%}' if key == 'top4_recall' else f'{x:.3f}' if key == 'mrr' else f'{x:.2f}' for x in values]
        if key in ('all_four_top4', 'at_least_three_top4'):
            cells = [str(x)+'/363' for x in values]
        lines.append(f'| {key} | '+' | '.join(cells)+' |')
    lines += ['', '| Color | Baseline Top4 recall | Per-group Top4 recall | Baseline recovered | Per-group recovered |',
              '|---|---:|---:|---:|---:|']
    for color in ('yellow', 'green', 'blue', 'purple'):
        cells = [color]+[f"{aggregate[v]['color_ranking'][color]['top4']:.2%}" for v in VARIANTS]
        cells += [str(aggregate[v]['color_recovery'][color])+'/363' for v in VARIANTS]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '| Purple detail | Probability mass | Per-group |', '|---|---:|---:|']
    for key in ('selected', 'forced'):
        lines.append('| '+key+' | '+' | '.join(str(aggregate[v]['purple'][key])+'/363' for v in VARIANTS)+' |')
    lines.append('| Targeted letter-edit Purple recovered | '+' | '.join(str(24-aggregate[v]['target_purple']['missed'])+'/24' for v in VARIANTS)+' |')
    lines += ['', '## Per seed', '', '| Seed | Objective | s4 Hit@1 | s3 Hit@1 | s2 Hit@1 | Solves | Purple | Selected update |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        cells=[str(r['seed']),r['variant']]+[f"{r['ranking'][s]['hit1']:.2%}" for s in ('s4','s3','s2')]
        cells += [str(r['solves'])+'/121', str(r['colors'].get('purple', 0))+'/121', str(r['best_update'])]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Paired changes', '',
              f"{paired['solve_gain']} new solves and {paired['solve_loss']} lost solves; "
              f"{paired['purple_gain']} new Purple recoveries and {paired['purple_loss']} lost Purple recoveries.", '',
              'Full metrics and per-date Purple changes: `gine-group-contrastive/summary.json`. '
              'Mechanism labels are analysis-only; see `group-ranks.json` for all 2,904 '
              'true-quartet rank records. The existing feature families and default '
              'objective remain unchanged outside this opt-in experiment.', '']
    lines += ['## Interpretation', '',
              'This replacement objective hurts the current GINE system at the fixed '
              'settings. Full-board Hit@1 falls 6.34 percentage points and harness '
              'solve rate falls 6.06 points (22 fewer solves). Every seed loses both '
              'Hit@1 and solves, so the aggregate decline is not driven by just one '
              'seed. Smaller-state Hit@1 also declines.', '',
              'There is a real change in the ranking distribution: mean true-group '
              'rank improves from 577.95 to 402.96, median from 328 to 184, and '
              'mean worst true-group rank from 1368.50 to 1000.29. However, true-group '
              'MRR and Top4 recall decline. Moving poorly ranked positives upward '
              'does not compensate for the loss of strong top choices.', '',
              'Purple has a modest direct-retrieval gain: Top4 recall rises from '
              '3.58% to 5.23%, and Purple selected before the forced last four rises '
              'from 62 to 69. Total Purple recovery nevertheless falls from 100 to '
              '87 as forced final-group recoveries fall from 38 to 18. The eight '
              'curated letter-edit puzzles fall from 5/24 to 3/24 Purple recoveries. '
              'Those small direct gains do not justify the broader regressions.', '',
              'All runs finish the full 15,000-update budget. Selected updates are '
              '1,750, 1,250, and 8,500; later training continues to reduce training '
              'loss without beating those validation checkpoints. This is not evidence '
              'that an insufficient update budget caused the observed decline.', '',
              'Keep the probability-mass baseline. The result applies to this full '
              'objective replacement with existing hyperparameters; it does not rule '
              'out a separately tuned objective or a weak all-group auxiliary term. '
              'Neither was tested here. Twelve targeted loss/training regression '
              'tests passed. All three new checkpoints loaded and ran through the '
              'normal validation harness, and the group-rank audit independently '
              'reproduced all six runs\' full-board Hit@1.', '']
    Path('experiments/gine-group-contrastive.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
