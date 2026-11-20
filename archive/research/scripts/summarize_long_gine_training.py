"""Compare early and later checkpoints within the same training trajectories."""
import json
import argparse
from pathlib import Path
from statistics import mean


def summarize(directory=Path('experiments/long-training'), variant='standalone-gine-scratch-projected'):
    runs = []
    for path in sorted(directory.glob(f'*{variant}.json')):
        run = json.loads(path.read_text())
        history = run['training']['history']
        select = lambda rows: max(rows, key=lambda r: (r['val_s4_hit1'], r['val_s4_mrr']))
        early = select([r for r in history if r['update'] <= 3000])
        later = select(history)
        runs.append({'seed': run['config']['seed'], 'source': str(path),
                     'within_3000': early, 'within_10000': later,
                     'final': history[-1], 'history': history})
    runs.sort(key=lambda r: r['seed'])
    if not runs:
        raise ValueError(f'No {variant} training results in {directory}')
    summary = {'runs': [{k: v for k, v in r.items() if k != 'history'} for r in runs],
               'means': {budget: {metric: mean(r[budget][metric] for r in runs)
                                  for metric in ('val_s4_hit1', 'val_s4_mrr', 'val_s4_hit4',
                                                 'val_s3_hit1', 'val_s2_hit1')}
                         for budget in ('within_3000', 'within_10000')}}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import altair as alt
    charts = []
    rule = alt.Chart(alt.Data(values=[{'update': 3000}])).mark_rule(strokeDash=[5, 5]).encode(x='update:Q')
    for run in runs:
        points = []; losses = []; best = 0
        for point in run['history']:
            best = max(best, point['val_s4_hit1'])
            for label, value in [('Validation Hit@1', point['val_s4_hit1']), ('Best so far', best)]:
                points.append({'update': point['update'], 'metric': label, 'value': value})
            losses.append({'update': point['update'], 'loss': point['rank_loss']})
        ranking = alt.Chart(alt.Data(values=points)).mark_line().encode(
            x='update:Q', y=alt.Y('value:Q', title='s4 Hit@1'), color='metric:N',
            tooltip=['update:Q', 'metric:N', 'value:Q'])
        loss = alt.Chart(alt.Data(values=losses)).mark_line().encode(
            x='update:Q', y=alt.Y('loss:Q', title='Mixed-state training loss'), tooltip=['update:Q', 'loss:Q'])
        charts.append(alt.hconcat((ranking + rule).properties(width=420, height=180, title=f"Seed {run['seed']}"),
                                 (loss + rule).properties(width=420, height=180)))
    alt.vconcat(*charts).save(directory / 'learning-curves.html')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('experiments/long-training'))
    parser.add_argument('--variant', default='standalone-gine-scratch-projected')
    args = parser.parse_args()
    summarize(args.directory, args.variant)
