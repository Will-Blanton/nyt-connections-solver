"""Report the bounded study separately from its matched fresh-seed reruns."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev

import altair as alt
import optuna


def main(directory=Path('experiments/gine-tuning')):
    manifest = json.loads((directory / 'manifest.json').read_text())
    best = json.loads((directory / 'best.json').read_text())
    study = optuna.load_study(study_name='gine_basic_20261025',
                              storage=f"sqlite:///{(directory / 'study.sqlite3').resolve()}")
    trial_rows = [dict(trial=t.number, status=t.state.name, value=t.value, params=t.params,
                       updates=t.user_attrs.get('updates_ran')) for t in study.trials]
    final = []; plot_rows = []
    for item in json.loads((directory / 'final-runs.json').read_text()):
        result = json.loads(Path(item['result']).read_text())
        harness = json.loads(Path(item['harness']).read_text())
        final.append(dict(model=item['model'], seed=item['seed'], params=result['params'],
                          checkpoint=result['checkpoint'], selected_update=result['training']['best_update'],
                          trainable_parameters=result['trainable_parameters'],
                          ranking=result['training']['validation'], harness=harness['summary'],
                          recovered_colors=dict(Counter(c for game in harness['games'] for c in game['solved_colors']))))
        running = 0
        for row in result['training']['history']:
            running = max(running, row['val_s4_hit1'])
            plot_rows.append(dict(model=item['model'], seed=item['seed'], update=row['update'],
                                  hit1=row['val_s4_hit1'], best_hit1=running, loss=row['train_loss']))
    aggregate = {}
    for model in ('baseline', 'tuned'):
        rows = [r for r in final if r['model'] == model]
        aggregate[model] = dict(
            ranking={s: {m: mean(r['ranking'][s][m] for r in rows) for m in ('hit1', 'mrr', 'hit4')}
                     for s in ('s4', 's3', 's2')},
            s4_hit1_seed_std=stdev(r['ranking']['s4']['hit1'] for r in rows),
            harness={m: mean(r['harness'][m] for r in rows) for m in
                     ('solved', 'solve_rate', 'first_guess_accuracy', 'mean_correct_groups', 'mean_seconds')},
            recovered_colors={c: sum(r['recovered_colors'].get(c, 0) for r in rows)
                              for c in ('yellow', 'green', 'blue', 'purple')},
            group_top4={c: mean(r['ranking']['s4']['by_color'][c]['top4'] for r in rows)
                        for c in ('yellow', 'green', 'blue', 'purple')})
    summary = dict(manifest=manifest, selected=best, trial_counts=dict(Counter(r['status'] for r in trial_rows)),
                   trials=trial_rows, aggregate=aggregate, final=final)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    data = alt.Data(values=plot_rows)
    curves = alt.Chart(data).mark_line().encode(
        x='update:Q', y=alt.Y('best_hit1:Q', title='Best validation s4 Hit@1 so far'), color='model:N',
        tooltip=['model:N', 'seed:N', 'update:Q', 'best_hit1:Q'])
    curves.properties(width=380, height=220).facet(facet='seed:N', columns=2).save(directory / 'rerun-curves.html')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
