"""Separate harness-selected search results from matched fresh-seed reruns."""
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, stdev

import altair as alt
import optuna

from summarize_gine_group_contrastive import purple_status


ROOT = Path('experiments/gine-floor-tuning')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    best = json.loads((ROOT / 'best.json').read_text())
    recovery_path = ROOT / 'recovery.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    study = optuna.load_study(study_name=manifest['study_name'], storage=f"sqlite:///{(ROOT/'study.sqlite3').resolve()}")
    trials = []
    for trial in study.trials:
        path = ROOT / 'runs' / f'trial-{trial.number:02d}.json'
        payload = json.loads(path.read_text()) if path.exists() else {}
        trials.append(dict(number=trial.number, status=trial.state.name, solve_rate=trial.value,
                           invalid_pruning=trial.number in recovery.get('invalid_pruning_trials', []),
                           retry_of_trial=trial.user_attrs.get('retry_of_trial'),
                           params=trial.params, updates=trial.user_attrs.get('updates_ran'),
                           selected_hit1=trial.user_attrs.get('selected_hit1'), solved=trial.user_attrs.get('solved'),
                           result=str(path), checkpoint=payload.get('checkpoint')))
    runs, histories, games_by_key = [], [], {}
    for entry in json.loads((ROOT / 'final-runs.json').read_text()):
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        assert result['status'] == 'complete' and result['training']['updates_ran'] == 15000
        assert harness['split'] == 'val' and len(harness['games']) == 121
        games = harness['games']
        games_by_key[entry['variant'], entry['seed']] = {g['date']: g for g in games}
        runs.append(dict(**entry, params=result['params'], checkpoint=result['checkpoint'],
                         selected_update=result['training']['best_update'],
                         trainable_parameters=result['trainable_parameters'], ranking=result['training']['validation'],
                         solves=sum(g['won'] for g in games), first_correct=sum(g['first_guess_correct'] for g in games),
                         harness_seconds=sum(g['seconds'] for g in games),
                         colors=dict(Counter(c for g in games for c in g['solved_colors'])),
                         purple=dict(Counter(purple_status(g) for g in games))))
        for row in result['training']['history']:
            histories.append(dict(variant=entry['variant'], seed=str(entry['seed']), update=row['update'],
                                  hit1=row['val_s4_hit1'], loss=row['train_loss']))
    assert len(runs) == 2*len(manifest['final_seeds'])
    audit_path = ROOT / 'probability-audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    aggregate = {}
    for variant in ('current-floor', 'tuned-floor'):
        rows = [r for r in runs if r['variant'] == variant]
        count = 121*len(rows)
        aggregate[variant] = dict(
            ranking={s: {key: mean(r['ranking'][s][key] for r in rows) for key in ('hit1', 'mrr', 'hit4')}
                     for s in ('s4', 's3', 's2')},
            s4_hit1_seed_std=stdev(r['ranking']['s4']['hit1'] for r in rows),
            appearances=count, solves=sum(r['solves'] for r in rows), solve_rate=sum(r['solves'] for r in rows)/count,
            first_guess_accuracy=sum(r['first_correct'] for r in rows)/count,
            mean_game_seconds=sum(r['harness_seconds'] for r in rows)/count,
            trainable_parameters=rows[0]['trainable_parameters'],
            recovered_colors={c: sum(r['colors'].get(c,0) for r in rows) for c in ('yellow','green','blue','purple')},
            purple={k: sum(r['purple'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            color_top4={c: mean(r['ranking']['s4']['by_color'][c]['top4'] for r in rows)
                        for c in ('yellow','green','blue','purple')})
        if audit:
            boards = [b for b in audit['records'] if b['variant'] == variant]
            assert len(boards) == count
            probabilities = [[math.exp(v) for v in b['gold_log_probabilities']] for b in boards]
            aggregate[variant]['probability'] = dict(correct_mass=mean(sum(p) for p in probabilities),
                other_correct_mass=mean(sum(p)-max(p) for p in probabilities),
                minimum_correct_probability=mean(min(p) for p in probabilities),
                dominant_share=mean(max(p)/sum(p) for p in probabilities),
                groups_above_floor=mean(p>=.025 for row in probabilities for p in row))
            assert abs(mean(1 in b['gold_ranks'] for b in boards)-aggregate[variant]['ranking']['s4']['hit1'])<1e-9
    paired = Counter()
    for seed in manifest['final_seeds']:
        a,b = games_by_key['current-floor',seed], games_by_key['tuned-floor',seed]
        assert a.keys() == b.keys()
        for date in a:
            paired['solve_gain'] += b[date]['won'] and not a[date]['won']
            paired['solve_loss'] += a[date]['won'] and not b[date]['won']
            paired['purple_gain'] += purple_status(a[date])=='missed' and purple_status(b[date])!='missed'
            paired['purple_loss'] += purple_status(b[date])=='missed' and purple_status(a[date])!='missed'
    summary = dict(manifest=manifest, best=best, trial_counts=dict(Counter(t['status'] for t in trials)),
                   trials=trials, aggregate=aggregate, runs=runs, paired=dict(paired), recovery=recovery)
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    chart = alt.Chart(alt.Data(values=histories)).mark_line().encode(
        x='update:Q', y=alt.Y('hit1:Q', title='Validation full-board Hit@1'), color='variant:N',
        tooltip=['variant:N','seed:N','update:Q','hit1:Q','loss:Q'])
    chart.properties(width=400,height=240).facet(facet='seed:N',columns=2).save(ROOT/'fresh-seed-curves.html')
    variants = ('current-floor','tuned-floor')
    lines = ['# GINE probability-floor hyperparameter tuning', '',
             'Validation-only bounded Optuna study followed by matched fresh-seed '
             'retrains. Notebooks and test-set evaluation untouched.', '',
             '## Protocol', '',
             'Sixteen trials at search seed 0; up to 15,000 updates each, batch 16, '
             'evaluate every 250 updates. First six trials finish the full budget. '
             'Subsequent trials can be pruned only from update 8,000, at 1,000-update '
             'intervals, below the 25th percentile of best-so-far s4 Hit@1 across '
             'at least four comparable completed trials. This proxy pruning can '
             'discard a model that would have played better; it is deliberately '
             'conservative. There is no harness run at each epoch/evaluation.', '',
             'Checkpoint selection within a run: validation s4 Hit@1 with MRR '
             'tie-break. Hyperparameter selection: one validation harness replay '
             'of the selected checkpoint per completed trial, maximize solve rate; '
             'ties retain the earliest Optuna trial. Harness: beam 20, full_board '
             'scores, four mistakes, public-feedback pruning, shuffle 0.', '',
             'Search GINE width 64/128/256, depth 1–3, readout width 64/128/256, '
             'dropout 0/0.05/0.1/0.2, LR 3e-5–6e-4, weight decay 1e-6–3e-3, and '
             'floor weight 0.01–0.3 (continuous ranges logarithmic). Fix beta=0.1 '
             'and the same 15 edge features. Frozen Qwen vectors; active norm, '
             'projection, GINE and readout fully trainable. Balanced mixed '
             's2/s3/s4 states, identical seed-specific schedules, no LR schedule '
             'or extra objective. Architecture remains candidate-local GINE.', '',
             'Retrain current-floor and tuned-floor from the same shared text-only '
             'reference on seeds 20,21,22, 15,000 updates each, no pruning or early '
             'stopping. Current-floor uses the previous tuned GINE plus floor '
             'weight 0.1. These are fresh initialization seeds, not fresh puzzles: '
             'all evaluations reuse the same 121 validation puzzles, which have '
             'been used repeatedly for selection. Treat tuning results as optimistic '
             'and seed reruns as a stability check, not independent generalization.', '',
             '## Selected configuration', '',
             f"Trial {best['trial']}, search-seed harness solve rate {best['value']:.2%}.", '',
             '```json', json.dumps(best['params'],indent=2), '```', '',
             'Trial states: '+json.dumps(summary['trial_counts'])+'.', '',
             'A pruning API interpretation was corrected during the study: Optuna '
             'keeps the specified percentile of top trials, so keeping 75% implements '
             'the intended bottom-quartile pruning. Trial 7 was incorrectly pruned '
             'and retried; trial 8 was interrupted to apply the fix and retried. '
             'Trial 6 would still prune under the corrected cutoff. No completed '
             'trial was discarded. Recovery metadata and both original and retry '
             'records are preserved; the invalid pruning and interrupted trial '
             'do not consume the sixteen-valid-trial budget. Boundary regression '
             'tests were strengthened and passed before resuming.', '',
             '## Matched fresh-seed results', '',
             '| Metric | Current floor loss | Tuned floor loss |', '|---|---:|---:|']
    for size in ('s4','s3','s2'):
        for key in ('hit1','mrr','hit4'):
            cells=[f"{aggregate[v]['ranking'][size][key]:.4f}" if key=='mrr' else
                   f"{aggregate[v]['ranking'][size][key]:.2%}" for v in variants]
            lines.append(f'| {size} {key} | '+' | '.join(cells)+' |')
    lines.append('| Harness solves | '+' | '.join(f"{aggregate[v]['solves']}/{aggregate[v]['appearances']} ({aggregate[v]['solve_rate']:.2%})" for v in variants)+' |')
    lines.append('| Harness first-guess accuracy | '+' | '.join(f"{aggregate[v]['first_guess_accuracy']:.2%}" for v in variants)+' |')
    lines.append('| Mean harness seconds per game | '+' | '.join(f"{aggregate[v]['mean_game_seconds']:.3f}" for v in variants)+' |')
    lines.append('| Trainable parameters | '+' | '.join(f"{aggregate[v]['trainable_parameters']:,}" for v in variants)+' |')
    for color in ('yellow','green','blue','purple'):
        lines.append(f'| {color} recovered | '+' | '.join(str(aggregate[v]['recovered_colors'][color])+'/363' for v in variants)+' |')
    for k in ('selected','forced'):
        lines.append(f'| Purple {k} | '+' | '.join(str(aggregate[v]['purple'][k])+'/363' for v in variants)+' |')
    lines += ['', '## Per seed', '', '| Seed | Configuration | s4 Hit@1 | Solves | Purple recovered | Selected update |',
              '|---|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | {r['solves']}/121 | "
                     f"{r['colors'].get('purple',0)}/121 | {r['selected_update']} |")
    lines += ['', f"Paired changes: {paired['solve_gain']} new solves, {paired['solve_loss']} lost solves; "
              f"{paired['purple_gain']} new Purple recoveries, {paired['purple_loss']} lost Purple recoveries.", '',
              '## Search trials', '', '| Trial | Status | Updates | Selected Hit@1 | Harness solves |', '|---|---|---:|---:|---:|']
    for t in trials:
        hit='—' if t['selected_hit1'] is None else f"{t['selected_hit1']:.2%}"
        status = 'INVALID_PRUNING' if t['invalid_pruning'] else t['status']
        if t['retry_of_trial'] is not None:
            status += f" (retry {t['retry_of_trial']})"
        lines.append(f"| {t['number']} | {status} | {t['updates']} | {hit} | {t['solved']} |")
    lines += ['', 'Full trial parameters and checkpoint paths: `gine-floor-tuning/summary.json`; '
              'Optuna state: `study.sqlite3`; learning curves: `fresh-seed-curves.html`. '
              'The existing default objective and notebooks remain unchanged.', '']
    Path('experiments/gine-floor-tuning.md').write_text('\n'.join(lines))
    print(json.dumps(summary['aggregate'],indent=2))


if __name__=='__main__':
    main()
