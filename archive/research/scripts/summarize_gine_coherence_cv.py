"""Write live/final reports for five-fold GINE/coherence tuning."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev

import optuna


def purple_status(game):
    accepted = 0
    for guess in game['history']:
        if guess['outcome'] == 'correct':
            if guess['color'] == 'purple':
                return 'forced' if accepted == 3 else 'selected'
            accepted += 1
    return 'missed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/gine-coherence-cv'))
    root = parser.parse_args().output
    manifest = json.loads((root/'manifest.json').read_text())
    status = json.loads((root/'status.json').read_text())
    summary = dict(manifest=manifest, status=status, studies={}, final={})
    lines = ['# Five-fold tuning: ordinary GINE versus edge coherence', '',
        f"Status: **{status['status']}**. This note is regenerated after each trial and at completion.", '',
        '## Protocol', '',
        f"{manifest['pool_puzzles']} puzzles from the original training + validation pool. "
        'Five deterministic puzzle-date folds; all derived states stay in their puzzle fold. '
        'Historical test puzzles are excluded. Each puzzle is held out exactly once per configuration.', '',
        f"Equal budgets of {manifest['trials_per_arm']} configurations per approach. "
        f"Every completed fold receives {manifest['updates_per_fold']:,} updates, batch 16, "
        f"balanced s2/s3/s4 exposure, evaluation every {manifest['eval_every']} updates. "
        'No within-fold early stopping, pruning, LR scheduler, or extra losses. '
        'Fixed probability-mass plus floor loss (.1 weight, .1 beta). '
        'Six complete startup configurations per arm; median pruning only after at least '
        'three fully trained folds of subsequent configurations, compared at the same fold count.', '',
        'All active learned parameters, including text projection and LayerNorm, initialize '
        'fresh for each fold; no learned checkpoint weights are loaded. Only the original '
        'Qwen embeddings are frozen. Models use the same original 15 edge channels and '
        'candidate-local GINE. The reference checkpoint supplies feature definitions, '
        'architecture metadata, and original split dates only. Folds use matched model '
        'seeds, schedules and per-size minibatch orders across approaches.', '',
        '## Search space', '',
        '| Parameter | Both approaches |', '|---|---|',
        '| GINE width | 32, 64, 128 |', '| GINE depth | 1–3 |',
        '| Text projection width | 128, 256 |', '| Readout width | 64, 128, 256 |',
        '| Dropout | 0, .1, .2, .3, .4 |', '| Learning rate | 3e-5–1e-3, logarithmic |',
        '| AdamW weight decay | 1e-6–1e-2, logarithmic |',
        '| Coherence strength | Baseline: 0; amplified: .1–3, logarithmic |', '',
        'Amplification is `edge_ij,k × (1 + strength × coherence_k)`, with coherence '
        'the weakest word’s average off-diagonal edge value to the other three words. '
        'No channels or edges are removed, and no new trainable parameters are added.', '',
        '## Search progress', '',
        '| Approach | Complete configurations | Pruned | Running | Best mean fold Hit@1 | Fold SD |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in manifest['arms']:
        study = optuna.load_study(study_name=arm, storage=f'sqlite:///{(root/"study.sqlite3").resolve()}')
        counts = Counter(t.state.name for t in study.trials)
        data = dict(states=dict(counts))
        best_text, deviation = 'pending', 'pending'
        if counts['COMPLETE']:
            best = study.best_trial
            scores = best.user_attrs['fold_scores']
            assert len(scores) == 5
            data.update(best_trial=best.number, mean_hit1=best.value,
                        fold_hit1=scores, fold_sd=stdev(scores), params=best.params)
            best_text, deviation = f'{best.value:.2%}', f'{stdev(scores):.2%}'
        summary['studies'][arm] = data
        lines.append(f"| {arm} | {counts['COMPLETE']} | {counts['PRUNED']} | {counts['RUNNING']} | {best_text} | {deviation} |")
    for arm, data in summary['studies'].items():
        if 'params' in data:
            lines += ['', f'### Current best: {arm}', '',
                      f"Trial {data['best_trial']}; fold Hit@1: "+', '.join(f'{v:.2%}' for v in data['fold_hit1'])+'.', '',
                      '```json', json.dumps(data['params'], indent=2), '```']
    entries = json.loads((root/'final-runs.json').read_text()) if (root/'final-runs.json').exists() else []
    by_arm = {}
    for entry in entries:
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        expected = set(manifest['folds'][entry['fold']]['val'])
        assert harness['split'] == 'val'
        assert {g['date'] for g in harness['games']} == expected
        assert len(harness['games']) == len(expected)
        assert abs(entry['audited_hit1']-result['training']['validation']['s4']['hit1']) < 1e-8
        by_arm.setdefault(entry['arm'], []).append((entry, result, harness))
    for arm, rows in by_arm.items():
        games = [g for _,_,h in rows for g in h['games']]
        assert len({g['date'] for g in games}) == len(games)
        n = len(games)
        summary['final'][arm] = dict(folds_evaluated=len(rows), puzzles=n,
            solves=sum(g['won'] for g in games), first_correct=sum(g['first_guess_correct'] for g in games),
            colors=dict(Counter(c for g in games for c in g['solved_colors'])),
            purple=dict(Counter(purple_status(g) for g in games)),
            ranking={s: {k: sum(r['training']['validation'][s][k]*len(h['games']) for _,r,h in rows)/n
                for k in ('hit1','mrr','hit4')} for s in ('s4','s3','s2')})
    if all(len(by_arm.get(arm, [])) == 5 for arm in manifest['arms']):
        games = {arm: {g['date']:g for _,_,h in rows for g in h['games']} for arm,rows in by_arm.items()}
        a, b = games['baseline'], games['coherence']
        assert a.keys() == b.keys() and len(a) == manifest['pool_puzzles']
        summary['paired'] = dict(solve_gain=sum(b[d]['won'] and not a[d]['won'] for d in a),
            solve_loss=sum(a[d]['won'] and not b[d]['won'] for d in a),
            purple_direct_gain=sum(purple_status(b[d])=='selected' and purple_status(a[d])!='selected' for d in a),
            purple_direct_loss=sum(purple_status(a[d])=='selected' and purple_status(b[d])!='selected' for d in a))
        lines += ['', '## Selected configurations: out-of-fold harness', '',
            'Each puzzle below is scored by a model that did not train on that puzzle. '
            'Hyperparameters and checkpoint updates were selected using these same folds; '
            'these are selection-affected CV results, not an independent final test.', '',
            '| Metric | Ordinary GINE | Coherence |', '|---|---:|---:|']
        for s in ('s4','s3','s2'):
            for k in ('hit1','mrr','hit4'):
                lines.append(f'| {s} {k} | '+' | '.join(f"{summary['final'][arm]['ranking'][s][k]:.4f}" for arm in manifest['arms'])+' |')
        for k in ('solves','first_correct'):
            lines.append(f'| Harness {k} | '+' | '.join(
                f"{summary['final'][arm][k]}/{len(a)} ({summary['final'][arm][k]/len(a):.2%})" for arm in manifest['arms'])+' |')
        for c in ('yellow','green','blue','purple'):
            lines.append(f'| {c} recovered | '+' | '.join(str(summary['final'][arm]['colors'].get(c,0)) for arm in manifest['arms'])+' |')
        for k in ('selected','forced'):
            lines.append(f'| Purple {k} | '+' | '.join(str(summary['final'][arm]['purple'].get(k,0)) for arm in manifest['arms'])+' |')
        lines += ['', '### Fold results', '',
            '| Fold | Approach | Puzzles | Hit@1 | Solves | Purple direct / forced | Selected update |',
            '|---|---|---:|---:|---:|---:|---:|']
        for arm, rows in by_arm.items():
            for entry,result,harness in rows:
                counts = Counter(purple_status(g) for g in harness['games'])
                lines.append(f"| {entry['fold']} | {arm} | {len(harness['games'])} | "
                    f"{result['training']['validation']['s4']['hit1']:.2%} | "
                    f"{sum(g['won'] for g in harness['games'])} | {counts['selected']} / {counts['forced']} | "
                    f"{result['training']['best_update']} |")
        lines += ['', 'Paired changes: '+json.dumps(summary['paired'])+'.', '']
        # Generate interactive curves for selected configurations without loading models.
        import altair as alt
        import pandas as pd
        curve_rows = []
        for arm, rows in by_arm.items():
            for entry,result,_ in rows:
                for row in result['training']['history']:
                    for metric in ('train_loss','val_s4_hit1','val_s3_hit1','val_s2_hit1'):
                        curve_rows.append(dict(arm=arm, fold=str(entry['fold']), update=row['update'],
                                               metric=metric, value=row[metric]))
        alt.Chart(pd.DataFrame(curve_rows)).mark_line().encode(
            x='update:Q', y=alt.Y('value:Q', scale=alt.Scale(zero=False)), color='arm:N', strokeDash='fold:N',
            tooltip=['arm','fold','update','metric','value']).properties(width=650,height=220).facet(
                row='metric:N').resolve_scale(y='independent').interactive().save(root/'learning-curves.html')
    lines += ['', '## Interpretation limits', '',
        'Cross-validation reduces dependence on a single puzzle split, but folds overlap in '
        'their training data and the selected scores are optimistic because the folds also '
        'guide checkpoint and hyperparameter selection. This is not nested cross-validation. '
        'Do not treat five fold estimates as five independent experiments or compare these '
        'fresh-initialization scores directly with earlier warm-started single-split results. '
        'Reserve newly ingested chronological puzzles for a locked final comparison.', '',
        'Harness settings are fixed: beam 20, original full-board quartet score sums, four '
        'mistakes and shuffle seed 0. No harness scores enter hyperparameter selection. '
        'Reduced-state s2/s3 metrics use oracle remaining groups. No notebooks changed.', '',
        'Resume: `nix develop -c uv run python -u scripts/tune_gine_coherence_cv.py --background`. '
        'Completed folds are reused; an interrupted partial fold restarts with its original seed. '
        'Progress: `progress.json`; worker status: `status.json`; log: `run.log`. '
        'Final harness results, checkpoint audits and this report are generated automatically.', '']
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root/'report.md').write_text('\n'.join(lines))
    print(status['status'], {arm: data.get('mean_hit1') for arm,data in summary['studies'].items()})


if __name__ == '__main__':
    main()
