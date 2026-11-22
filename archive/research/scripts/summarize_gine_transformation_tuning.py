"""Reproduce the validation-only learned-transformation optimization report."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import altair as alt
from summarize_gine_group_contrastive import purple_status

ROOT = Path('experiments/gine-transformation-tuning')
VARIANTS = ('control', 'edges-text', 'original-only', 'tuned', 'tuned-original-only')


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    best = json.loads((ROOT/'best.json').read_text())
    entries = json.loads((ROOT/'runs.json').read_text())
    assert len(entries) == 15
    audit = json.loads((ROOT/'probability-audit.json').read_text())
    assert len(audit['records']) == 1815
    targets = json.loads(Path('experiments/gine-transformations/target-coverage.json').read_text())['records']
    target_dates = {r['date'] for r in targets}
    rows, curves, diag_curves, games_by_key = [], [], [], {}
    for entry in entries:
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        t = result['training']
        assert t['updates_ran'] == manifest['updates'] == 30000
        assert len(t['history']) == 60 and harness['split'] == 'val'
        games = harness['games']
        assert len(games) == len({g['date'] for g in games}) == 121
        games_by_key[entry['variant'], entry['seed']] = {g['date']:g for g in games}
        rows.append(dict(**entry, best_update=t['best_update'], ranking=t['validation'],
            last_hit1=t['history'][-1]['val_s4_hit1'], checkpoint=result['checkpoint'],
            solves=sum(g['won'] for g in games), first_correct=sum(g['first_guess_correct'] for g in games),
            colors=dict(Counter(c for g in games for c in g['solved_colors'])),
            purple=dict(Counter(purple_status(g) for g in games)),
            targeted=dict(Counter(purple_status(g) for g in games if g['date'] in target_dates))))
        for h in t['history']:
            curves.append(dict(variant=entry['variant'], seed=entry['seed'], update=h['update'],
                hit1=h['val_s4_hit1'], loss=h['train_loss']))
        for d in result.get('diagnostics', []):
            for split in ('train', 'validation'):
                diag_curves.append(dict(variant=entry['variant'], seed=entry['seed'], split=split,
                    update=d['update'], loss=d[split]['s4']['loss'], hit1=d[split]['s4']['hit1']))
    aggregates = {}
    for v in VARIANTS:
        runs = [r for r in rows if r['variant']==v]
        ranks = [r for r in audit['records'] if r['variant']==v]
        assert len(runs)==3 and len(ranks)==363
        aggregates[v] = dict(
            ranking={s:{k:mean(r['ranking'][s][k] for r in runs) for k in ('hit1','mrr','hit4')}
                for s in ('s4','s3','s2')},
            solves=sum(r['solves'] for r in runs), first_correct=sum(r['first_correct'] for r in runs),
            colors={c:sum(r['colors'].get(c,0) for r in runs) for c in ('yellow','green','blue','purple')},
            purple={k:sum(r['purple'].get(k,0) for r in runs) for k in ('selected','forced','missed')},
            targeted={k:sum(r['targeted'].get(k,0) for r in runs) for k in ('selected','forced','missed')})
        assert abs(mean(1 in r['gold_ranks'] for r in ranks)-aggregates[v]['ranking']['s4']['hit1'])<1e-9
    paired={}
    for left,right in (('edges-text','tuned'),('original-only','tuned-original-only'),
                       ('control','tuned'),('tuned-original-only','tuned')):
        counter=Counter()
        for seed in (10,11,12):
            a,b=games_by_key[left,seed],games_by_key[right,seed]
            assert a.keys()==b.keys()
            for date in a:
                counter['solve_gain']+=b[date]['won'] and not a[date]['won']
                counter['solve_loss']+=a[date]['won'] and not b[date]['won']
                p,q=purple_status(a[date]),purple_status(b[date])
                counter['purple_gain']+=p=='missed' and q!='missed'
                counter['purple_loss']+=q=='missed' and p!='missed'
        paired[f'{right} versus {left}']=dict(counter)
    trials=[]; trial_curves=[]; gradient_rows=[]
    for path in sorted((ROOT/'runs').glob('trial-*.json')):
        r=json.loads(path.read_text())
        h=r.get('training',{}).get('history',r.get('history'))
        trials.append(dict(label=r['label'],status=r['status'],params=r['params'],
            updates=h[-1]['update'],best_hit1=max(x['val_s4_hit1'] for x in h),
            best_update=r.get('training',{}).get('best_update',
                max(h,key=lambda x:(x['val_s4_hit1'],x['val_s4_mrr']))['update']),
            final_loss=h[-1]['train_loss'],diagnostics=r.get('diagnostics')))
        for row in h:
            trial_curves.append(dict(trial=r['label'],update=row['update'],
                hit1=row['val_s4_hit1'],loss=row['train_loss']))
        for record in r.get('training',{}).get('gradient_diagnostics',[]):
            if record['update'] not in (1,10,500,h[-1]['update']):
                continue
            for module in ('proj','transform_projection','transform_encoder','pair_encoder'):
                gradient_rows.append(dict(trial=r['label'],update=record['update'],
                    module=module,**record['parameters'][module]))
    summary=dict(manifest=manifest,best=best,aggregate=aggregates,paired=paired,runs=rows,
        trials=trials,gradient_samples=gradient_rows)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    data=alt.Data(values=curves)
    hit=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('hit1:Q',axis=alt.Axis(format='.0%')),
        color='variant:N',tooltip=['variant:N','update:Q','hit1:Q']).properties(width=320,height=220).facet(column='seed:O')
    loss=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('loss:Q',scale=alt.Scale(type='log')),
        color='variant:N').properties(width=320,height=220).facet(column='seed:O')
    alt.vconcat(hit,loss).save(ROOT/'learning-curves.html')
    diag=alt.Data(values=diag_curves)
    diag_loss=alt.Chart(diag).mark_line().encode(x='update:Q',y=alt.Y('loss:Q',scale=alt.Scale(type='log')),
        color='variant:N',strokeDash='split:N',tooltip=['variant:N','split:N','update:Q','loss:Q']).properties(width=320,height=220).facet(column='seed:O')
    diag_hit=alt.Chart(diag).mark_line().encode(x='update:Q',y='hit1:Q',color='variant:N',
        strokeDash='split:N').properties(width=320,height=220).facet(column='seed:O')
    alt.vconcat(diag_loss,diag_hit).save(ROOT/'diagnostics.html')
    trial_data=alt.Data(values=trial_curves)
    trial_hit=alt.Chart(trial_data).mark_line().encode(x='update:Q',y='hit1:Q',
        color='trial:N',tooltip=['trial:N','update:Q','hit1:Q']).properties(width=220,height=150).facet(facet='trial:N',columns=4)
    trial_loss=alt.Chart(trial_data).mark_line().encode(x='update:Q',
        y=alt.Y('loss:Q',scale=alt.Scale(type='log')),color='trial:N').properties(width=220,height=150).facet(facet='trial:N',columns=4)
    alt.vconcat(trial_hit,trial_loss).save(ROOT/'trial-curves.html')
    lines=['# Learned-transformation edge hyperparameter tuning','',
        'Validation-only tuning of the full-transformation, text-node GINE. '
        'Architecture and probability-mass objective are fixed. Frozen Qwen vectors '
        'remain frozen; all active projection and GINE parameters train. '
        'No notebook or test-set changes.','',
        '## Setup','',
        '12 Optuna trials, seed 0. Six full startup trials include the inherited '
        'settings, lower LRs (1e-4 and 3e-5), and regularized variants. Search: '
        'LR 1e-5–5e-4 (log), weight decay 1e-5–1e-2 (log), GINE dropout '
        '0 / .05 / .1 / .2. Hidden width 128, two GINE layers, readout 128, '
        'transformation width 64 and 16 learned edge channels remain fixed.','',
        'After trial 7, explicitly queued dropout .05 and .2 at LR 3e-4 / WD .001 '
        'to cover the proposed range instead of relying on the adaptive sampler '
        'to explore unseen dropout values. These remain within the twelve trials; '
        'the decision and exact parameters are in dropout-coverage-queue.json.','',
        'Each completed run gets 30,000 updates, batch 16, balanced 10,000 updates '
        'at each state size, validation every 500. No scheduler or early stopping. '
        'Median pruning starts after six full trials, at >=20,000 updates with '
        'at least four comparisons. Selection uses full-board Hit@1, checkpoints '
        'use MRR to break ties. Rerun selected parameters with seeds 10/11/12; '
        'also repeat original-only masking at the same parameters. The latter '
        'is not independently tuned. Existing controls have the same budget.','',
        'Diagnostics: eval-mode loss, Hit@1 and MRR on all validation states and '
        'a fixed 96-state training sample per size every 2,000 updates plus first/last. '
        'Training/validation differences are sampled estimates. Gradient probes '
        'record new encoders and base modules at updates 1, 10 and every 500. '
        'The additional GINE edge columns start at zero, so zero encoder gradient '
        'on the very first step is expected. Probes are tested to preserve updates.','',
        'Trials use one tuning seed; three-seed reruns check sensitivity to '
        'initialization, not generalization to new data. The 121 validation '
        'puzzles have been reused extensively. Harness: beam 20, full-board '
        'scores, four mistakes, shuffle seed zero.','',
        '## Selected settings','',f"Trial {best['trial']}, tuning-seed Hit@1 {best['value']:.2%}.",'',
        '```json',json.dumps(best['params'],indent=2),'```','',
        '## Trial results','', '| Trial | Status | LR | WD | Dropout | Updates | Best Hit@1 | Best update |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for t in trials:
        p=t['params']
        lines.append(f"| {t['label']} | {t['status']} | {p['lr']:.3g} | {p['weight_decay']:.3g} | {p['dropout']} | {t['updates']} | {t['best_hit1']:.2%} | {t['best_update']} |")
    lines+=['','## Optimization diagnostics (tuning seed)','',
        '| Trial | Update | Train s4 loss | Val s4 loss | Train s4 Hit@1 | Val s4 Hit@1 |',
        '|---|---:|---:|---:|---:|---:|']
    for t in trials:
        for d in (t['diagnostics'][0],t['diagnostics'][-1]):
            a,b=d['train']['s4'],d['validation']['s4']
            lines.append(f"| {t['label']} | {d['update']} | {a['loss']:.4f} | {b['loss']:.4f} | {a['hit1']:.2%} | {b['hit1']:.2%} |")
    lines+=['','Gradient/update measurements are available in summary.json and per-run '
        'training.gradient_diagnostics. Zero gradient into the extra branch at step 1 '
        'is expected from zero-initialized downstream columns; subsequent nonzero '
        'measurements distinguish that from a permanently blocked branch.','']
    lines+=['','## Matched reruns','',
        '| Metric | Ordinary GINE | Old full | Old original-only | Tuned full | Tuned original-only |',
        '|---|---:|---:|---:|---:|---:|']
    for s in ('s4','s3','s2'):
        for k in ('hit1','mrr','hit4'):
            lines.append(f'| {s} {k} | '+' | '.join(f"{aggregates[v]['ranking'][s][k]:.4f}" for v in VARIANTS)+' |')
    for key in ('solves','first_correct'):
        lines.append(f'| Harness {key} | '+' | '.join(f"{aggregates[v][key]}/363 ({aggregates[v][key]/363:.2%})" for v in VARIANTS)+' |')
    for c in ('yellow','green','blue','purple'):
        lines.append(f'| {c} recovered | '+' | '.join(str(aggregates[v]['colors'][c]) for v in VARIANTS)+' |')
    for k in ('selected','forced'):
        lines.append(f'| Purple {k} | '+' | '.join(str(aggregates[v]['purple'][k]) for v in VARIANTS)+' |')
        lines.append(f'| Targeted Purple {k} | '+' | '.join(str(aggregates[v]['targeted'][k])+'/12' for v in VARIANTS)+' |')
    lines+=['','## Per seed','', '| Arm | Seed | Hit@1 | Last Hit@1 | Best update | Solves | Purple direct / forced |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['variant']} | {r['seed']} | {r['ranking']['s4']['hit1']:.2%} | {r['last_hit1']:.2%} | {r['best_update']} | {r['solves']} | {r['purple'].get('selected',0)} / {r['purple'].get('forced',0)} |")
    lines+=['','## Paired harness changes','']
    for label,p in paired.items():
        lines.append(f"- {label}: {p['solve_gain']} gains / {p['solve_loss']} losses; Purple {p['purple_gain']} gains / {p['purple_loss']} losses.")
    lines+=['','## Concrete target groups','', '| Category | Ordinary GINE | Old full | Old original-only | Tuned full | Tuned original-only |',
        '|---|---:|---:|---:|---:|---:|']
    for t in targets:
        cells=[]
        for v in VARIANTS:
            c=Counter(purple_status(games_by_key[v,s][t['date']]) for s in (10,11,12))
            cells.append(f"{c['selected']} direct / {c['forced']} forced")
        lines.append('| '+t['category']+' | '+' | '.join(cells)+' |')
    lines+=['','## Artifacts','',
        '[Learning curves](gine-transformation-tuning/learning-curves.html), '
        '[train/validation diagnostics](gine-transformation-tuning/diagnostics.html), '
        '[trial curves](gine-transformation-tuning/trial-curves.html), '
        '[raw summary](gine-transformation-tuning/summary.json).','']
    long_path=ROOT/'long-check.json'
    if long_path.exists():
        spec=json.loads(long_path.read_text())
        previous=json.loads(Path(spec['matched_30k']['result']).read_text())
        short=json.loads(Path(spec['within_run_30k']['result']).read_text())
        short_game=json.loads(Path(spec['within_run_30k']['harness']).read_text())
        extended=json.loads(Path(spec['result']).read_text())
        game=json.loads(Path(spec['harness']).read_text())
        assert game['split']=='val' and extended['training']['updates_ran']==60000
        short_hist=short['training']['history']; long_hist=extended['training']['history']
        prefix=max(abs(a['val_s4_hit1']-b['val_s4_hit1']) for a,b in zip(short_hist,long_hist))
        assert prefix==0
        prior_difference=max(abs(a['val_s4_hit1']-b['val_s4_hit1'])
            for a,b in zip(previous['training']['history'],long_hist))
        assert short['training']['validation']['s4']['hit1']==max(h['val_s4_hit1'] for h in short_hist)
        long_summary=dict(**spec,ranking=extended['training']['validation'],
            best_update=extended['training']['best_update'],
            solves=sum(g['won'] for g in game['games']),
            purple=dict(Counter(purple_status(g) for g in game['games'])),
            within_run_30k_solves=sum(g['won'] for g in short_game['games']),
            within_run_30k_ranking=short['training']['validation'],
            earlier_30k_repeat_max_hit1_difference=prior_difference,
            best_first30k=max(h['val_s4_hit1'] for h in long_hist if h['update']<=30000),
            best_after30k=max(h['val_s4_hit1'] for h in long_hist if h['update']>30000))
        long_audit=json.loads((ROOT/'long-audit'/'probability-audit.json').read_text())
        assert len(long_audit['records'])==242
        for variant,model_result in [('long-60k',extended),('long-prefix30k',short)]:
            records=[r for r in long_audit['records'] if r['variant']==variant]
            assert len(records)==121
            assert abs(mean(1 in r['gold_ranks'] for r in records)-model_result['training']['validation']['s4']['hit1'])<1e-9
        summary['long_check']=long_summary
        (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        long_curves=[]
        for label,r in [('earlier 30k run',previous),('60k run',extended)]:
            for h in r['training']['history']:
                long_curves.append(dict(budget=label,update=h['update'],hit1=h['val_s4_hit1'],loss=h['train_loss']))
        data=alt.Data(values=long_curves)
        chart=alt.Chart(data).mark_line().encode(x='update:Q',y='hit1:Q',color='budget:N',
            tooltip=['budget:N','update:Q','hit1:Q']).properties(width=700,height=240)
        loss_data=[]
        for d in extended['diagnostics']:
            for split in ('train','validation'):
                loss_data.append(dict(update=d['update'],split=split,loss=d[split]['s4']['loss']))
        loss_chart=alt.Chart(alt.Data(values=loss_data)).mark_line().encode(x='update:Q',
            y=alt.Y('loss:Q',scale=alt.Scale(type='log')),color='split:N',
            tooltip=['split:N','update:Q','loss:Q']).properties(width=700,height=240)
        alt.vconcat(chart,loss_chart).save(ROOT/'long-check-curves.html')
        lines+=['','## Extended budget check','',
            f"Selected seed {spec['seed']} because its best checkpoint was latest among the three full-model reruns. "
            'Run 60k updates with unchanged optimizer settings, batch size and balanced state schedule; '
            'save the best checkpoint from the first 30k updates of that SAME uninterrupted run. '
            'No freezing, schedule, optimizer restart or auxiliary objective. '
            'This is a separate diagnostic, not part of the matched 30k aggregates.','',
            'The first attempted extension was interrupted and preserved after its early same-seed '
            'trajectory differed from the earlier 30k run. The cause of that cross-process stochastic '
            'difference was not established. Compare the within-run checkpoints below to isolate budget; '
            'the earlier 30k run is shown separately in the plot. '
            f'Maximum first-30k Hit@1 difference from that earlier run: {prior_difference:.4f}.','',
            f"30k selected Hit@1 {short['training']['validation']['s4']['hit1']:.2%}; "
            f"60k selected Hit@1 {long_summary['ranking']['s4']['hit1']:.2%} "
            f"at update {long_summary['best_update']}. "
            f"Best after 30k: {long_summary['best_after30k']:.2%}. "
            f"Within-run 30k / 60k harness solves: {long_summary['within_run_30k_solves']} / {long_summary['solves']} out of 121. "
            f"Purple direct/forced: {long_summary['purple'].get('selected',0)}/{long_summary['purple'].get('forced',0)}.",'',
            '[Extended learning curve](gine-transformation-tuning/long-check-curves.html).','']
    tuned=aggregates['tuned'];old=aggregates['edges-text'];control=aggregates['control'];original=aggregates['tuned-original-only']
    lines+=['','## Findings','',
        f"Tuning improves full-transformation s4 Hit@1 from {old['ranking']['s4']['hit1']:.2%} "
        f"to {tuned['ranking']['s4']['hit1']:.2%}, solves from {old['solves']} to {tuned['solves']}/363, "
        f"and Purple recovery from {old['colors']['purple']} to {tuned['colors']['purple']}/363. "
        'All three full-model seeds improve selected Hit@1 and solve count over their old settings. '
        'The original negative comparison therefore overstated the regression caused by the architecture alone: '
        'inherited optimization/regularization settings materially contributed.','',
        'The winner keeps LR essentially unchanged (3e-4) and increases dropout to .2 and WD to .001. '
        'Lower LR alone was less useful. The .05 versus .2 comparison at identical LR/WD also favors .2 '
        'on the tuning seed. This supports stronger regularization rather than a blanket conclusion '
        'that the LR was too high. Dropout .2 is the largest tested, so this study does not establish '
        'an optimum beyond the searched range. Width, depth, batch size and schedules were not swept.','',
        f"Ordinary GINE remains stronger: {control['ranking']['s4']['hit1']:.2%} Hit@1, "
        f"{control['solves']}/363 solves, {control['colors']['purple']}/363 Purple recoveries. "
        f"Tuned original-only is close to tuned full: {original['ranking']['s4']['hit1']:.2%} Hit@1 "
        f"and {original['solves']}/363 solves. Full readings gain 31 paired solves and lose 28; "
        'their incremental benefit is small and mixed. Original-only is evaluated at the full model\'s '
        'selected hyperparameters rather than independently tuned.','',
        'None of the four concrete covered substring categories is solved directly by either tuned arm '
        '(0/12 group-seed opportunities each). Both recover the zero-prefix group twice as a forced last group. '
        'Broad Purple gains therefore do not establish successful use of the intended transformations.','',
        'Learning curves still show overfitting: eval-mode training-sample Hit@1 reaches roughly 100%, '
        'while validation loss increases and late validation ranking fluctuates below selected peaks. '
        'On the tuned full reruns, best checkpoints occur at 26.5k / 22.5k / 2k updates, demonstrating '
        'why short universal budgets would be misleading. The added branch receives nonzero gradients '
        'and parameter updates after its deliberately blocked first step. That rules out a permanently '
        'disconnected branch, not every possible optimization difficulty.','',
        'Keep ordinary GINE as the baseline. Retain this experiment as evidence that each new architecture '
        'needs its own tuning and measured curves. These repeatedly reused validation puzzles cannot '
        'establish gains on fresh data. No default model or notebook was changed.','']
    if 'long_check' in summary:
        l=summary['long_check']
        lines+=['The controlled extended-budget check selects '
            f"{l['within_run_30k_ranking']['s4']['hit1']:.2%} Hit@1 after the first30k "
            f"and {l['ranking']['s4']['hit1']:.2%} after60k, with "
            f"{l['within_run_30k_solves']} versus {l['solves']} solves. "
            f"The best score observed strictly after30k is {l['best_after30k']:.2%}. "
            'For this checked trajectory, another 30k updates does not fix the regression. '
            'This does not imply every future architecture or optimizer should use the same budget.','']
    ROOT.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__': main()
