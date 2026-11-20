"""Summarize matched width/depth trials, including mechanism-specific Purple recovery."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import altair as alt
from summarize_gine_group_contrastive import purple_status

ROOT=Path('experiments/gine-transformation-capacity')
VARIANTS=('ordinary','full','edge-small','all-small','one-gine','linear-pair')


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entries=json.loads((ROOT/'runs.json').read_text())
    audit=json.loads((ROOT/'probability-audit.json').read_text())
    assert len(entries)==18 and len(audit['records'])==2178
    targets=json.loads(Path('experiments/gine-transformations/target-coverage.json').read_text())['records']
    target_dates={t['date'] for t in targets}
    runs=[];curves=[];diagnostics=[];games_by_key={}
    for e in entries:
        r=json.loads(Path(e['result']).read_text());h=json.loads(Path(e['harness']).read_text())
        t=r['training'];g=h['games']
        assert r['status']=='complete' and h['split']=='val' and len(g)==121
        assert t['updates_ran']==30000 and len(t['history'])==60
        games_by_key[e['variant'],e['seed']]={x['date']:x for x in g}
        runs.append(dict(**e,checkpoint=r['checkpoint'],ranking=t['validation'],
            best_update=t['best_update'],parameters=r['parameters'],
            solves=sum(x['won'] for x in g),first_correct=sum(x['first_guess_correct'] for x in g),
            colors=dict(Counter(c for x in g for c in x['solved_colors'])),
            purple=dict(Counter(purple_status(x) for x in g)),
            targeted=dict(Counter(purple_status(x) for x in g if x['date'] in target_dates))))
        for row in t['history']:
            curves.append(dict(variant=e['variant'],seed=e['seed'],update=row['update'],
                loss=row['train_loss'],hit1=row['val_s4_hit1']))
        for row in r.get('diagnostics',[]):
            for split in ('train','validation'):
                d=row[split]['s4']
                diagnostics.append(dict(variant=e['variant'],seed=e['seed'],update=row['update'],
                    split=split,loss=d['loss'],hit1=d['hit1']))
    aggregate={}
    for v in VARIANTS:
        rows=[r for r in runs if r['variant']==v]
        records=[r for r in audit['records'] if r['variant']==v]
        assert len(rows)==3 and len(records)==363
        aggregate[v]=dict(ranking={s:{k:mean(r['ranking'][s][k] for r in rows)
            for k in ('hit1','mrr','hit4')} for s in ('s4','s3','s2')},
            solves=sum(r['solves'] for r in rows),first_correct=sum(r['first_correct'] for r in rows),
            colors={c:sum(r['colors'].get(c,0) for r in rows) for c in ('yellow','green','blue','purple')},
            purple={k:sum(r['purple'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            targeted={k:sum(r['targeted'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            parameters=rows[0]['parameters'])
        assert abs(mean(1 in r['gold_ranks'] for r in records)-aggregate[v]['ranking']['s4']['hit1'])<1e-9
    paired={}
    for left,right in [('full',v) for v in VARIANTS[2:]]+list(zip(VARIANTS[2:-1],VARIANTS[3:])):
        counts=Counter()
        for seed in (10,11,12):
            a,b=games_by_key[left,seed],games_by_key[right,seed]
            assert a.keys()==b.keys()
            for date in a:
                counts['solve_gain']+=b[date]['won'] and not a[date]['won']
                counts['solve_loss']+=a[date]['won'] and not b[date]['won']
                p,q=purple_status(a[date]),purple_status(b[date])
                counts['purple_gain']+=p=='missed' and q!='missed'
                counts['purple_loss']+=q=='missed' and p!='missed'
        paired[f'{right} versus {left}']=dict(counts)
    old=json.loads(Path('experiments/gine-transformation-tuning/summary.json').read_text())['aggregate']['tuned']
    summary=dict(manifest=manifest,aggregate=aggregate,runs=runs,paired=paired,previous_tuned_full=old)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    data=alt.Data(values=curves)
    hit=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('hit1:Q',axis=alt.Axis(format='.0%')),
        color='variant:N',tooltip=['variant:N','update:Q','hit1:Q']).properties(width=330,height=220).facet(column='seed:O')
    loss=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('loss:Q',scale=alt.Scale(type='log')),
        color='variant:N').properties(width=330,height=220).facet(column='seed:O')
    alt.vconcat(hit,loss).save(ROOT/'learning-curves.html')
    data=alt.Data(values=diagnostics)
    loss=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('loss:Q',scale=alt.Scale(type='log')),
        color='variant:N',strokeDash='split:N',tooltip=['variant:N','split:N','update:Q','loss:Q']).properties(width=330,height=220).facet(column='seed:O')
    hit=alt.Chart(data).mark_line().encode(x='update:Q',y='hit1:Q',color='variant:N',strokeDash='split:N').properties(width=330,height=220).facet(column='seed:O')
    alt.vconcat(loss,hit).save(ROOT/'diagnostics.html')
    lines=['# Smaller learned-transformation GINE models','',
        'Validation-only capacity ladder: five freshly trained model variants, seeds 10/11/12. '
        'Keep ordinary GINE from the earlier study as a reference. No notebook or test-set evaluation.','',
        '## Existing MLPs and projections','',
        '- Text input: LayerNorm(1024), Linear 1024→256, ReLU. This pretrained initialization remains trainable and unchanged.',
        '- Transformation input: Linear 1024→64; concatenate an 8-dimensional type embedding and five position values; Linear 77→64, ReLU.',
        '- Pair matcher: symmetric difference/product plus compatibility (136 values) →64→16, ReLU after each linear; max across valid reading pairs.',
        '- Candidate GINE input: Linear 256→128; two message-passing blocks, each with its own edge projection and 128→128→128 MLP.',
        '- Score: mean/max pooling produces 256 values; readout 256→128→1.',
        '', 'The full model has 14 trainable linear layers across these modules. Dropout is applied in the GINE update MLPs and readout. '
        'The transformation and pair branches do not receive additional dropout in this comparison.','',
        '## Controlled changes','',
        'All arms keep the same eight cached alternatives, original 15 supplied edges, learned edge dimension 16, '
        'text nodes, pairwise max pooling, source split and probability-mass loss. Qwen vectors stay frozen; '
        'all active projection, LayerNorm, GINE and learned-edge parameters train. '
        'This preserves the text projection initialization; changing its width would introduce another factor.','',
        'Each run gets 30,000 updates (balanced 10,000 per state size), batch 16, evaluation every 500, '
        'AdamW LR 3e-4, WD .001, GINE/readout dropout .2. No scheduler, stopping or pruning. '
        'Restore best s4 Hit@1 with MRR tie-break. Use identical seeded batch permutations and the previous fixed '
        '96-state training diagnostic sample per size; full validation loss/ranking every 2k plus first/last. '
        'These optimizer settings were selected for the full model; smaller models are not independently retuned.','',
        'The budget corresponds to about 165 full-board training passes, 41 passes over the 12-word '
        'states and 28 over the 8-word states. The training-loss plot averages stochastic training '
        'batches across state sizes; the diagnostic plot compares s4 training/validation with dropout off.','',
        'The fresh full-size control runs in the same process as the smaller arms. Same seeds do not guarantee '
        'bitwise GPU training replay; prior results are shown separately rather than silently substituted. '
        'The 121 validation puzzles are reused across seeds and previous experiments, so this is not fresh-data confirmation.','',
        'Harness: beam 20, full-board scores, four mistakes, shuffle seed zero. Purple direct and forced-last '
        'recoveries are separate; four previously selected concrete substring categories are inspected separately.','',
        '| Arm | Transformation width | Pair hidden / linear depth | GINE width / blocks | Readout hidden | Trainable parameters | Trainable linear layers |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v,c in manifest['variants'].items():
        p=aggregate[v]['parameters']
        hidden=str(c['pair_hidden']) if c['pair_layers']==2 else 'none'
        count=sum(len(shapes) for shapes in p['active_linear_layers'].values())
        lines.append(f"| {v} | {c['transformation_width']} | {hidden} / {c['pair_layers']} | {c['hidden_width']} / {c['gine_layers']} | {c['readout_width']} | {p['trainable']:,} | {count} |")
    text_count=sum(aggregate['full']['parameters']['by_module'][k] for k in ('proj','input_norm'))
    lines+=['',f'The unchanged text projection and LayerNorm account for {text_count:,} trainable parameters. '
        'This study reduces the relation/GINE/readout portions, not every source of capacity. '
        'Width changes can also discard useful information, and GINE depth changes the message computation; '
        'the intervention is not a pure scalar parameter-count adjustment.','',
        '## Aggregate validation results','',
        '| Metric | Ordinary reference | Fresh full | Small edges | Small widths | One GINE | Linear pair |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in ('s4','s3','s2'):
        for k in ('hit1','mrr','hit4'):
            lines.append(f'| {s} {k} | '+' | '.join(f"{aggregate[v]['ranking'][s][k]:.4f}" for v in VARIANTS)+' |')
    for k in ('solves','first_correct'):
        lines.append(f'| Harness {k} | '+' | '.join(f"{aggregate[v][k]}/363 ({aggregate[v][k]/363:.2%})" for v in VARIANTS)+' |')
    for c in ('yellow','green','blue','purple'):
        lines.append(f'| {c} recovered | '+' | '.join(str(aggregate[v]['colors'][c]) for v in VARIANTS)+' |')
    for k in ('selected','forced'):
        lines.append(f'| Purple {k} | '+' | '.join(str(aggregate[v]['purple'][k]) for v in VARIANTS)+' |')
        lines.append(f'| Target Purple {k} | '+' | '.join(str(aggregate[v]['targeted'][k])+'/12' for v in VARIANTS)+' |')
    lines+=['','## Per seed','', '| Arm | Seed | Hit@1 | Solves | Purple direct / forced | Best update |',
        '|---|---:|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['variant']} | {r['seed']} | {r['ranking']['s4']['hit1']:.2%} | {r['solves']} | {r['purple'].get('selected',0)} / {r['purple'].get('forced',0)} | {r['best_update']} |")
    lines+=['','## Paired differences','']
    for label,p in paired.items():
        lines.append(f"- {label}: solves {p['solve_gain']} gained / {p['solve_loss']} lost; Purple {p['purple_gain']} gained / {p['purple_loss']} lost.")
    lines+=['','## End-of-run diagnostics','',
        '| Arm | Train sample Hit@1 | Validation Hit@1 | Train sample loss | Validation loss |',
        '|---|---:|---:|---:|---:|']
    for v in VARIANTS[1:]:
        ds=[d for d in diagnostics if d['variant']==v and d['update']==30000]
        a=[d for d in ds if d['split']=='train'];b=[d for d in ds if d['split']=='validation']
        lines.append(f"| {v} | {mean(d['hit1'] for d in a):.2%} | {mean(d['hit1'] for d in b):.2%} | {mean(d['loss'] for d in a):.4f} | {mean(d['loss'] for d in b):.4f} |")
    lines+=['','## Concrete targeted Purple groups','', '| Category | Ordinary | Full | Small edges | Small widths | One GINE | Linear pair |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for target in targets:
        cells=[]
        for v in VARIANTS:
            c=Counter(purple_status(games_by_key[v,seed][target['date']]) for seed in (10,11,12))
            cells.append(f"{c['selected']} direct / {c['forced']} forced")
        lines.append('| '+target['category']+' | '+' | '.join(cells)+' |')
    lines+=['','## Previous full-model comparison','',
        f"Previous tuned full: {old['ranking']['s4']['hit1']:.2%} Hit@1, {old['solves']}/363 solves. "
        f"Fresh full: {aggregate['full']['ranking']['s4']['hit1']:.2%} Hit@1, {aggregate['full']['solves']}/363 solves. "
        'Use the fresh full control for the capacity comparison.','',
        '## Artifacts','', '[Learning curves](gine-transformation-capacity/learning-curves.html), '
        '[train/validation diagnostics](gine-transformation-capacity/diagnostics.html), '
        '[raw summary](gine-transformation-capacity/summary.json).','']
    ROOT.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps({v:dict(hit1=a['ranking']['s4']['hit1'],solves=a['solves'],purple=a['purple'])
        for v,a in aggregate.items()},indent=2))


if __name__=='__main__':main()
