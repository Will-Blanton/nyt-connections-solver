"""Isolate non-original alternatives in the learned-edge branch."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import altair as alt
from summarize_gine_group_contrastive import purple_status

ROOT=Path('experiments/gine-original-only')
VARIANTS=('control','original-only','edges-text')


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entries=json.loads((ROOT/'runs.json').read_text())
    assert len(entries)==9
    audit=json.loads((ROOT/'probability-audit.json').read_text())
    coverage=json.loads(Path('experiments/gine-transformations/target-coverage.json').read_text())
    targets={r['date'] for r in coverage['records'] if r['all_targets_present']}
    runs=[]; games_by_key={};curves=[]
    for entry in entries:
        result=json.loads(Path(entry['result']).read_text())
        harness=json.loads(Path(entry['harness']).read_text())
        t=result['training'];games=harness['games']
        assert result['status']=='complete' and t['updates_ran']==30000 and len(t['history'])==60
        assert harness['split']=='val' and len(games)==121
        games_by_key[entry['variant'],entry['seed']]={g['date']:g for g in games}
        curves.extend(dict(variant=entry['variant'],seed=entry['seed'],update=h['update'],
            hit1=h['val_s4_hit1'],loss=h['train_loss']) for h in t['history'])
        runs.append(dict(**entry,ranking=t['validation'],parameters=result['parameters'],
            checkpoint=result['checkpoint'],best_update=t['best_update'],
            best_first15k=max(h['val_s4_hit1'] for h in t['history'] if h['update']<=15000),
            solves=sum(g['won'] for g in games),first_correct=sum(g['first_guess_correct'] for g in games),
            colors=dict(Counter(c for g in games for c in g['solved_colors'])),
            purple=dict(Counter(purple_status(g) for g in games)),
            targeted=dict(Counter(purple_status(g) for g in games if g['date'] in targets))))
    aggregate={}; paired={}
    for v in VARIANTS:
        rows=[r for r in runs if r['variant']==v];assert len(rows)==3
        boards=[b for b in audit['records'] if b['variant']==v];assert len(boards)==363
        aggregate[v]=dict(
            ranking={s:{k:mean(r['ranking'][s][k] for r in rows) for k in ('hit1','mrr','hit4')} for s in ('s4','s3','s2')},
            solves=sum(r['solves'] for r in rows),first_correct=sum(r['first_correct'] for r in rows),
            colors={c:sum(r['colors'].get(c,0) for r in rows) for c in ('yellow','green','blue','purple')},
            purple={k:sum(r['purple'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            targeted={k:sum(r['targeted'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            trainable_parameters=rows[0]['parameters']['trainable'],
            color_top4={c:mean(b['gold_ranks'][i]<=4 for b in boards) for i,c in enumerate(audit['colors'])})
        assert abs(mean(1 in b['gold_ranks'] for b in boards)-aggregate[v]['ranking']['s4']['hit1'])<1e-9
    for a,b in [('original-only','edges-text'),('control','original-only')]:
        counts=Counter()
        for seed in manifest['seeds']:
            left,right=games_by_key[a,seed],games_by_key[b,seed]
            assert left.keys()==right.keys()
            for date in left:
                counts['solve_gain']+=right[date]['won'] and not left[date]['won']
                counts['solve_loss']+=left[date]['won'] and not right[date]['won']
                p,q=purple_status(left[date]),purple_status(right[date])
                counts['purple_gain']+=p=='missed' and q!='missed'
                counts['purple_loss']+=q=='missed' and p!='missed'
        paired[f'{b}_versus_{a}']=dict(counts)
    summary=dict(manifest=manifest,aggregate=aggregate,runs=runs,paired=paired,
        targeted=[dict(date=date,variant=v,seed=seed,status=purple_status(games_by_key[v,seed][date]))
                  for date in sorted(targets) for v in VARIANTS for seed in manifest['seeds']])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    data=alt.Data(values=curves)
    hit=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('hit1:Q',axis=alt.Axis(format='.0%')),color='variant:N',tooltip=['variant:N','update:Q','hit1:Q']).properties(width=330,height=220).facet(column='seed:O')
    loss=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('loss:Q',scale=alt.Scale(type='log')),color='variant:N').properties(width=330,height=220).facet(column='seed:O')
    alt.vconcat(hit,loss).save(ROOT/'learning-curves.html')
    lines=['# Original-only learned-edge isolation','',
        'The large regression remains when the learned matcher receives only '
        'original words. Original-only and full transformations are close in '
        'aggregate (59 versus 57 solves), while both trail the existing GINE '
        '(96). Extra alternatives are not established as the main cause of '
        'the earlier performance drop.','',
        'Validation only: three new matched seeds (10/11/12), 30,000 updates each; '
        'reuse the completed full-transformation/text-node and original GINE controls. '
        '121 puzzles per seed. Repeated appearances are correlated, and these '
        'puzzles have been used in previous experiments. No test-set evaluation '
        'or notebook changes.','', '## Controlled intervention','',
        'Keep text nodes, the 64-dimensional transformation encoder, 16-dimensional '
        'learned edge vector, pair MLP and max pooling unchanged. Both learned-edge '
        'arms have exactly 484,787 trainable parameters, matched initialization, '
        'batch permutations and balanced schedules. In the new arm, valid '
        'alternatives are masked to type=original only before pair pooling. '
        'The payload, padding, parameter layout and number of computed slots remain '
        'unchanged. Original alternatives are lowercased frozen Qwen vectors, '
        'identical to slot zero of the prior full-transformation arm.','',
        'This masks only the learned-edge branch. The original 15 hand-built '
        'edge channels, including existing substring similarity metrics, remain '
        'present in every arm. This is not a comparison against a system with '
        'all transformation information removed. The control column is the '
        'ordinary GINE without any added learned edges.','',
        'Full budget: 30,000 updates, batch 16, 10,000 updates per 8/12/16-word '
        'state size, evaluation every 500 updates, four CPU threads. Ordinary '
        'probability-mass loss, LR 0.0003016685, weight decay 0.0000950089, '
        'dropout zero, no scheduling, auxiliary/floor loss, pruning or early '
        'stopping. Restore best s4 Hit@1 with MRR tie-break. Harness: beam 20, '
        'full-board score sums, four mistakes and shuffle seed zero.','',
        'A unit test verifies identical initialization/counts, equivalence to a '
        'single original-pair calculation, no output/gradient dependence on '
        'excluded readings, and checkpoint reconstruction. The new flag defaults '
        'to false, preserving existing model behavior.','',
        '## Aggregate','', '| Metric | Existing GINE | Original-only learned edge | Full-transformation learned edge |','|---|---:|---:|---:|']
    for s in ('s4','s3','s2'):
        for k in ('hit1','mrr','hit4'):
            lines.append(f'| {s} {k} | '+' | '.join(f"{aggregate[v]['ranking'][s][k]:.4f}" if k=='mrr' else f"{aggregate[v]['ranking'][s][k]:.2%}" for v in VARIANTS)+' |')
    for k in ('solves','first_correct'):
        lines.append(f'| Harness {k} | '+' | '.join(f"{aggregate[v][k]}/363 ({aggregate[v][k]/363:.2%})" for v in VARIANTS)+' |')
    for c in ('yellow','green','blue','purple'):
        lines.append(f'| {c} recovered | '+' | '.join(str(aggregate[v]['colors'][c])+'/363' for v in VARIANTS)+' |')
    for k in ('selected','forced'):
        lines.append(f'| Purple {k} | '+' | '.join(str(aggregate[v]['purple'][k])+'/363' for v in VARIANTS)+' |')
    for k in ('selected','forced'):
        lines.append(f'| Concrete covered Purple {k} | '+' | '.join(str(aggregate[v]['targeted'][k])+'/12' for v in VARIANTS)+' |')
    lines+=['','## Per seed','', '| Seed | Arm | s4 Hit@1 | Solves | Purple direct / forced | Best update |','|---|---|---:|---:|---:|---:|']
    for seed in manifest['seeds']:
        for v in VARIANTS:
            r=next(r for r in runs if r['seed']==seed and r['variant']==v)
            lines.append(f"| {seed} | {v} | {r['ranking']['s4']['hit1']:.2%} | {r['solves']}/121 | {r['purple'].get('selected',0)} / {r['purple'].get('forced',0)} | {r['best_update']} |")
    lines+=['','## Paired changes','']
    for label,p in paired.items():
        lines.append(f"- {label}: {p['solve_gain']} new solves, {p['solve_loss']} lost; {p['purple_gain']} new Purple recoveries, {p['purple_loss']} lost.")
    lines+=['','## Concrete targeted cases','', '| Mechanism | Existing GINE | Original-only | Full transformations |','|---|---:|---:|---:|']
    for row in coverage['records']:
        cells=[]
        for v in VARIANTS:
            counts=Counter(purple_status(games_by_key[v,seed][row['date']]) for seed in manifest['seeds'])
            cells.append(f"{counts['selected']} direct / {counts['forced']} forced")
        lines.append('| '+row['category']+' | '+' | '.join(cells)+' |')
    lines+=['','These four intended-mechanism examples were chosen in the previous '
        'coverage audit; all sixteen readings are available in the full payload. '
        'They are analysis-only labels.','', '## Conclusions','',
        'Adding full transformations changes 16-word Hit@1 from 28.65% to '
        '29.20% (+0.55 percentage points), while solves change from 59 to 57 '
        '(-0.55 points). Per-seed solves are 20/18/21 original-only versus '
        '17/20/20 full. Full transformations gain 25 paired solves and lose '
        '27: the effect is not uniformly negative or positive. These small '
        'mixed aggregate differences do not establish equivalence, but do '
        'not show a useful consistent improvement either.','',
        'The dominant loss is already present in the original-only learned '
        'branch: 96 to 59 solves versus ordinary GINE, with losses in every '
        'seed. Thus the earlier full-transformation result should not be '
        'attributed solely to substring alternatives or their extraction. '
        'The learned matcher, added capacity and optimization interactions '
        'are stronger candidates for investigation. This experiment does '
        'not separately identify which one is responsible.','',
        'Full transformations do not show targeted Purple improvements here: '
        'overall Purple recovery changes 75 to 70 and direct selections '
        '52 to 46 versus original-only. Neither learned-edge arm directly '
        'selects any of the four coverage-audited Purple groups. The one '
        'rock-band success in the earlier study belonged to the fused-node '
        'arm, which was deliberately not varied in this isolation.','',
        'All three new runs completed 30,000 updates and sixty validation '
        'observations. Two selected checkpoints occur after 15,000 updates '
        '(22,000 and 24,000), improving on their first-half Hit@1. Final '
        'training losses are approximately 0.05; insufficient ability to '
        'fit the training objective is not the observed problem. Dataset '
        'size remains a plausible generalization limit, not a demonstrated '
        'sole cause.','',
        'Keep the baseline and do not discount the entire transformation '
        'family. The next informative test is a smaller, structured matcher '
        '(shared projection/cosine with explicit operation/position evidence), '
        'rather than widening this pair MLP or merely extending its budget. '
        'No such follow-up was run as part of this requested isolation.','',
        'Seven targeted tests passed. The original-only mask preserves '
        'parameter counts and initialization, matches a single-original-pair '
        'calculation, excludes non-original outputs and gradients, and '
        'survives checkpoint reconstruction. The independent 1,089-board '
        'audit reproduces all nine selected full-board Hit@1 values.','',
        '## Interpretation limits','',
        'This isolates the presence of additional alternatives in this matcher. '
        'It does not distinguish useful transformed content from the effects '
        'of extra comparisons, positional variety, or elementwise-max selection. '
        'Same parameter count does not imply identical effective complexity. '
        'These are matched existing optimizer settings, not tuned upper bounds. '
        'Do not extrapolate a negative result to all transformation representations.','',
        'All nine checkpoints are audited on all 1,820 full-board candidates, '
        'independently reproducing selected Hit@1. Checkpoint paths, per-game '
        'guesses, curves and per-state metrics are retained under '
        '`experiments/gine-original-only/`. Defaults remain unchanged.','',
        'Run: `nix develop -c uv run python scripts/run_gine_original_only.py`. '
        'Audit: `nix develop -c uv run python scripts/audit_gine_probability_floor.py '
        '--results experiments/gine-original-only`. '
        'Summary: `nix develop -c uv run python scripts/summarize_gine_original_only.py`.','']
    Path('experiments/gine-original-only.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__': main()
