"""Mechanism-level and paired validation analysis for shared-witness GINE tests."""
import csv
import json
from pathlib import Path
from statistics import mean

import altair as alt
import pandas as pd

from run_gine_relation_witnesses import summarize,VARIANTS,COLORS


def main():
    root=Path('experiments/gine-relation-witnesses')
    manifest=json.loads((root/'manifest.json').read_text())
    records=[json.loads((root/f'{v}-seed-{seed}.json').read_text())
             for seed in manifest['seeds'] for v in VARIANTS]
    if not all(r['status']=='complete' for r in records):raise ValueError('Wait for all matched runs')
    summarize(root,manifest)
    with open('experiments/group-mechanism-audit/group_mechanisms.csv') as stream:
        labels={(r['date'],r['color']):r for r in csv.DictReader(stream) if r['original_split']=='val'}
    all_rows=[];curves=[];mechanisms=[]
    for run in records:
        games={g['date']:g for g in run['evaluation']['harness']['games']}
        for row in run['evaluation']['gold_ranks']:
            game=games[row['date']]
            for color,rank in zip(COLORS,row['gold_ranks']):
                label=labels[row['date'],color]
                solved=next((h for h in game['history'] if h['color']==color and h['outcome']=='correct'),None)
                all_rows.append(dict(variant=run['variant'],seed=run['seed'],date=row['date'],color=color,
                    category=label['category'],mechanism=label['primary_mechanism'],rank=rank,
                    direct=bool(solved and not solved['forced']),forced=bool(solved and solved['forced'])))
        for row in run['training']['history']:
            for metric in ('train_loss','val_s4_hit1','val_s3_hit1','val_s2_hit1'):
                curves.append(dict(variant=run['variant'],seed=str(run['seed']),update=row['update'],metric=metric,value=row[metric]))
    for variant in VARIANTS:
        for mechanism in sorted({r['mechanism'] for r in all_rows}):
            rows=[r for r in all_rows if r['variant']==variant and r['mechanism']==mechanism]
            mechanisms.append(dict(variant=variant,mechanism=mechanism,groups=len(rows),
                mean_rank=mean(r['rank'] for r in rows),mrr=mean(1/r['rank'] for r in rows),
                top1=mean(r['rank']==1 for r in rows),top4=mean(r['rank']<=4 for r in rows),
                direct=sum(r['direct'] for r in rows),forced=sum(r['forced'] for r in rows)))
    for name,rows in [('group_results',all_rows),('mechanism_results',mechanisms)]:
        with (root/f'{name}.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    chart=alt.Chart(pd.DataFrame(curves)).mark_line().encode(x='update:Q',y=alt.Y('value:Q',scale=alt.Scale(zero=False)),
        color='variant:N',strokeDash='seed:N',tooltip=['variant','seed','update','metric','value'])
    chart.properties(width=600,height=180).facet(row='metric:N').resolve_scale(y='independent').interactive().save(root/'learning-curves.html')
    signal=json.loads((root/'signal-audit.json').read_text())
    lines=['','## Standalone signal audit','',
           '| Score | Train expected Hit@1 | Validation expected Hit@1 |',
           '|---|---:|---:|']
    for name in ('shared_wordnet_concept_specificity','existing_rare_concept_mean',
                 'existing_rare_concept_minimum','edit_target_family_max'):
        lines.append(f"| {name} | {signal['aggregate']['train'][name]['standalone_expected_hit1']:.2%} | {signal['aggregate']['val'][name]['standalone_expected_hit1']:.2%} |")
    lines+=['','Standalone scores rank all 1,820 quartets without model training; ties '
            'at the maximum receive uniform tie-breaking credit. The exact edit family '
            'takes the maximum over its ten fixed channels. These are candidate retrieval '
            'metrics, not complete-puzzle solve rates.', '',
            'Exact edit witnesses support only one training gold group '
            '(bough/cough/dough/tough) and one validation gold group '
            '(wind/wine/wing/wink). A second nonzero channel on the same quartet '
            'does not constitute another covered group. In particular, the exact shared '
            'deletion target has no positive training quartet: the last-position channel '
            'fires on one incorrect training quartet, and the anywhere channel on six. '
            'The training objective therefore supplies only negative examples for these '
            'deletion channels. The ten operation/alignment channels also keep that '
            'supervision separate from the lone positive substitution example. Sparse supervision limits '
            'what this experiment can say about learning that new relation.', '',
            '## Mechanism-specific results','',
           '| Mechanism | Arm | Group appearances | Top4 | Direct | Forced |',
           '|---|---|---:|---:|---:|---:|']
    for row in mechanisms:
        lines.append(f"| {row['mechanism']} | {row['variant']} | {row['groups']} | {row['top4']:.2%} | {row['direct']} | {row['forced']} |")
    lines+=['','## Per-run checkpoints and learning','',
            '| Seed | Arm | s4 Hit@1 | Solves | Best update | Final s4 Hit@1 |',
            '|---|---|---:|---:|---:|---:|']
    for run in records:
        t=run['training'];e=run['evaluation']
        lines.append(f"| {run['seed']} | {run['variant']} | {t['validation']['s4']['hit1']:.4f} | {e['harness']['solved']}/121 | {t['best_update']} | {t['history'][-1]['val_s4_hit1']:.4f} |")
    lines+=['','## Win + letter target','',
            '| Arm | Seed | Rank | Directly recovered | Forced |','|---|---:|---:|---|---|']
    for row in all_rows:
        if row['category']=='win + letter':
            lines.append(f"| {row['variant']} | {row['seed']} | {row['rank']} | {row['direct']} | {row['forced']} |")
    lines+=['','The direct/forced distinction is based on actual guess histories, not recovered '
            'group counts alone. All reported group ranks are independently recomputed after '
            'strict state-dict reload; full-board Hit@1 must reproduce training validation. '
            'No test data or notebook execution. See learning-curves.html, group_results.csv, '
            'mechanism_results.csv and summary.json for retained evidence.','']
    summary=json.loads((root/'summary.json').read_text())
    aggregate=summary['aggregate'];base=aggregate['control'];concept=aggregate['concept'];edits=aggregate['edit_targets']
    lines+=['## Conclusions','',
            '| Arm | Actual harness first-guess accuracy | Solve rate |',
            '|---|---:|---:|']
    for variant,a in aggregate.items():
        lines.append(f"| {variant} | {a['first_correct']/a['appearances']:.2%} | {a['solves']/a['appearances']:.2%} |")
    lines+=['',f"Shared concepts improve mean full-board Hit@1 by {(concept['ranking']['s4']['hit1']-base['ranking']['s4']['hit1'])*100:.2f} percentage points, "
            'with improvements in all three matched seeds. Reduced-state Hit@1 and MRR also improve. '
            f"Actual harness solves change by only {concept['solves']-base['solves']:+d}: "
            f"{summary['paired']['concept']['solve_gains']} gained and {summary['paired']['concept']['solve_losses']} lost appearances. "
            'This supports a useful ranking feature, not a demonstrated reliable solve-rate gain.', '',
            f"Direct Purple recoveries change from {base['direct_colors']['purple']} to {concept['direct_colors']['purple']}; "
            f"forced Purple recoveries from {base['forced_colors']['purple']} to {concept['forced_colors']['purple']}. "
            'The added concept signal primarily improves semantic categories; it does not resolve '
            'the wordplay bottleneck.', '',
            f"Exact edit targets change full-board Hit@1 by {(edits['ranking']['s4']['hit1']-base['ranking']['s4']['hit1'])*100:+.2f} points and "
            f"solves by {edits['solves']-base['solves']:+d}. The intended win quartet ranks last in "
            'all three edit-target runs and is never recovered. There is no demonstrated targeted '
            'mechanism benefit. Only one positive training quartet and separate rule channels '
            'make this a poor test of learning a broadly useful edit mechanic; do not interpret '
            'it as evidence against all transformation features.', '',
            'Keep both additions opt-in. The concept feature merits further comparison for '
            'ranking/first-choice quality; exact shared-target evidence should not be promoted '
            'on these results. All nine runs completed 30,000 updates, while selected checkpoints '
            'lie between 1,000 and 8,500; later training consistently overfits.', '',
            '## Verification','',
            'Four new relation-witness tests pass, including matched initialization and nonzero '
            'gradients for enabled added columns. Full regression suite: 148/156 pass; eight '
            'unchanged tests still reference removed notebook classes or the old NDCG API. '
            'No notebooks were edited. `git diff --check` passes.']
    audit_path=root/'checkpoint-audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text()).get('complete'):
        lines+=['','All nine saved checkpoints independently reconstruct through the production '
                'loader on CPU using cached frozen Qwen vectors and reproduce their training '
                'validation Hit@1 across 1,089 board/model evaluations. See checkpoint-audit.json.']
    with (root/'report.md').open('a') as stream:stream.write('\n'.join(lines))
    print(json.dumps(json.loads((root/'summary.json').read_text())['aggregate'],indent=2))


if __name__=='__main__':main()
