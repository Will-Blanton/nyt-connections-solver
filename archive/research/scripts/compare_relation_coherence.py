"""Compare matched regular/coherence GINE runs with shared relation witnesses."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import altair as alt
import pandas as pd

from run_gine_relation_witnesses import COLORS, VARIANTS, summarize, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regular', type=Path, default=Path('experiments/gine-relation-witnesses-regular'))
    parser.add_argument('--coherence', type=Path, default=Path('experiments/gine-relation-witnesses'))
    args = parser.parse_args()
    roots = {'regular': args.regular, 'coherence': args.coherence}
    manifests = {name: json.loads((root/'manifest.json').read_text()) for name, root in roots.items()}
    assert manifests['regular']['edge_coherence_scale'] == 0
    assert manifests['coherence']['edge_coherence_scale'] == 1
    assert {k: v for k, v in manifests['regular'].items() if k != 'edge_coherence_scale'} == {
        k: v for k, v in manifests['coherence'].items() if k != 'edge_coherence_scale'}
    assert json.loads((args.regular/'feature-metadata.json').read_text()) == json.loads(
        (args.coherence/'feature-metadata.json').read_text())
    records = {}
    for name, root in roots.items():
        assert json.loads((root/'checkpoint-audit.json').read_text())['complete'], 'Finish production-loader audit first'
        for seed in manifests[name]['seeds']:
            for variant in VARIANTS:
                run = json.loads((root/f'{variant}-seed-{seed}.json').read_text())
                assert run['status'] == 'complete'
                assert run['training']['history'][-1]['update'] == manifests[name]['updates']
                assert run['config']['edge_coherence_scale'] == manifests[name]['edge_coherence_scale']
                records[name, variant, seed] = run
    for variant in VARIANTS:
        for seed in manifests['regular']['seeds']:
            a, b = (records[name, variant, seed] for name in roots)
            assert {k: v for k, v in a['config'].items() if k != 'edge_coherence_scale'} == {
                k: v for k, v in b['config'].items() if k != 'edge_coherence_scale'}
    # Only rewrite the new run's summary; preserve the previous detailed report.
    summarize(args.regular, manifests['regular'])
    aggregates = {name: json.loads((root/'summary.json').read_text())['aggregate'] for name, root in roots.items()}
    with open('experiments/group-mechanism-audit/group_mechanisms.csv') as stream:
        labels = {(r['date'], r['color']): r for r in csv.DictReader(stream) if r['original_split'] == 'val'}
    groups, curves, mechanisms, pairs = [], [], [], []
    for (architecture, variant, seed), run in records.items():
        games = {g['date']: g for g in run['evaluation']['harness']['games']}
        for row in run['evaluation']['gold_ranks']:
            for color, rank in zip(COLORS, row['gold_ranks']):
                label = labels[row['date'], color]
                recovered = next((h for h in games[row['date']]['history']
                                  if h['color'] == color and h['outcome'] == 'correct'), None)
                groups.append(dict(architecture=architecture, variant=variant, seed=seed,
                    date=row['date'], color=color, category=label['category'],
                    mechanism=label['primary_mechanism'], rank=rank,
                    direct=bool(recovered and not recovered['forced']), forced=bool(recovered and recovered['forced'])))
        for row in run['training']['history']:
            for metric in ('train_loss', 'val_s4_hit1', 'val_s3_hit1', 'val_s2_hit1'):
                curves.append(dict(architecture=architecture, variant=variant, seed=str(seed),
                                   update=row['update'], metric=metric, value=row[metric]))
    for architecture in roots:
        for variant in VARIANTS:
            for mechanism in sorted({r['mechanism'] for r in groups}):
                rows = [r for r in groups if (r['architecture'], r['variant'], r['mechanism']) ==
                        (architecture, variant, mechanism)]
                mechanisms.append(dict(architecture=architecture, variant=variant, mechanism=mechanism,
                    appearances=len(rows), top4=mean(r['rank'] <= 4 for r in rows),
                    mrr=mean(1/r['rank'] for r in rows), direct=sum(r['direct'] for r in rows),
                    forced=sum(r['forced'] for r in rows)))
    for variant in VARIANTS:
        gains = losses = 0
        for seed in manifests['regular']['seeds']:
            a, b = ({g['date']: g['won'] for g in records[name, variant, seed]['evaluation']['harness']['games']}
                    for name in roots)
            assert a.keys() == b.keys()
            gained = sum(b[d] and not a[d] for d in a)
            lost = sum(a[d] and not b[d] for d in a)
            gains += gained
            losses += lost
            pairs.append(dict(variant=variant, seed=seed, coherence_solve_gains=gained, coherence_solve_losses=lost))
        pairs.append(dict(variant=variant, seed='all', coherence_solve_gains=gains, coherence_solve_losses=losses))
    for filename, rows in (('group_results', groups), ('mechanism_results', mechanisms), ('paired_solves', pairs)):
        with (args.regular/f'{filename}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    chart = alt.Chart(pd.DataFrame(curves)).mark_line().encode(
        x='update:Q', y=alt.Y('value:Q', scale=alt.Scale(zero=False)), color='architecture:N',
        strokeDash='seed:N', tooltip=['architecture', 'variant', 'seed', 'update', 'metric', 'value'])
    chart.properties(width=330, height=150).facet(row='metric:N', column='variant:N').resolve_scale(
        y='independent').interactive().save(args.regular/'learning-curves.html')
    lines = ['# Regular versus coherence GINE with new relation features', '',
        'Validation only. Same 121 puzzles reused across three matched seeds; 363 appearances are correlated. '
        'Regular GINE uses coherence strength 0; earlier runs use strength 1. '
        'All other model configurations, fresh initial parameters, minibatch seeds, splits, '
        '30,000-update budgets and checkpoint selection rules match. '
        'Each of the three feature arms has the same 26-channel parameter shape; unused channels are masked. '
        'Only original Qwen vectors are frozen. No hyperparameter retuning, scheduler or auxiliary loss changes. '
        'Checkpoint selection uses s4 Hit@1, then MRR. '
        'Harness: beam 20, original full-board score sums, four mistakes, shuffle seed 0.', '',
        '| Features | Architecture | s4 Hit@1 | MRR | Hit@4 | s3 Hit@1 | s2 Hit@1 | First guess | Solves | Direct Purple | Forced Purple |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for name in roots:
            a = aggregates[name][variant]
            lines.append(f"| {variant} | {name} | {a['ranking']['s4']['hit1']:.2%} | {a['ranking']['s4']['mrr']:.4f} | "
                f"{a['ranking']['s4']['hit4']:.2%} | {a['ranking']['s3']['hit1']:.2%} | {a['ranking']['s2']['hit1']:.2%} | "
                f"{a['first_correct']/a['appearances']:.2%} | {a['solves']}/{a['appearances']} | "
                f"{a['direct_colors']['purple']} | {a['forced_colors']['purple']} |")
    lines += ['', '## Matched coherence effects', '',
              '| Features | s4 Hit@1 difference | Solve difference | Solves gained | Solves lost |',
              '|---|---:|---:|---:|---:|']
    effects = {}
    for variant in VARIANTS:
        a, b = (aggregates[name][variant] for name in roots)
        p = next(p for p in pairs if p['variant'] == variant and p['seed'] == 'all')
        effects[variant] = dict(hit1=b['ranking']['s4']['hit1']-a['ranking']['s4']['hit1'],
                               solves=b['solves']-a['solves'])
        lines.append(f"| {variant} | {100*effects[variant]['hit1']:+.2f} pp | {effects[variant]['solves']:+d} | "
                     f"{p['coherence_solve_gains']} | {p['coherence_solve_losses']} |")
    lines += ['', 'Differences above are coherence minus regular. Feature effects must also be compared '
              'against the control with the same architecture:', '',
              '| Architecture | Added feature | s4 Hit@1 difference from control | Solve difference from control |',
              '|---|---|---:|---:|']
    for name in roots:
        base = aggregates[name]['control']
        for variant in VARIANTS[1:]:
            a = aggregates[name][variant]
            lines.append(f"| {name} | {variant} | {100*(a['ranking']['s4']['hit1']-base['ranking']['s4']['hit1']):+.2f} pp | "
                         f"{a['solves']-base['solves']:+d} |")
    lines += ['', '## Per-seed results and training', '',
              '| Seed | Features | Architecture | s4 Hit@1 | Solves | Best update | Final s4 Hit@1 |',
              '|---|---|---|---:|---:|---:|---:|']
    for seed in manifests['regular']['seeds']:
        for variant in VARIANTS:
            for name in roots:
                r = records[name, variant, seed]
                t = r['training']
                lines.append(f"| {seed} | {variant} | {name} | {t['validation']['s4']['hit1']:.2%} | "
                    f"{r['evaluation']['harness']['solved']} | {t['best_update']} | {t['history'][-1]['val_s4_hit1']:.2%} |")
    lines += ['', '## Mechanism-specific retrieval and recovery', '',
              '| Mechanism | Features | Architecture | Top4 | Direct | Forced |',
              '|---|---|---|---:|---:|---:|']
    for r in mechanisms:
        lines.append(f"| {r['mechanism']} | {r['variant']} | {r['architecture']} | {r['top4']:.2%} | {r['direct']} | {r['forced']} |")
    lines += ['', '## Exact win target', '',
              'The intended wind/wine/wing/wink target is Green. Exact edit witnesses have only one '
              'positive training quartet, and none for the deletion channels. This experiment does not '
              'establish the value of broadly supervised transformation features.', '',
              '| Architecture | Features | Seed | Rank | Direct | Forced |',
              '|---|---|---:|---:|---|---|']
    for r in groups:
        if r['category'] == 'win + letter':
            lines.append(f"| {r['architecture']} | {r['variant']} | {r['seed']} | {r['rank']} | {r['direct']} | {r['forced']} |")
    lines += ['', '## Conclusions', '',
              'Coherence is not a consistent solve-rate improvement at these fixed settings. '
              'The interaction depends on the feature arm:', '']
    for variant in VARIANTS:
        e = effects[variant]
        lines.append(f"- {variant}: coherence changes mean full-board Hit@1 by {100*e['hit1']:+.2f} percentage points "
                     f"and complete solves by {e['solves']:+d} appearances.")
    best_name, best_variant = max(((name, variant) for name in roots for variant in VARIANTS),
                                 key=lambda pair: aggregates[pair[0]][pair[1]]['solves'])
    best = aggregates[best_name][best_variant]
    lines += ['', f"The largest solve count in this comparison is {best_name}/{best_variant}: "
              f"{best['solves']}/{best['appearances']}. This is descriptive, not an independent "
              'confirmation of superiority. Seed variation and repeated validation reuse limit '
              'strong conclusions. Adding a feature may improve ranking without improving '
              'full-partition decisions; the control comparisons above are necessary before promotion.', '',
              'All runs finish their full budget. The learning curves show late overfitting; '
              'longer training does not eliminate the ranking-versus-solve discrepancy. '
              'This ablation does not replace the ongoing separate five-fold tuning comparison.']
    lines += ['', '## Verification and limits', '',
              'All 18 runs strictly reload parameters and independently reproduce training s4 Hit@1 '
              'before harness evaluation. Manifests and model configurations match except coherence strength. '
              'A focused regression test verifies identical initial parameters across coherence strengths. '
              'Separate checkpoint-audit.json files validate reconstruction through the production loader. '
              'This is a fixed-hyperparameter ablation, not a comparison of separately tuned architectures. '
              'Repeated reuse of this validation set makes the results exploratory. The ongoing five-fold '
              'tuning is separate and unchanged. No test evaluation or notebook changes.']
    write(args.regular/'coherence-comparison.json', dict(aggregate=aggregates, effects=effects, paired=pairs))
    (args.regular/'coherence-comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(aggregate=aggregates, effects=effects), indent=2))


if __name__ == '__main__':
    main()
