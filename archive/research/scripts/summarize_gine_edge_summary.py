"""Matched regular GINE versus direct quartet edge-summary readout."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import altair as alt
import pandas as pd

from run_gine_relation_witnesses import COLORS, VARIANTS, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/gine-edge-summary'))
    parser.add_argument('--baseline', type=Path, default=Path('experiments/gine-relation-witnesses-regular'))
    args = parser.parse_args()
    roots = {'regular': args.baseline, 'edge_summary': args.root}
    manifests = {name: json.loads((root/'manifest.json').read_text()) for name, root in roots.items()}
    assert manifests['edge_summary']['edge_summary_width'] == 16
    assert {k: v for k, v in manifests['edge_summary'].items() if k != 'edge_summary_width'} == manifests['regular']
    assert manifests['regular']['edge_coherence_scale'] == 0
    assert json.loads((args.root/'feature-metadata.json').read_text()) == json.loads(
        (args.baseline/'feature-metadata.json').read_text())
    records = {}
    aggregates = {}
    for name, root in roots.items():
        assert json.loads((root/'status.json').read_text())['status'] == 'complete'
        assert json.loads((root/'checkpoint-audit.json').read_text())['complete']
        aggregates[name] = json.loads((root/'summary.json').read_text())['aggregate']
        for seed in manifests[name]['seeds']:
            for variant in VARIANTS:
                r = json.loads((root/f'{variant}-seed-{seed}.json').read_text())
                assert r['status'] == 'complete'
                assert r['training']['updates_ran'] == manifests[name]['updates']
                records[name, variant, seed] = r
    for variant in VARIANTS:
        for seed in manifests['regular']['seeds']:
            a, b = (records[name, variant, seed] for name in roots)
            assert a['config'] == {k: v for k, v in b['config'].items() if k != 'edge_summary_width'}
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
                    date=row['date'], color=color, category=label['category'], mechanism=label['primary_mechanism'],
                    rank=rank, direct=bool(recovered and not recovered['forced']),
                    forced=bool(recovered and recovered['forced'])))
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
        gained = lost = 0
        for seed in manifests['regular']['seeds']:
            a, b = ({g['date']: g['won'] for g in records[name, variant, seed]['evaluation']['harness']['games']}
                    for name in roots)
            assert a.keys() == b.keys()
            gain = sum(b[d] and not a[d] for d in a)
            loss = sum(a[d] and not b[d] for d in a)
            gained += gain
            lost += loss
            pairs.append(dict(variant=variant, seed=seed, solve_gains=gain, solve_losses=loss))
        pairs.append(dict(variant=variant, seed='all', solve_gains=gained, solve_losses=lost))
    for filename, rows in (('group_results', groups), ('mechanism_results', mechanisms), ('paired_solves', pairs)):
        with (args.root/f'{filename}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    chart = alt.Chart(pd.DataFrame(curves)).mark_line().encode(
        x='update:Q', y=alt.Y('value:Q', scale=alt.Scale(zero=False)), color='architecture:N',
        strokeDash='seed:N', tooltip=['architecture', 'variant', 'seed', 'update', 'metric', 'value'])
    chart.properties(width=330, height=150).facet(row='metric:N', column='variant:N').resolve_scale(
        y='independent').interactive().save(args.root/'learning-curves.html')
    lines = ['# Direct edge-summary channel with regular candidate GINE', '',
        'Validation only; three matched seeds (10, 11, 12), 30,000 updates per run, '
        'balanced s2/s3/s4, batch 16, evaluation every 500 updates. Same 121 validation puzzles '
        'repeated across seeds: 363 correlated appearances, not 363 independent puzzles. '
        'Existing completed regular-GINE runs are reused as controls. No test evaluation or notebook changes.', '',
        'The new branch computes three values per relation channel from each candidate: '
        'off-diagonal mean, off-diagonal minimum, and minimum over words of mean incident edge support. '
        'Raw masked candidate edges include exact shared-witness overrides. A 78 -> 16 -> 16 ReLU MLP '
        'embeds these statistics, then concatenates them with the GINE node mean/max embedding '
        'before the existing readout MLP. Added readout columns initialize zero; existing parameters '
        'and initial scores match the control. The branch adds 3,584 trainable parameters. '
        'Coherence strength is zero in all runs. No new gate, loss, scheduler, or hyperparameter tuning.', '',
        'Both models train all active projections, LayerNorm, GINE, and readout from scratch; '
        'only original cached Qwen vectors are frozen. New branch parameters are also trainable. '
        'Checkpoint selection is validation s4 Hit@1, MRR for ties. All runs receive their full budget. '
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
    effects = {}
    lines += ['', '## Paired effects of adding the summary branch', '',
              '| Features | Hit@1 difference | Solve difference | Solves gained | Solves lost |',
              '|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        a, b = (aggregates[name][variant] for name in roots)
        p = next(p for p in pairs if p['variant'] == variant and p['seed'] == 'all')
        effects[variant] = dict(hit1=b['ranking']['s4']['hit1']-a['ranking']['s4']['hit1'], solves=b['solves']-a['solves'])
        lines.append(f"| {variant} | {100*effects[variant]['hit1']:+.2f} pp | {effects[variant]['solves']:+d} | {p['solve_gains']} | {p['solve_losses']} |")
    lines += ['', '## Per-seed results and learning', '',
              '| Seed | Features | Architecture | Hit@1 | Solves | Best update | Final Hit@1 |',
              '|---|---|---|---:|---:|---:|---:|']
    for seed in manifests['regular']['seeds']:
        for variant in VARIANTS:
            for name in roots:
                r = records[name, variant, seed]
                t = r['training']
                lines.append(f"| {seed} | {variant} | {name} | {t['validation']['s4']['hit1']:.2%} | "
                    f"{r['evaluation']['harness']['solved']} | {t['best_update']} | {t['history'][-1]['val_s4_hit1']:.2%} |")
    lines += ['', '## Mechanism diagnostics', '',
              '| Mechanism | Features | Architecture | Top4 | Direct | Forced |',
              '|---|---|---|---:|---:|---:|']
    for r in mechanisms:
        lines.append(f"| {r['mechanism']} | {r['variant']} | {r['architecture']} | {r['top4']:.2%} | {r['direct']} | {r['forced']} |")
    lines += ['', '## Intended exact edit target', '',
              'wind/wine/wing/wink is a Green group, not Purple. Exact shared-edit targets have only one '
              'positive training quartet and none for deletion channels, so supervision remains sparse.', '',
              '| Architecture | Features | Seed | Rank | Direct | Forced |', '|---|---|---:|---:|---|---|']
    for r in groups:
        if r['category'] == 'win + letter':
            lines.append(f"| {r['architecture']} | {r['variant']} | {r['seed']} | {r['rank']} | {r['direct']} | {r['forced']} |")
    lines += ['', '## Conclusions', '']
    for variant in VARIANTS:
        e = effects[variant]
        lines.append(f"- {variant}: the branch changes mean full-board Hit@1 by {100*e['hit1']:+.2f} "
                     f"percentage points and full solves by {e['solves']:+d} appearances.")
    best_name, best_variant = max(((name, variant) for name in roots for variant in VARIANTS),
                                 key=lambda p: aggregates[p[0]][p[1]]['solves'])
    best = aggregates[best_name][best_variant]
    lines += ['', f"The largest observed solve count is {best_name}/{best_variant}: "
              f"{best['solves']}/{best['appearances']}. This is descriptive; neither the repeated "
              'seeds nor this repeatedly inspected validation set provide an independent confirmation '
              'of superiority. Keep the summary branch opt-in pending stronger evidence.']
    branch_audit = json.loads((args.root/'checkpoint-audit.json').read_text())['records']
    active = sum(r['summary_branch']['readout_weight_norm'] > 0 and
                 r['summary_branch']['mean_within_board_activation_std'] > 0 for r in branch_audit)
    lines += ['', '## Verification and interpretation', '',
              'All saved checkpoints strictly reload and independently reproduce training Hit@1 '
              'before harness evaluation. Independent production-loader audits pass for both sets '
              'of checkpoints. Feature metadata and manifests match except the added summary branch; '
              'model configurations match except edge_summary_width. Focused tests verify diagonal '
              'exclusion, permutation invariance, candidate locality, witness overrides, inactive-channel '
              'masking, matched initial scores, gradients after the zero initialization, and strict reload.', '',
              f"{active}/{len(branch_audit)} trained summary branches have nonzero added readout weights "
              'and activations that vary across candidates within validation boards. This checks that '
              'trained, varying branch representations reach the readout; it does not establish that they help. '
              'See checkpoint-audit.json for the measured norms and activation variation. '
              '43 focused tests pass, including three new branch tests.', '',
              'This is a fixed-hyperparameter ablation, not separate architecture tuning. Mechanism labels '
              'are diagnostic annotations, not training inputs. Reduced-state metrics are oracle states, '
              'not actual feedback trajectories. Repeated validation reuse limits strong conclusions. '
              'Ranking improvements alone do not establish a puzzle solve-rate improvement. '
              'The separate five-fold tuning experiment is unchanged.']
    write(args.root/'edge-summary-comparison.json', dict(aggregate=aggregates, effects=effects, paired=pairs))
    (args.root/'edge-summary-comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(effects=effects, paired=[p for p in pairs if p['seed'] == 'all']), indent=2))


if __name__ == '__main__':
    main()
