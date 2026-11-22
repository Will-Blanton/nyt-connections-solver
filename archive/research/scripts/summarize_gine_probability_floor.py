"""Compare weak probability floors with the matched probability-mass baseline."""
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from audit_letter_edit_coverage import EXAMPLES
from summarize_gine_group_contrastive import purple_status


ROOT = Path('experiments/gine-probability-floor')
VARIANTS = ('probability-mass', 'floor-0.05', 'floor-0.1')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    audit = json.loads((ROOT / 'probability-audit.json').read_text())
    entries = json.loads((ROOT / 'runs.json').read_text())
    assert len(entries) == 9 and len(audit['records']) == 1089
    runs, games_by_key = [], {}
    for entry in entries:
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        assert result['status'] == 'complete' and harness['split'] == 'val'
        assert result['training']['updates_ran'] == 15000
        games = harness['games']
        assert len(games) == len({g['date'] for g in games}) == 121
        games_by_key[entry['variant'], entry['seed']] = {g['date']: g for g in games}
        runs.append(dict(**entry, checkpoint=result['checkpoint'], ranking=result['training']['validation'],
                         best_update=result['training']['best_update'],
                         solves=sum(g['won'] for g in games), first_correct=sum(g['first_guess_correct'] for g in games),
                         colors=dict(Counter(c for g in games for c in g['solved_colors'])),
                         purple=dict(Counter(purple_status(g) for g in games)),
                         target_purple=dict(Counter(purple_status(g) for g in games if g['date'] in EXAMPLES))))
    aggregate, paired = {}, {}
    for variant in VARIANTS:
        rows = [r for r in runs if r['variant'] == variant]
        boards = [r for r in audit['records'] if r['variant'] == variant]
        assert len(boards) == 363
        probabilities = [[math.exp(x) for x in b['gold_log_probabilities']] for b in boards]
        ranks = [rank for b in boards for rank in b['gold_ranks']]
        aggregate[variant] = dict(
            ranking={s: {k: mean(r['ranking'][s][k] for r in rows) for k in ('hit1', 'mrr', 'hit4')}
                     for s in ('s4', 's3', 's2')},
            solves=sum(r['solves'] for r in rows), first_correct=sum(r['first_correct'] for r in rows),
            color_recovery={c: sum(r['colors'].get(c, 0) for r in rows) for c in audit['colors']},
            purple={k: sum(r['purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')},
            target_purple={k: sum(r['target_purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')},
            all_group_ranking=dict(mean_rank=mean(ranks), mrr=mean(1/r for r in ranks), top4_recall=mean(r <= 4 for r in ranks)),
            probability=dict(correct_mass=mean(sum(p) for p in probabilities),
                             other_correct_mass=mean(sum(p)-max(p) for p in probabilities),
                             minimum_correct_probability=mean(min(p) for p in probabilities),
                             correct_groups_above_floor=mean(p >= .025 for row in probabilities for p in row),
                             log_floor_shortfall=mean(max(0., math.log(.025)-p) for b in boards for p in b['gold_log_probabilities']),
                             dominant_share_of_correct_mass=mean(max(p)/sum(p) for p in probabilities)),
            color_top4={color: mean(b['gold_ranks'][i] <= 4 for b in boards) for i, color in enumerate(audit['colors'])})
        assert abs(mean(1 in b['gold_ranks'] for b in boards)-aggregate[variant]['ranking']['s4']['hit1']) < 1e-9
        if variant != 'probability-mass':
            counters = Counter()
            for seed in manifest['seeds']:
                a, b = games_by_key['probability-mass', seed], games_by_key[variant, seed]
                assert a.keys() == b.keys()
                for date in a:
                    counters['solve_gain'] += b[date]['won'] and not a[date]['won']
                    counters['solve_loss'] += a[date]['won'] and not b[date]['won']
                    p, q = purple_status(a[date]), purple_status(b[date])
                    counters['purple_gain'] += p == 'missed' and q != 'missed'
                    counters['purple_loss'] += q == 'missed' and p != 'missed'
            paired[variant] = dict(counters)
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs, paired=paired)
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Weak probability-floor penalty on GINE', '',
             'Validation only: three matched seeds (10/11/12), 121 puzzles per seed, '
             '15,000 updates per run. Reuse the completed probability-mass controls '
             'from `gine-curriculum/mixed`. Six new runs, one at a time. Repeated '
             'seeds reuse the same puzzles; 363 appearances are not 363 independent '
             'puzzles. No notebooks or test-set evaluation.', '', '## Setup', '',
             '`loss = probability_mass_loss + weight * mean_correct relu(log(beta/m) - log p(group))`.', '',
             'Fix beta=0.1 and try weights 0.05 and 0.1. With m=4/3/2 correct groups '
             'on 16/12/8-word training states, each correct group has a soft floor of '
             '2.5%/3.33%/5%. The penalty becomes zero above the floor; it does not '
             'require equal positive probabilities or assign floors to incorrect '
             'candidates. The floor is an optimization target, not a guaranteed '
             'prediction bound.', '',
             'Hold architecture, initialization, features and training settings fixed: '
             'standalone GINE, width 128, two layers, readout 128, original 15 edges, '
             'frozen Qwen vectors, fully trainable active norm/projection/GINE/readout, '
             'no phonetics or letter-edit additions. Batch 16, LR 0.0003016685, '
             'weight decay 0.0000950089, dropout 0, 5,000 updates per s2/s3/s4 '
             'state size, matched per-seed schedules and batch permutations. No '
             'early stopping, pruning, LR schedule or tuning beyond the two stated '
             'weights. Evaluate every 250 updates; select checkpoints by s4 Hit@1 '
             'with MRR tie-break. Compare these selected checkpoints primarily by '
             'harness solve rate; do not select checkpoints separately by harness.', '',
             'Harness: normal checkpoint loader, beam 20, full-board group-score '
             'sums, four mistakes, public-feedback pruning and shuffle seed 0.', '',
             '## Aggregate', '', '| Metric | Probability mass | Floor weight 0.05 | Floor weight 0.1 |',
             '|---|---:|---:|---:|']
    for size in ('s4', 's3', 's2'):
        for key in ('hit1', 'mrr', 'hit4'):
            values = [aggregate[v]['ranking'][size][key] for v in VARIANTS]
            cells = [f'{x:.4f}' if key == 'mrr' else f'{x:.2%}' for x in values]
            lines.append(f'| {size} {key} | '+' | '.join(cells)+' |')
    for key in ('solves', 'first_correct'):
        lines.append(f'| Harness {key} | '+' | '.join(f"{aggregate[v][key]}/363 ({aggregate[v][key]/363:.2%})" for v in VARIANTS)+' |')
    for color in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f'| {color} recovered | '+' | '.join(str(aggregate[v]['color_recovery'][color])+'/363' for v in VARIANTS)+' |')
    for key in ('selected', 'forced'):
        lines.append(f'| Purple {key} | '+' | '.join(str(aggregate[v]['purple'][key])+'/363' for v in VARIANTS)+' |')
    lines.append('| Targeted letter-edit Purple recovered | '+' | '.join(str(24-aggregate[v]['target_purple']['missed'])+'/24' for v in VARIANTS)+' |')
    lines += ['', '## Did it lift the other correct groups?', '',
              'Computed on all 1,820 full-board candidates at the same selected '
              'checkpoints, using saved frozen vectors and no additional embedding '
              'inference. Other-correct mass excludes the highest-probability true '
              'group, whether or not that group is the top prediction.', '',
              '| Metric | Probability mass | Weight 0.05 | Weight 0.1 |', '|---|---:|---:|---:|']
    for key in ('correct_mass', 'other_correct_mass', 'minimum_correct_probability',
                'correct_groups_above_floor', 'dominant_share_of_correct_mass', 'log_floor_shortfall'):
        cells = [f"{aggregate[v]['probability'][key]:.4f}" if key == 'log_floor_shortfall' else
                 f"{aggregate[v]['probability'][key]:.3e}" if key == 'minimum_correct_probability' else
                 f"{aggregate[v]['probability'][key]:.3%}" for v in VARIANTS]
        lines.append(f'| {key} | '+' | '.join(cells)+' |')
    for key in ('mean_rank', 'mrr', 'top4_recall'):
        cells = [f"{aggregate[v]['all_group_ranking'][key]:.3f}" for v in VARIANTS]
        lines.append(f'| True-group {key} | '+' | '.join(cells)+' |')
    lines += ['', '## Per seed', '', '| Seed | Variant | s4 Hit@1 | Solves | Purple recovered | Selected update |',
              '|---|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | "
                     f"{r['solves']}/121 | {r['colors'].get('purple',0)}/121 | {r['best_update']} |")
    lines += ['', '## Paired changes versus probability mass', '']
    for variant, p in paired.items():
        lines.append(f"- {variant}: {p['solve_gain']} new solves, {p['solve_loss']} lost solves; "
                     f"{p['purple_gain']} new Purple recoveries, {p['purple_loss']} lost Purple recoveries.")
    lines += ['', 'All experiment weights and settings are recorded in `manifest.json`; '
              'full metrics and checkpoint paths are in `summary.json`, with per-board '
              'gold-group ranks and log probabilities in `probability-audit.json`. '
              'The default training objective remains unchanged. These fixed-setting '
              'validation comparisons are not independent confirmation on unseen puzzles.', '']
    lines += ['## Interpretation', '',
              'Weight 0.1 is the promising validation variant: 97/363 solves versus '
              '86/363 (26.72% versus 23.69%, +3.03 percentage points). Per-seed solve '
              'counts are 33 to 33, 28 to 33, and 25 to 31; the gain is not concentrated '
              'in one seed. The paired changes are 30 new solves and 19 lost solves, '
              'not eleven universally improved puzzles. Weight 0.05 instead loses '
              'ten net solves. Both regularized variants have exactly 36.36% aggregate '
              'raw s4 Hit@1, but different solve rates (20.94% versus 26.72%). '
              'This supports evaluating the harness rather than selecting a penalty '
              'weight on standalone Hit@1 alone.', '',
              'The penalty modestly changes the intended behavior on validation: '
              'other-correct probability mass rises from 0.349% to 0.424%, mean '
              'log-floor shortfall falls from 10.0819 to 6.6726, and the strongest '
              'correct group\'s mean conditional share falls from 92.06% to 90.48%. '
              'The easy correct choice still dominates. Only 14.88% of correct '
              'groups meet the 2.5% full-board floor, versus 13.98% for control; '
              'the regularizer does not enforce the floor on unseen validation '
              'groups. Total correct mass is slightly lower (23.16% to 22.63%), '
              'so probability redistribution alone is not a sufficient explanation '
              'for the solve improvement.', '',
              'This is not a direct Purple breakthrough. Purple recovery rises '
              '100 to 106, but early Purple selections fall 62 to 61 and forced '
              'last-group recoveries rise 38 to 45. Curated letter-edit Purple '
              'recoveries fall 5/24 to 4/24. Blue recovery has the largest absolute '
              'color gain, 122 to 136. The new letter-edit edge features remain '
              'disabled in this comparison.', '',
              'Retain weight 0.1 as an opt-in validation candidate, not an '
              'independently confirmed replacement baseline. The same puzzles '
              'have been used repeatedly for selection and there is no new '
              'test-set evidence. Inference uses the unchanged single-GINE '
              'architecture, with no extra inference stage or floor computation. '
              'All six runs completed 15,000 updates; selected checkpoints occur '
              'at 1,750 to 5,500 updates. Fourteen targeted loss/training regression '
              'tests passed. All six checkpoints ran through the normal harness '
              'loader, and the independent rank/probability audit reproduced '
              'the s4 Hit@1 metrics for all nine runs.', '']
    Path('experiments/gine-probability-floor.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
