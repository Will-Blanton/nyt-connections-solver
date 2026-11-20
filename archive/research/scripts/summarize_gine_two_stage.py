"""Summarize staged-logit GINE against its baseline and logit ensemble."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from audit_letter_edit_coverage import EXAMPLES


ROOT = Path('experiments/gine-two-stage')


def purple_status(game):
    correct = 0
    for step in game['history']:
        if step['outcome'] == 'correct':
            if step['color'] == 'purple':
                return 'forced' if correct == 3 else 'selected'
            correct += 1
    return 'missed'


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    runs = []
    games_by_key = {}
    for entry in json.loads((ROOT / 'runs.json').read_text()):
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        assert result['status'] == 'complete' and harness['split'] == 'val'
        games = harness['games']
        assert len(games) == 121 and len({g['date'] for g in games}) == 121
        games_by_key[entry['variant'], entry['seed']] = {g['date']: g for g in games}
        runs.append({**entry, 'checkpoint': result['checkpoint'],
                     'ranking': result['training']['validation'],
                     'best_update': result['training']['best_update'],
                     'updates_ran': result['training']['updates_ran'],
                     'solves': sum(g['won'] for g in games),
                     'first_correct': sum(g['first_guess_correct'] for g in games),
                     'correct_groups': sum(g['correct_groups'] for g in games),
                     'color_recovery': dict(Counter(c for g in games for c in g['solved_colors'])),
                     'purple': dict(Counter(purple_status(g) for g in games)),
                     'target_purple': dict(Counter(purple_status(g) for g in games if g['date'] in EXAMPLES)),
                     'harness_seconds': sum(g['seconds'] for g in games)})
    assert len(runs) == 9
    aggregate = {}
    paired = {}
    for variant in ('baseline', 'residual', 'ensemble'):
        rows = [r for r in runs if r['variant'] == variant]
        aggregate[variant] = dict(
            ranking={s: {k: mean(r['ranking'][s][k] for r in rows) for k in ('hit1', 'mrr', 'hit4')}
                     for s in ('s4', 's3', 's2')},
            color_top4={c: mean(r['ranking']['s4']['by_color'][c]['top4'] for r in rows)
                        for c in ('yellow', 'green', 'blue', 'purple')},
            solves=sum(r['solves'] for r in rows), first_correct=sum(r['first_correct'] for r in rows),
            mean_correct_groups=sum(r['correct_groups'] for r in rows)/363,
            color_recovery={c: sum(r['color_recovery'].get(c, 0) for r in rows)
                            for c in ('yellow', 'green', 'blue', 'purple')},
            purple={k: sum(r['purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')},
            target_purple={k: sum(r['target_purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')})
        if variant == 'baseline':
            continue
        counts = Counter()
        for seed in manifest['seeds']:
            control, other = games_by_key['baseline', seed], games_by_key[variant, seed]
            assert control.keys() == other.keys()
            for date in control:
                a, b = control[date], other[date]
                counts['solve_gain'] += b['won'] and not a['won']
                counts['solve_loss'] += a['won'] and not b['won']
                counts['purple_gain'] += purple_status(a) == 'missed' and purple_status(b) != 'missed'
                counts['purple_loss'] += purple_status(b) == 'missed' and purple_status(a) != 'missed'
        paired[variant] = dict(counts)
    output = dict(manifest=manifest, aggregate=aggregate, paired=paired, runs=runs)
    (ROOT / 'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = ['# Two-stage additive GINE', '',
             'Validation only: 121 puzzles, three matched baseline seeds (10/11/12). '
             'The three seeds reuse the same puzzles and are not independent held-out samples. '
             'No notebooks or test-set evaluation.', '',
             '## Setup', '',
             'Reuse the completed original-15-edge mixed-state GINE controls from '
             '`gine-curriculum` (15,000 updates each). Freeze each entire selected baseline. '
             'Train a separate two-layer GINE correction with width 64 and readout width 64 '
             '(baseline widths 128/128). The correction has its own trainable LayerNorm and '
             'text projection, initialized from the same shared text-only reference as the '
             'controls; fresh GINE weights use seeds 110/111/112. Only the original Qwen '
             'embeddings and first-stage model are frozen. All existing 15 edge features '
             'are available to both stages; no phonetics, letter edits or hypergraph changes.', '',
             'Final candidate logit = frozen baseline logit + 0.25 × correction logit. '
             'Zero-initialize the correction final output layer, preserving the baseline at '
             'update 0. Optimize the existing correct-choice probability-mass loss for '
             '15,000 additional updates, batch 16, balanced mixed s2/s3/s4 states, fixed '
             'LR 0.0003016685 and weight decay 0.0000950089. Evaluate every 250 updates, '
             'select by s4 Hit@1 then MRR, including the unchanged baseline at update 0. '
             'No early stopping, auxiliary loss or LR scheduling. Assert frozen baseline '
             'tensors remain bit-for-bit unchanged.', '',
             'Ensemble control: equally average logits from already trained baseline pairs '
             '(10,11), (11,12), (12,10). These require the same aggregate 30,000 training '
             'updates per pair as two stages, but use two full-width models, so capacity '
             'is not matched. Both approaches add model inference work. The ensemble '
             'pairs share components and are correlated. No ensemble weight tuning.', '',
             'All checkpoints contain both components and load through the normal model '
             'registry/harness. Harness: beam 20, original full-board group-score sums, '
             'four mistakes, public-feedback pruning, shuffle seed 0.', '',
             '## Aggregate', '', '| Metric | Baseline | Residual stage | Ensemble |',
             '|---|---:|---:|---:|']
    variants = ('baseline', 'residual', 'ensemble')
    for s in ('s4', 's3', 's2'):
        for k in ('hit1', 'mrr', 'hit4'):
            values = [aggregate[v]['ranking'][s][k] for v in variants]
            cells = [f'{x:.4f}' if k == 'mrr' else f'{x:.2%}' for x in values]
            lines.append(f'| {s} {k} | '+' | '.join(cells)+' |')
    for k in ('solves', 'first_correct'):
        lines.append(f'| Harness {k} | '+' | '.join(f"{aggregate[v][k]}/363 ({aggregate[v][k]/363:.2%})" for v in variants)+' |')
    for c in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f'| {c} recovered | '+' | '.join(f"{aggregate[v]['color_recovery'][c]}/363" for v in variants)+' |')
    lines.append('| Purple selected before final four | '+' | '.join(str(aggregate[v]['purple']['selected'])+'/363' for v in variants)+' |')
    lines.append('| Targeted letter-edit Purple recovered | '+' | '.join(str(24-aggregate[v]['target_purple']['missed'])+'/24' for v in variants)+' |')
    lines += ['', 'The targeted eight letter-edit puzzles are an analysis subset only; their '
              'new features are intentionally disabled in this experiment.', '',
              '## Per seed', '', '| Seed | Variant | s4 Hit@1 | s3 Hit@1 | s2 Hit@1 | Solves | Purple recovered | Selected update within run |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        cells=[str(r['seed']),r['variant']]+[f"{r['ranking'][s]['hit1']:.2%}" for s in ('s4','s3','s2')]
        cells += [str(r['solves'])+'/121',str(r['color_recovery'].get('purple',0))+'/121',str(r['best_update'])]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Paired changes versus baseline', '']
    for v, p in paired.items():
        lines.append(f"- {v}: {p['solve_gain']} new solves, {p['solve_loss']} lost solves; "
                     f"{p['purple_gain']} new Purple recoveries, {p['purple_loss']} lost Purple recoveries.")
    lines += ['', '## Interpretation', '',
              'The residual adds four net full-board correct top choices and three net '
              'harness solves across 363 puzzle-seed attempts. All three net harness '
              'solves come from seed 12 (25 to 28); seed 10 stays at 33, and seed 11 '
              'rejects its trained correction in favor of the unchanged baseline. '
              'Selected residual updates are 1,250, 0, and 500 despite completing all '
              '15,000 updates each. Later training lowers training loss while validation '
              'performance declines.', '',
              'Purple recovery rises from 100 to 103, but selections before the forced '
              'last four fall from 62 to 57. The aggregate gain therefore does not '
              'demonstrate stronger direct Purple recognition. The eight curated '
              'letter-edit puzzles have identical per-seed Purple recovery outcomes '
              'between baseline and residual, not just identical totals.', '',
              'The simple ensemble has better aggregate ranking and solve rate than '
              'the residual, but its solve improvement is concentrated in one pair '
              '(seeds 12 and 10), and it uses more capacity. This is weak evidence '
              'for this particular correction-stage setup, not a rejection of all '
              'staged training. Keep the current baseline; no default-model change. '
              'A relation-aware or hypergraph branch would test complementary evidence '
              'rather than asking another GINE with the same inputs to specialize.', '',
              'Validation: 27 model/checkpoint/diagnostic regression tests passed; '
              'all six additive checkpoints were evaluated by the normal harness '
              'loader. Frozen first-stage tensors are checked after every training '
              'run. Machine-readable metrics, histories and checkpoint paths are '
              'in `gine-two-stage/summary.json`.']
    lines += ['', 'Results are selected on this repeatedly used validation set. Improvements '
              'do not establish performance on unseen puzzles or automatic specialization '
              'by connection type. See `gine-two-stage/summary.json` for complete metrics '
              'and the saved per-guess harness histories for feedback outcomes.', '']
    Path('experiments/gine-two-stage.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
