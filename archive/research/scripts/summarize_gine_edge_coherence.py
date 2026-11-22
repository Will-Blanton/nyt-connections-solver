"""Audit and summarize matched validation-only candidate edge amplification."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import altair as alt
import pandas as pd

from summarize_gine_group_contrastive import purple_status


ROOT = Path('experiments/gine-edge-coherence')
VARIANTS = ('control', 'amplified')
COLORS = ('purple', 'blue', 'green', 'yellow')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    entries = json.loads((ROOT / 'runs.json').read_text())
    audit = json.loads((ROOT / 'probability-audit.json').read_text())
    assert len(entries) == 6 and len(audit['records']) == 726
    inventory = json.loads(Path('experiments/gine-phonetics/mechanism-inventory.json').read_text())
    mechanisms = {(r['date'], r['color']): r['mechanism'] for r in inventory['groups_inventory']}
    runs, games, curve_rows = [], {}, []
    for entry in entries:
        result = json.loads(Path(entry['result']).read_text())
        harness = json.loads(Path(entry['harness']).read_text())
        assert result['status'] == 'complete' and result['training']['updates_ran'] == 15000
        assert harness['split'] == 'val'
        checked = [r for r in audit['records']
                   if r['variant'] == entry['variant'] and r['seed'] == entry['seed']]
        assert len(checked) == 121
        assert abs(mean(1 in r['gold_ranks'] for r in checked) -
                   result['training']['validation']['s4']['hit1']) < 1e-8
        rows = harness['games']
        assert len(rows) == len({r['date'] for r in rows}) == 121
        games[entry['variant'], entry['seed']] = {r['date']: r for r in rows}
        runs.append(dict(**entry, ranking=result['training']['validation'],
            best_update=result['training']['best_update'], solves=sum(g['won'] for g in rows),
            first_correct=sum(g['first_guess_correct'] for g in rows),
            colors=dict(Counter(c for g in rows for c in g['solved_colors'])),
            purple=dict(Counter(purple_status(g) for g in rows)),
            train_seconds=result['elapsed_seconds']))
        for row in result['training']['history']:
            for metric in ('train_loss', 'val_s4_hit1', 'val_s3_hit1', 'val_s2_hit1'):
                curve_rows.append(dict(variant=entry['variant'], seed=str(entry['seed']),
                    update=row['update'], metric=metric, value=row[metric]))
    aggregate = {}
    for variant in VARIANTS:
        rows = [r for r in runs if r['variant'] == variant]
        ranks = [r for r in audit['records'] if r['variant'] == variant]
        assert len(rows) == 3 and len(ranks) == 363
        scores = {s: {k: mean(r['ranking'][s][k] for r in rows) for k in ('hit1', 'mrr', 'hit4')}
                  for s in ('s4', 's3', 's2')}
        assert abs(mean(1 in r['gold_ranks'] for r in ranks) - scores['s4']['hit1']) < 1e-8
        mechanism_rows = {}
        for rank in ranks:
            for i, color in enumerate(COLORS):
                if color != 'purple':
                    continue
                key = mechanisms[rank['date'], color]
                stats = mechanism_rows.setdefault(key, dict(ranks=[], selected=0, forced=0))
                stats['ranks'].append(rank['gold_ranks'][i])
                status = purple_status(games[variant, rank['seed']][rank['date']])
                if status != 'missed':
                    stats[status] += 1
        aggregate[variant] = dict(ranking=scores, solves=sum(r['solves'] for r in rows),
            first_correct=sum(r['first_correct'] for r in rows),
            colors={c: sum(r['colors'].get(c, 0) for r in rows) for c in COLORS},
            purple={k: sum(r['purple'].get(k, 0) for r in rows) for k in ('selected', 'forced', 'missed')},
            purple_mechanisms={k: dict(appearances=len(v['ranks']), mean_rank=mean(v['ranks']),
                top4=mean(r <= 4 for r in v['ranks']), selected=v['selected'], forced=v['forced'])
                for k, v in mechanism_rows.items()},
            group_top4={c: mean(r['gold_ranks'][i] <= 4 for r in ranks) for i, c in enumerate(COLORS)})
    paired = Counter()
    changes = []
    for seed in manifest['seeds']:
        a, b = games['control', seed], games['amplified', seed]
        assert a.keys() == b.keys()
        for date in a:
            paired['solve_gain'] += b[date]['won'] and not a[date]['won']
            paired['solve_loss'] += a[date]['won'] and not b[date]['won']
            p, q = purple_status(a[date]), purple_status(b[date])
            paired['purple_direct_gain'] += q == 'selected' and p != 'selected'
            paired['purple_direct_loss'] += p == 'selected' and q != 'selected'
            if p != q:
                changes.append(dict(seed=seed, date=date, mechanism=mechanisms[date, 'purple'],
                                    control=p, amplified=q))
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs, paired=dict(paired), purple_changes=changes)
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    chart = alt.Chart(pd.DataFrame(curve_rows)).mark_line().encode(
        x='update:Q', y=alt.Y('value:Q', scale=alt.Scale(zero=False)), color='variant:N',
        strokeDash='seed:N', tooltip=['variant', 'seed', 'update', 'metric', 'value'])
    chart.properties(width=570, height=210).facet(row='metric:N').resolve_scale(y='independent').interactive().save(
        ROOT / 'learning-curves.html')
    a, b = aggregate['control'], aggregate['amplified']
    lines = ['# Candidate-local edge coherence amplification', '',
        f"Validation solves: control {a['solves']}/363; amplified {b['solves']}/363. "
        f"Direct Purple selections: {a['purple']['selected']} versus {b['purple']['selected']}.", '',
        '## Design', '',
        'Three matched seeds (10/11/12), 121 validation puzzles repeated across seeds. '
        'Fresh controls; 15,000 updates per arm, exactly 5,000 at each state size, batch 16. '
        'No notebooks or test evaluation. These are correlated appearances of the same validation puzzles.', '',
        'For each candidate quartet and each raw edge channel, compute the mean connection '
        'from each word to the other three, then take the minimum across the four words. '
        'Multiply every edge in that channel by (1 + coherence). Ignore diagonals, preserve '
        'all channels and connectivity, and calculate once before message passing. '
        'No trainable parameters, additional providers, global encoder, or separate score channel. '
        'Raw 0–1 inputs can become 0–2 after amplification. This tests coherence-dependent '
        'amplification, not hard selection of a winning relation or shared witness identity.', '',
        'Original 15 edge channels, frozen Qwen embeddings, trainable active LayerNorm/text projection, '
        'two GINE layers of width 128 and readout 128. '
        f"AdamW LR {manifest['params']['lr']}, weight decay {manifest['params']['weight_decay']}, "
        'dropout zero. Probability-mass loss plus existing floor regularizer (weight .1, beta .1) '
        'unchanged. Evaluate every 250 updates, restore s4 Hit@1 with MRR tie-break. '
        'No early stopping, scheduler, tuning, or pruning.', '',
        'Harness: beam 20, original-board scores for all four groups, four mistakes, '
        'shuffle seed 0, normal public-feedback pruning. Normal checkpoint reconstruction '
        'retains the amplification setting, including Streamlit. Independent saved-checkpoint '
        'audit reproduces all six full-board Hit@1 values across 726 board/model evaluations. '
        'Reduced-state s3/s2 metrics use oracle remaining groups, not actual search trajectories.', '',
        '## Results', '', '| Metric | Control | Amplified |', '|---|---:|---:|']
    for s in ('s4', 's3', 's2'):
        for k in ('hit1', 'mrr', 'hit4'):
            lines.append(f'| {s} {k} | '+ ' | '.join(f"{aggregate[v]['ranking'][s][k]:.4f}" for v in VARIANTS)+' |')
    for k in ('solves', 'first_correct'):
        lines.append(f'| Harness {k} | '+ ' | '.join(f"{aggregate[v][k]}/363 ({aggregate[v][k]/363:.2%})" for v in VARIANTS)+' |')
    for c in reversed(COLORS):
        lines.append(f'| {c} recovered | '+' | '.join(str(aggregate[v]['colors'][c]) for v in VARIANTS)+' |')
    for k in ('selected', 'forced'):
        lines.append(f'| Purple {k} | '+' | '.join(str(aggregate[v]['purple'][k]) for v in VARIANTS)+' |')
    lines += ['', '## Per seed', '', '| Seed | Arm | s4 Hit@1 | Solves | Purple direct / forced | Selected update |',
              '|---|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | {r['solves']}/121 | "
                     f"{r['purple'].get('selected',0)} / {r['purple'].get('forced',0)} | {r['best_update']} |")
    lines += ['', '## Purple mechanisms', '',
        'Existing coarse mechanism labels are diagnostic only; they never enter training. '
        'Each group appears three times across seeds. Direct / forced distinguishes '
        'model-selected groups from the last four words being accepted.', '',
        '| Mechanism | Appearances | Control direct / forced | Amplified direct / forced | Control mean rank | Amplified mean rank |',
        '|---|---:|---:|---:|---:|---:|']
    for key, x in a['purple_mechanisms'].items():
        y = b['purple_mechanisms'][key]
        lines.append(f"| {key} | {x['appearances']} | {x['selected']} / {x['forced']} | "
                     f"{y['selected']} / {y['forced']} | {x['mean_rank']:.1f} | {y['mean_rank']:.1f} |")
    lines += ['', '## Paired outcomes', '',
        f"{paired['solve_gain']} new solves and {paired['solve_loss']} lost solves; "
        f"{paired['purple_direct_gain']} gained direct Purple selections and "
        f"{paired['purple_direct_loss']} lost direct selections.", '',
        '## Interpretation', '',
        'Promising matched validation signal, not an established replacement baseline. '
        'Selected s4 Hit@1 improves in all three seeds; complete solves improve in two '
        'and decline in one. Direct Purple improves in all three, while total forced '
        'Purple recoveries are unchanged. There is substantial per-puzzle turnover.', '',
        'The intended shared-phrase mechanism does not improve here: phrase-template '
        'Purple direct recovery stays at 31/135 appearances, and mean full-board rank '
        'slightly worsens. Direct gains instead occur in other semantic, sound, fragment, '
        'anagram, cultural, and polysemy buckets, with letter-edit recovery declining. '
        'These coarse labels cannot identify the causal feature channel. No sound or '
        'letter-edit providers were added in this study.', '',
        'Learning curves show later overfitting rather than an unfinished learning plateau: '
        'training loss continues down while validation ranking declines. Amplified checkpoints '
        'are selected at 6,000 / 4,250 / 2,000 updates, after training every arm for 15,000. '
        'More updates alone are not indicated by these trajectories.', '',
        'An earlier floor-0.1 experiment recorded 97/363 solves, versus 94/363 here. '
        'That is a historical comparison, not a matched control; this study supports '
        'a gain over its fresh 84/363 control but does not establish a new best solve rate. '
        'Keep the mechanism opt-in. The separate coherence-vector scoring channel is '
        'a reasonable next matched comparison, and a uniform-scale control would help '
        'determine whether the benefit comes from coherence specifically.', '',
        '## Limits and next comparison', '',
        'Channel means can differ despite their common numeric range. This experiment does not '
        'calibrate those distributions or verify that pairwise phrase matches share one actual '
        'completion. Amplification also changes message magnitude; a uniform amplification '
        'control would be needed to isolate coherence from a general change in edge scale. '
        'The separate coherence-vector scoring channel has not been tested.', '',
        '31 targeted and related regression tests passed: coherence values, an excluded outsider, '
        'diagonal exclusion, unchanged input, candidate locality, permutation equivariance, '
        'gradients, zero-scale equivalence, checkpoint reconstruction, existing sparse/GINE '
        'behavior, probability-floor loss, training probes, and transformation models.', '',
        'Reproduce with `nix develop -c uv run python scripts/run_gine_edge_coherence.py`, '
        'then `scripts/audit_gine_probability_floor.py --results experiments/gine-edge-coherence` '
        'and `scripts/summarize_gine_edge_coherence.py` in the same environment. '
        'Curves: `experiments/gine-edge-coherence/learning-curves.html`. '
        'Per-channel gold/near-miss/other means: `coherence-diagnostics.json`. '
        'Detailed paired Purple outcomes: `summary.json`.', '']
    Path('experiments/gine-edge-coherence.md').write_text('\n'.join(lines))
    print(json.dumps(summary['aggregate'], indent=2))


if __name__ == '__main__':
    main()
