"""Report the fixed pair/triple/quad GINE curriculum experiment."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev


ROOT = Path('experiments/gine-candidate-curriculum')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    runs, curves = [], []
    for item in json.loads((ROOT / 'runs.json').read_text()):
        result = json.loads(Path(item['result']).read_text())
        harness = json.loads(Path(item['harness']).read_text())
        assert result['training']['updates_ran'] == 15000 and harness['split'] == 'val'
        history = result['training']['history']
        runs.append(dict(variant=item['variant'], seed=item['seed'], result=item['result'],
                         harness_path=item['harness'], ranking=result['training']['validation'],
                         harness=harness['summary'], checkpoint=result['checkpoint'],
                         best_update=result['training']['best_update'],
                         recovered_colors=dict(Counter(c for g in harness['games'] for c in g['solved_colors'])),
                         initial_gold_in_beam=sum(g['history'][0]['gold_partition_in_beam'] for g in harness['games']),
                         phase_peak={name: max(row['val_s4_hit1'] for row in history
                                               if low < row['update'] <= high)
                                     for name, low, high in [('first_3000', 0, 3000),
                                                             ('next_3000', 3000, 6000),
                                                             ('last_9000', 6000, 15000)]},
                         final_s4_hit1=history[-1]['val_s4_hit1'],
                         train_seconds=sum(row['train_seconds'] for row in history),
                         validation_seconds=sum(row['validation_seconds'] for row in history)))
        for row in history:
            for state in ('s4', 's3', 's2'):
                curves.append(dict(update=row['update'], hit1=row[f'val_{state}_hit1'],
                                   variant=item['variant'], seed=str(item['seed']), size=state,
                                   candidate_size=row['candidate_size']))
    assert len(runs) == len(manifest['seeds']) * len(manifest['variants'])
    aggregate = {}
    for variant in manifest['variants']:
        rows = [r for r in runs if r['variant'] == variant]
        aggregate[variant] = dict(
            ranking={s: {m: mean(r['ranking'][s][m] for r in rows)
                         for m in ('hit1', 'mrr', 'hit4')} for s in ('s4', 's3', 's2')},
            s4_hit1_sd=stdev(r['ranking']['s4']['hit1'] for r in rows),
            harness={m: mean(r['harness'][m] for r in rows)
                     for m in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups')},
            color_top4={c: mean(r['ranking']['s4']['by_color'][c]['top4'] for r in rows)
                        for c in ('yellow', 'green', 'blue', 'purple')},
            color_recovery={c: mean(r['recovered_colors'].get(c, 0) / r['harness']['puzzles'] for r in rows)
                            for c in ('yellow', 'green', 'blue', 'purple')},
            initial_gold_in_beam=mean(r['initial_gold_in_beam'] / r['harness']['puzzles'] for r in rows),
            train_seconds=mean(r['train_seconds'] for r in rows))
    audit = json.loads((ROOT / 'near-miss-audit.json').read_text())
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs, near_misses=audit['aggregate'])
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    a, b = manifest['variants']
    lines = ['# GINE candidate-size curriculum: pairs to triples to quads', '',
             'Three matched seeds (10–12), validation only, 15,000 updates each, batch 16, '
             'evaluation every 250 updates. Same tuned standalone GINE, original 15 edge features, '
             'frozen Qwen embeddings and shared train-only text-projection initialization. '
             'Active LayerNorm, projection, GINE and readout train throughout. '
             'No phonetic additions, auxiliary loss, optimizer/LR changes, early stopping or tuning. '
             'Notebooks and test set untouched.', '',
             '## Intervention', '',
             '- Every training input is the full 16-word board. The board batch stream and '
             'initial model weights are identical across the matched arms.',
             '- Control: 15,000 updates scoring four-word candidates.',
             '- Curriculum: 3,000 updates scoring pairs, 3,000 scoring triples, then 9,000 '
             'scoring quads. The same GINE and readout weights transfer between stages; no reset.',
             '- Labels: a candidate is positive exactly when all its words belong to the same '
             'official group. All candidates are enumerated. There are 24/120 positive pairs, '
             '16/560 positive triples, and 4/1,820 positive quads.',
             '- The original probability-mass loss is retained: maximize total softmax mass '
             'over positive candidates. Correct subsets still compete collectively against '
             'incorrect subsets, and the objective can prioritize easier source groups.',
             '- GINE sum aggregation and mean/max readout are unchanged. Only the hard-coded '
             'four-word reshape was generalized. The four-word computation is numerically '
             'identical and checkpoint shapes/normal loading are unchanged.', '',
             'Candidate size is distinct from board size. Validation always scores quads on '
             '16/12/8-word canonical states (s4/s3/s2); it never evaluates pair/triple accuracy '
             'as the primary metric. Chance rates and loss baselines differ by candidate size, '
             'so training losses across stages are not directly comparable. Some officially '
             'negative pairs/triples can have other legitimate connections; labels follow only '
             'the historical puzzle solution.', '',
             'All runs complete 15,000 updates. Checkpoint selection is validation full-board '
             'Hit@1, MRR tie-break, across all 60 evaluations, including before quad training. '
             'Curriculum receives 9,000 quad updates versus 15,000 for control; the matched '
             'budget is board examples and optimizer updates, not quad exposure or FLOPs. '
             'This is a curriculum within the existing warm-start setup, not text pretraining '
             'or training all projections from scratch.', '',
             '## Results', '', '| Metric | Quad-only | Pairs → triples → quads | Change |',
             '|---|---:|---:|---:|']
    for size in ('s4', 's3', 's2'):
        for m in ('hit1', 'mrr', 'hit4'):
            x, y = aggregate[a]['ranking'][size][m], aggregate[b]['ranking'][size][m]
            cells = (f'{x:.4f} | {y:.4f} | {y-x:+.4f}' if m == 'mrr' else
                     f'{x:.2%} | {y:.2%} | {(y-x)*100:+.2f} pp')
            lines.append(f'| {size} {m} | {cells} |')
    for m in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups'):
        x, y = aggregate[a]['harness'][m], aggregate[b]['harness'][m]
        cells = (f'{x:.3f} | {y:.3f} | {y-x:+.3f}' if m == 'mean_correct_groups' else
                 f'{x:.2%} | {y:.2%} | {(y-x)*100:+.2f} pp')
        lines.append(f'| Harness {m} | {cells} |')
    lines += ['', 'Harness: same 121 validation puzzles, beam 20, full-board scoring, four allowed '
              'mistakes, shuffle seed 0, same public-feedback pruning. Saved weights were loaded '
              'through the standard harness/checkpoint interface. Smaller-state ranking is oracle '
              'evaluation, not feedback trajectories.', '',
              '| Seed | Variant | s4 Hit@1 | Solves /121 | Selected update |',
              '|---:|---|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | "
                     f"{r['harness']['solved']} | {r['best_update']} |")
    lines += ['', '| Color-specific group Top4 | Quad-only | Curriculum |', '|---|---:|---:|']
    for c in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f"| {c} | {aggregate[a]['color_top4'][c]:.2%} | {aggregate[b]['color_top4'][c]:.2%} |")
    lines += ['', '| Harness color recovery | Quad-only | Curriculum |', '|---|---:|---:|']
    for c in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f"| {c} | {aggregate[a]['color_recovery'][c]:.2%} | {aggregate[b]['color_recovery'][c]:.2%} |")
    lines += ['', 'The true full partition is initially retained in the beam on '
              f"{aggregate[a]['initial_gold_in_beam']:.2%} of control attempts versus "
              f"{aggregate[b]['initial_gold_in_beam']:.2%} with the curriculum."]
    lines += ['', '## Stage progression', '',
              'Peaks use validation quad Hit@1 on full boards in each time window. '
              'Only the curriculum changes candidate size; control uses quads throughout.', '',
              '| Seed | Arm | First 3,000 peak | Next 3,000 peak | Last 9,000 peak | Final update |',
              '|---:|---|---:|---:|---:|---:|']
    for r in runs:
        p = r['phase_peak']
        lines.append(f"| {r['seed']} | {r['variant']} | {p['first_3000']:.2%} | "
                     f"{p['next_3000']:.2%} | {p['last_9000']:.2%} | {r['final_s4_hit1']:.2%} |")
    lines += ['', '## Near-miss diagnostics', '',
              'Raw full-board quad predictions before search. Near misses are incorrect '
              'quads with three words from an official group.', '',
              '| Diagnostic | Quad-only | Curriculum |', '|---|---:|---:|']
    for label, key in [('Top1 is a 3-of-4 negative, all boards', 'wrong_top1_3over4_rate'),
                       ('Best positive beats best near miss', 'positive_beats_best_near_miss'),
                       ('Softmax mass on near misses', 'near_miss_probability_mass')]:
        x, y = (audit['aggregate'][v][key] for v in (a, b))
        lines.append(f'| {label} | {x:.2%} | {y:.2%} |')
    lines += ['', '## Interpretation', '',
              'The curriculum leaves mean full-board Hit@1 exactly unchanged at 30.03%, '
              'but improves Hit@4 (45.45→48.48%) and MRR (0.3912→0.4037). '
              'Oracle s3/s2 Hit@1 improves modestly. Harness solve rate increases '
              '18.46→21.21%, or 67→77 solves over 363 repeated-seed attempts. '
              'Paired solves are 18→21, 29→28, and 20→28; the largest gain comes '
              'from seed 12, so this is not uniform across seeds.', '',
              'There is some global-search benefit despite no Hit@1 gain: initial true '
              'partition retention rises 9.92→12.67%, and purple recovery rises '
              '20.94→25.34%. These diagnostics are consistent with better coverage '
              'beyond the top candidate, but do not establish a causal explanation. '
              'The raw 3-of-4 Top1 error rate does not improve (35.81→36.64%).', '',
              'Pair-only training has weak direct transfer to quad ranking (peak '
              '7.44–11.57% full-board Hit@1), followed by 21.49–26.45% during '
              'triple training. All selected curriculum checkpoints occur shortly after '
              'switching to quads: 6,250, 6,750 and 6,500 updates. Later quad training '
              'does not surpass those peaks. Low training loss only means the '
              'probability-mass objective is satisfied; it does not imply all correct '
              'pairs/triples were learned, since a single easy positive can dominate.', '',
              'This is a modest positive result against this full-board quad-only control, '
              'not a new baseline. For context, the earlier fragment-state experiment '
              'reached 37.47% Hit@1 and 26.17% solves with different training-state '
              'sampling. That cross-experiment comparison is not the matched causal '
              'comparison here, and no combination of the two interventions was tested. '
              'Keep the durable notebook baseline unchanged. A shorter or interleaved '
              'warm-up remains an untested possibility; this run does not establish '
              'that candidate-size curriculum in general is ineffective.', '',
              'Softmax mass depends on score scale; rank comparisons are less sensitive. '
              'All seeds reuse the same validation boards; repeated-seed attempts are not '
              'independent held-out puzzles. Repeated validation selection makes absolute '
              'scores optimistic. This tests one fixed candidate-size curriculum, without '
              'combining it with fragment training or the earlier board-size curriculum.', '',
              'Raw data: [summary.json](gine-candidate-curriculum/summary.json). '
              'Curves: [learning-curves.html](gine-candidate-curriculum/learning-curves.html). '
              'Near-miss details: [near-miss-audit.json](gine-candidate-curriculum/near-miss-audit.json).', '',
              'Reproduce: `nix develop -c uv run python scripts/run_gine_candidate_curriculum.py`, '
              'then `nix develop -c uv run python scripts/audit_fragment_predictions.py '
              '--results experiments/gine-candidate-curriculum`, then '
              '`nix develop -c uv run python scripts/summarize_gine_candidate_curriculum.py`.']
    Path('experiments/gine-candidate-curriculum.md').write_text('\n'.join(lines) + '\n')
    import altair as alt
    chart = alt.Chart(alt.Data(values=curves)).mark_line().encode(
        x='update:Q', y=alt.Y('hit1:Q', scale=alt.Scale(domain=[0, 1])),
        color='variant:N', strokeDash='seed:N',
        tooltip=['update:Q', 'hit1:Q', 'variant:N', 'seed:N', 'size:N', 'candidate_size:Q']).properties(
            width=360, height=240).facet(column='size:N').interactive()
    chart.save(str(ROOT / 'learning-curves.html'))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
