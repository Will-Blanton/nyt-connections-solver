"""Aggregate the fixed, validation-only state-order comparison."""
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    root = Path('experiments/gine-curriculum')
    manifest = json.loads((root / 'manifest.json').read_text())
    runs = []
    for item in json.loads((root / 'runs.json').read_text()):
        result = json.loads(Path(item['result']).read_text())
        harness = json.loads(Path(item['harness']).read_text())
        assert result['training']['updates_ran'] == manifest['updates']
        assert harness['split'] == 'val'
        runs.append({**item, 'ranking': result['training']['validation'],
                     'best_update': result['training']['best_update'],
                     'phase_peak_s4_hit1': {
                         label: max(row['val_s4_hit1'] for row in result['training']['history']
                                    if low < row['update'] <= high)
                         for label, low, high in [('first_2500', 0, 2500),
                                                  ('next_2500', 2500, 5000),
                                                  ('last_10000', 5000, 15000)]},
                     'final_s4_hit1': result['training']['history'][-1]['val_s4_hit1'],
                     'harness_summary': harness['summary'], 'checkpoint': result['checkpoint']})
    assert len(runs) == len(manifest['seeds']) * len(manifest['variants'])
    aggregate = {}
    for variant in manifest['variants']:
        rows = [r for r in runs if r['variant'] == variant]
        aggregate[variant] = dict(
            ranking={s: {m: mean(r['ranking'][s][m] for r in rows)
                         for m in ('hit1', 'mrr', 'hit4')} for s in ('s4', 's3', 's2')},
            hit1_sd=stdev(r['ranking']['s4']['hit1'] for r in rows),
            harness={m: mean(r['harness_summary'][m] for r in rows)
                     for m in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups')},
            color_top4={c: mean(r['ranking']['s4']['by_color'][c]['top4'] for r in rows)
                        for c in ('yellow', 'green', 'blue', 'purple')})
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# GINE state-size curriculum', '',
             'Validation only: three matched seeds (10–12), 15,000 updates per arm, batch 16, '
             'evaluation every 250 updates. Same original 15 edge features and frozen Qwen text inputs '
             'as the durable GINE notebook baseline; no phonetic node/edge additions. '
             'Active input LayerNorm, text projection, two GINE layers, and readout all train. '
             'Tuned width/readout 128, dropout 0, fixed AdamW LR '
             f"{manifest['params']['lr']}, weight decay {manifest['params']['weight_decay']}. "
             'No auxiliary loss, LR scheduling, early stopping, or tuning. Notebooks and test '
             'puzzles were not evaluated or modified. Both arms use the same train-only text '
             'projection initialization from the baseline reference checkpoint (not a frozen '
             'projection); GINE/readout are initialized identically per seed. This tests '
             'curriculum in the existing warm-start training setup, not training everything '
             'from random initialization.', '',
             '## Controlled schedule', '',
             '- Mixed control: shuffle one update each of s2, s3, and s4 in successive blocks.',
             '- Curriculum: 2,500 updates of s2, then 2,500 of s3; final 10,000 updates '
             'shuffle blocks of s2, s3, s4, s4.',
             '- Both arms consume exactly 5,000 batches of each state size, with identical '
             'per-size DataLoader seeds/order and identical per-seed starting weights. '
             'Only update ordering changes. s2/s3/s4 mean 8/12/16 remaining words.', '',
             'Checkpoint selection is validation s4 Hit@1, MRR tie-break, across all 60 evaluations. '
             'Selected checkpoints need not be the final update. Every run still completes all '
             '15,000 updates. s2/s3 validation is oracle reduced states; harness results use actual '
             'feedback trajectories. Harness: 121 validation puzzles, beam 20, full-board scoring, '
             'four allowed mistakes, shuffle seed 0, same pruning and standard checkpoint loader.', '',
             '| Metric | Mixed control | Curriculum | Change |', '|---|---:|---:|---:|']
    for s in ('s4', 's3', 's2'):
        for m in ('hit1', 'mrr', 'hit4'):
            a, b = (aggregate[v]['ranking'][s][m] for v in manifest['variants'])
            if m == 'mrr':
                cells = f'{a:.4f} | {b:.4f} | {b-a:+.4f}'
            else:
                cells = f'{a:.2%} | {b:.2%} | {(b-a)*100:+.2f} pp'
            lines.append(f'| {s} {m} | {cells} |')
    for m in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups'):
        a, b = (aggregate[v]['harness'][m] for v in manifest['variants'])
        cells = (f'{a:.3f} | {b:.3f} | {b-a:+.3f}' if m == 'mean_correct_groups' else
                 f'{a:.2%} | {b:.2%} | {(b-a)*100:+.2f} pp')
        lines.append(f'| Harness {m} | {cells} |')
    lines += ['', '| Seed | Arm | s4 Hit@1 | Solves /121 | Selected update |', '|---:|---|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | "
                     f"{r['harness_summary']['solved']} | {r['best_update']} |")
    lines += ['', '| Color | Mixed group Top4 | Curriculum group Top4 |', '|---|---:|---:|']
    for c in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f"| {c} | {aggregate['mixed']['color_top4'][c]:.2%} | "
                     f"{aggregate['curriculum']['color_top4'][c]:.2%} |")
    lines += ['', '## Training progression', '',
              'Peak validation s4 Hit@1 within each time window. For curriculum these are '
              's2-only, s3-only, and mixed-size stages; the control is mixed throughout. '
              'The final column is the unselected final update, not the restored checkpoint.', '',
              '| Seed | Arm | First 2,500 peak | Next 2,500 peak | Last 10,000 peak | Final update |',
              '|---:|---|---:|---:|---:|---:|']
    for r in runs:
        peaks = r['phase_peak_s4_hit1']
        lines.append(f"| {r['seed']} | {r['variant']} | {peaks['first_2500']:.2%} | "
                     f"{peaks['next_2500']:.2%} | {peaks['last_10000']:.2%} | "
                     f"{r['final_s4_hit1']:.2%} |")
    lines += ['', '## Interpretation', '',
              'The curriculum gives a modest aggregate improvement: full-board Hit@1 rises '
              '33.61% to 34.99%, and harness solve rate rises 23.69% to 25.34%. Paired solves '
              'are 33→31, 28→32, and 25→29: two seeds improve, one worsens. The net six extra '
              'solves are over 363 repeated-seed game attempts, not six new unique puzzles. '
              's2/s3 ranking improves slightly too. Specific-group Top4 improves for green, '
              'blue, and purple, with a small yellow decline; purple remains very weak.', '',
              'All three selected curriculum checkpoints precede full-board training: updates '
              '1,750 and 2,000 during s2-only training, and 4,250 during s3-only training. '
              'None of the later mixed/full-board stages surpasses its earlier peak. Thus the '
              'evidence supports useful transfer from smaller states, but does not show that '
              'this build-up schedule improves performance after reaching full boards. Final '
              'update scores are lower than selected scores in both arms; lower training loss '
              'does not translate into sustained validation improvement.', '',
              'Keep this as a small positive experimental result, without promoting it as a '
              'reliable new baseline. The one-correct-group-plus-fragments proposal is a useful '
              'next isolated experiment: it changes the hard negatives and forces selection '
              'of the remaining positive rather than testing presentation order alone. '
              'No additional schedule search was performed on these results.', '',
              'Interpret the paired seeds and learning curves together. This compares a single '
              'predeclared ordering schedule, not all possible curricula. It holds total updates '
              'and per-size exposure fixed, so it tests ordering rather than extra training on '
              'small states. Seeds reuse the same validation puzzles and are not independent '
              'held-out datasets. Checkpoint selection and repeated experiments on this validation '
              'set make absolute scores optimistic.', '',
              'The proposed alternative—one complete correct group mixed with fragments of other '
              'groups—is not included. It needs example-specific relevance labels rather than '
              'the current canonical complete-partition labels. It could directly train rejection '
              'of near-miss groups while leaving exactly one positive; this would change the '
              'training-state distribution and available correct choices, not merely ordering. '
              'It is a separate follow-up, not evidence provided by this curriculum experiment.', '',
              'Raw results, full histories, schedules, checkpoints and harness records: '
              '[summary.json](gine-curriculum/summary.json). Interactive learning curves: '
              '[learning-curves.html](gine-curriculum/learning-curves.html).', '',
              'Reproduce: `nix develop -c uv run python scripts/run_gine_curriculum.py` then '
              '`nix develop -c uv run python scripts/summarize_gine_curriculum.py`.']
    Path('experiments/gine-curriculum.md').write_text('\n'.join(lines) + '\n')
    import altair as alt
    values = []
    for r in runs:
        payload = json.loads(Path(r['result']).read_text())
        for row in payload['training']['history']:
            for s in ('s2', 's3', 's4'):
                values.append(dict(update=row['update'], hit1=row[f'val_{s}_hit1'],
                                   variant=r['variant'], seed=str(r['seed']), size=s))
    chart = alt.Chart(alt.Data(values=values)).mark_line().encode(
        x='update:Q', y=alt.Y('hit1:Q', scale=alt.Scale(domain=[0, 1])),
        color='variant:N', strokeDash='seed:N',
        tooltip=['update:Q', 'hit1:Q', 'variant:N', 'seed:N', 'size:N']).properties(
            width=360, height=240).facet(column='size:N').interactive()
    chart.save(str(root / 'learning-curves.html'))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
