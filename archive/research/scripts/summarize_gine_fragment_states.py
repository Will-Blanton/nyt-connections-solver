"""Summarize matched fragment-state training and validation harness results."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev


ROOT = Path('experiments/gine-fragment-states')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    runs = []
    curves = []
    for item in json.loads((ROOT / 'runs.json').read_text()):
        result = json.loads(Path(item['result']).read_text())
        harness = json.loads(Path(item['harness']).read_text())
        assert result['training']['updates_ran'] == manifest['updates']
        assert harness['split'] == 'val'
        history = result['training']['history']
        first_correct = [g for g in harness['games'] if g['first_guess_correct']]
        first_wrong = [g for g in harness['games'] if not g['first_guess_correct']]
        runs.append({**item, 'ranking': result['training']['validation'],
                     'best_update': result['training']['best_update'],
                     'phase_peak_s4_hit1': {
                         name: max(row['val_s4_hit1'] for row in history if low < row['update'] <= high)
                         for name, low, high in [('s2_only', 0, 2500), ('s3_only', 2500, 5000),
                                                 ('mixed', 5000, 15000)]},
                     'final_s4_hit1': history[-1]['val_s4_hit1'],
                     'recovered_colors': dict(Counter(c for game in harness['games'] for c in game['solved_colors'])),
                     'trajectory_counts': dict(
                         first_correct=len(first_correct),
                         wins_after_first_correct=sum(g['won'] for g in first_correct),
                         first_wrong=len(first_wrong),
                         wins_after_first_wrong=sum(g['won'] for g in first_wrong),
                         initial_gold_in_beam=sum(g['history'][0]['gold_partition_in_beam'] for g in harness['games'])),
                     'harness_summary': harness['summary'], 'checkpoint': result['checkpoint']})
        for row in history:
            for size in ('s2', 's3', 's4'):
                curves.append(dict(update=row['update'], hit1=row[f'val_{size}_hit1'],
                                   variant=item['variant'], seed=str(item['seed']), size=size))
    assert len(runs) == len(manifest['seeds']) * len(manifest['variants'])
    aggregate = {}
    for variant in manifest['variants']:
        rows = [r for r in runs if r['variant'] == variant]
        aggregate[variant] = dict(
            ranking={size: {m: mean(r['ranking'][size][m] for r in rows)
                            for m in ('hit1', 'mrr', 'hit4')} for size in ('s4', 's3', 's2')},
            s4_hit1_sd=stdev(r['ranking']['s4']['hit1'] for r in rows),
            harness={m: mean(r['harness_summary'][m] for r in rows)
                     for m in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups')},
            color_top4={c: mean(r['ranking']['s4']['by_color'][c]['top4'] for r in rows)
                        for c in ('yellow', 'green', 'blue', 'purple')},
            color_recovery={c: mean(r['recovered_colors'].get(c, 0) / r['harness_summary']['puzzles']
                                   for r in rows) for c in ('yellow', 'green', 'blue', 'purple')},
            trajectory_counts={k: sum(r['trajectory_counts'][k] for r in rows)
                               for k in rows[0]['trajectory_counts']})
    audit = json.loads((ROOT / 'near-miss-audit.json').read_text())
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs, near_misses=audit['aggregate'])
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    control, treatment = manifest['variants']
    lines = ['# One correct group plus partial groups', '',
             'Three matched seeds (10–12), validation only. Each arm gets 15,000 updates, batch 16, '
             'evaluation every 250. Same tuned standalone GINE, original 15 edge features and frozen '
             'Qwen input embeddings. Active LayerNorm, text projection, GINE and readout all train. '
             'Both arms warm-start the same train-only text projection; GINE/readout initialization '
             'is identical per seed. No phonetic additions, auxiliary loss, LR schedule, tuning or '
             'early stopping. Notebooks and test set untouched.', '',
             '## State construction and control', '',
             '- Both arms use the prior curriculum: 2,500 eight-word updates, 2,500 twelve-word '
             'updates, then 10,000 mixed updates in an 8/12/16-word ratio of 1/1/2. '
             'Total is exactly 5,000 updates at each size.',
             '- Fragment arm, 8 words: one complete group plus fragments of 2/1/1 words '
             'from the three other groups. For 12 words: one complete group plus 2/3/3 fragments. '
             'There is exactly one official positive among all 70 or 495 four-word candidates.',
             '- Control: two or three complete groups. Every complete group present is positive; '
             'the control is not forced to choose the designated target.',
             '- Both arms enumerate 12 fixed examples per training board per small size, '
             'balanced across four designated targets and three distractor configurations. '
             'This is 11,616 examples at each small size. Same board/index sampling, target '
             'positions, word permutations and DataLoader seed/order across the pair. '
             'Fragment choices are deterministic per index/seed, not newly resampled each epoch.',
             '- Full 16-word training boards are unchanged, with four positives, because '
             'one complete group plus three strictly incomplete groups can contain at most 13 words.',
             '- Word order is shuffled and relevance computed from original group IDs for each '
             'sample. The model sees only frozen word vectors and sliced cached edges, not '
             'group IDs, target identity or labels as input.', '',
             'The probability-mass loss is unchanged mathematically; it now accepts per-example '
             'positive masks. With one positive it is standard softmax cross-entropy. This tests '
             'removing alternative positive choices while diversifying distractors across all four '
             'source groups. Those two effects are not separately isolated here. All canonical '
             'validation states and harness puzzles remain unchanged.', '',
             'Fresh controls were retrained because the paired small-state pools and permutations '
             'differ from the previous curriculum runner. These controls, rather than older '
             'reported numbers, are the causal comparison. Checkpoint selection is validation '
             'full-board Hit@1 with MRR tie-break. Every arm completes the full training budget.', '',
             '## Results', '',
             '| Metric | Complete-group control | One correct + fragments | Change |',
             '|---|---:|---:|---:|']
    for size in ('s4', 's3', 's2'):
        for m in ('hit1', 'mrr', 'hit4'):
            a, b = aggregate[control]['ranking'][size][m], aggregate[treatment]['ranking'][size][m]
            cells = (f'{a:.4f} | {b:.4f} | {b-a:+.4f}' if m == 'mrr' else
                     f'{a:.2%} | {b:.2%} | {(b-a)*100:+.2f} pp')
            lines.append(f'| {size} {m} | {cells} |')
    for m in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups'):
        a, b = aggregate[control]['harness'][m], aggregate[treatment]['harness'][m]
        cells = (f'{a:.3f} | {b:.3f} | {b-a:+.3f}' if m == 'mean_correct_groups' else
                 f'{a:.2%} | {b:.2%} | {(b-a)*100:+.2f} pp')
        lines.append(f'| Harness {m} | {cells} |')
    lines += ['', 's2/s3 ranking evaluates oracle reduced states of 8/12 words, not synthetic '
              'fragment states. Harness: the same 121 validation puzzles, beam 20, full-board '
              'scoring, four allowed mistakes, shuffle seed 0, same public-feedback pruning. '
              'All saved weights reconstruct through the normal checkpoint loader.', '',
              '| Seed | Arm | Full-board Hit@1 | Solves /121 | Selected update |',
              '|---:|---|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | "
                     f"{r['harness_summary']['solved']} | {r['best_update']} |")
    lines += ['', '| Specific true-group Top4 | Control | Fragments |', '|---|---:|---:|']
    for c in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f"| {c} | {aggregate[control]['color_top4'][c]:.2%} | "
                     f"{aggregate[treatment]['color_top4'][c]:.2%} |")
    lines += ['', '| Harness recovery of each color | Control | Fragments |', '|---|---:|---:|']
    for c in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f"| {c} | {aggregate[control]['color_recovery'][c]:.2%} | "
                     f"{aggregate[treatment]['color_recovery'][c]:.2%} |")
    lines += ['', '## Full-board near-miss diagnostic', '',
              'Unconstrained raw model predictions on validation boards, before search/pruning. '
              'A near miss is an incorrect candidate overlapping an official true group in '
              'exactly three words. This is not game feedback or a claim that a specific triple '
              'is known to the harness.', '',
              '| Diagnostic | Control | Fragments |', '|---|---:|---:|']
    for label, key in [('Top1 is an incorrect 3-of-4 group (all boards)', 'wrong_top1_3over4_rate'),
                       ('3-of-4 among incorrect Top1 choices', 'wrong_top1_3over4_given_wrong'),
                       ('Best positive beats best 3-of-4 negative', 'positive_beats_best_near_miss'),
                       ('Total softmax mass on 3-of-4 negatives', 'near_miss_probability_mass')]:
        a, b = (audit['aggregate'][v][key] for v in (control, treatment))
        lines.append(f'| {label} | {a:.2%} | {b:.2%} |')
    lines += ['', 'Near-miss mass also depends on score scale/calibration; it is descriptive. '
              'Relative rank versus the best near miss is less sensitive to scale.', '',
              '## Training progression', '',
              '| Seed | Arm | s2-only peak s4 Hit@1 | s3-only peak | Mixed-stage peak | Final update |',
              '|---:|---|---:|---:|---:|---:|']
    for r in runs:
        p = r['phase_peak_s4_hit1']
        lines.append(f"| {r['seed']} | {r['variant']} | {p['s2_only']:.2%} | "
                     f"{p['s3_only']:.2%} | {p['mixed']:.2%} | {r['final_s4_hit1']:.2%} |")
    lines += ['', '## Interpretation', '',
              'Fragment training improves raw full-board Hit@1 in all three paired seeds '
              '(32.23→36.36%, 36.36→37.19%, 33.88→38.84%). Mean Hit@1 increases '
              '34.16→37.47%, MRR 0.4375→0.4790, and Hit@4 50.96→57.02%. '
              'Oracle reduced-state metrics improve too. The raw Top1 near-miss rate falls '
              '36.64→32.51%; specific-group Top4 retrieval improves across all four colors. '
              'These results support the sampling intervention as a useful ranking experiment.', '',
              'The harness does not gain in mean solve rate: both arms solve 95/363 repeated-seed '
              'attempts (26.17%). Paired solves are 29→28, 30→32, 36→35. First-guess accuracy '
              'improves 36.09→41.87%, but mean correct groups barely changes, blue recovery '
              'declines slightly, and purple recovery stays at 28.93%. Better initial choice '
              'and specific-group retrieval have not translated into stronger full partitions.', '',
              'First guesses are correct on 131 control attempts versus 152 fragment attempts. '
              'Of those, 64 and 75 eventually solve (48.85% and 49.34%). After an incorrect '
              'first guess, 31/232 control attempts solve versus 20/211 fragment attempts '
              '(13.36% and 9.48%). These conditional populations differ, so this does not '
              'prove that fragment training causes worse recovery. The correct full partition '
              'is initially retained in the search beam on only 57/363 versus 59/363 '
              'attempts, showing little change in this global diagnostic.', '',
              'Unlike the previous curriculum experiment, selected fragment checkpoints are '
              'all in the mixed stage (6,500–9,250 updates). Keep the ranking improvement as '
              'a promising candidate, but do not promote it as a solve-rate improvement. '
              'A next experiment should assess ranking/coverage of all true groups or partition '
              'quality, rather than assume another boost to any-correct Hit@1 will improve '
              'whole-puzzle solves. No additional experiment was selected or run from these results.', '',
              'This is one fixed sampling scheme with three seeds on a repeatedly used validation '
              'set. Selected-checkpoint scores are optimistic, and seeds do not provide new '
              'held-out puzzles. No claim of test generalization follows. These synthetic states '
              'are not legal remaining boards after removing whole correct groups; they are a '
              'training intervention to alter competition among candidate sets.', '',
              'Raw results, selected weights and full histories: [summary.json]'
              '(gine-fragment-states/summary.json). Near-miss records: '
              '[near-miss-audit.json](gine-fragment-states/near-miss-audit.json). '
              'Curves: [learning-curves.html](gine-fragment-states/learning-curves.html). '
              'Per-seed example files show exact words, source positions and positive candidate '
              'indices for all four designated colors at each small size.', '',
              'Reproduce: `nix develop -c uv run python scripts/run_gine_fragment_states.py`, '
              'then `nix develop -c uv run python scripts/audit_fragment_predictions.py`, '
              'then `nix develop -c uv run python scripts/summarize_gine_fragment_states.py`.']
    Path('experiments/gine-fragment-states.md').write_text('\n'.join(lines) + '\n')
    import altair as alt
    chart = alt.Chart(alt.Data(values=curves)).mark_line().encode(
        x='update:Q', y=alt.Y('hit1:Q', scale=alt.Scale(domain=[0, 1])),
        color='variant:N', strokeDash='seed:N',
        tooltip=['update:Q', 'hit1:Q', 'variant:N', 'seed:N', 'size:N']).properties(
            width=360, height=240).facet(column='size:N').interactive()
    chart.save(str(ROOT / 'learning-curves.html'))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
