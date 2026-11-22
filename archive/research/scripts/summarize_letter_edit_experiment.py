"""Aggregate the fixed, validation-only one-letter feature comparison."""
import json
from pathlib import Path
from statistics import mean, stdev
from collections import Counter

from audit_letter_edit_coverage import EXAMPLES


ROOT = Path('experiments/gine-letter-edits')


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    coverage = json.loads((ROOT / 'feature-coverage.json').read_text())
    group_ranks = json.loads((ROOT / 'group-ranks.json').read_text())
    runs, curves = [], []
    for item in json.loads((ROOT / 'runs.json').read_text()):
        result = json.loads(Path(item['result']).read_text())
        harness = json.loads(Path(item['harness']).read_text())
        assert result['training']['updates_ran'] == 15000 and harness['split'] == 'val'
        targeted_games = [g for g in harness['games'] if g['date'] in EXAMPLES]
        runs.append(dict(variant=item['variant'], seed=item['seed'], result=item['result'],
                         harness_path=item['harness'], checkpoint=result['checkpoint'],
                         ranking=result['training']['validation'], harness=harness['summary'],
                         best_update=result['training']['best_update'],
                         target_harness=dict(attempts=len(targeted_games),
                                             solved=sum(g['won'] for g in targeted_games),
                                             recovered=sum('purple' in g['solved_colors'] for g in targeted_games)),
                         recovered_colors=dict(Counter(c for g in harness['games'] for c in g['solved_colors']))))
        for row in result['training']['history']:
            for size in ('s4', 's3', 's2'):
                curves.append(dict(update=row['update'], hit1=row[f'val_{size}_hit1'],
                                   variant=item['variant'], seed=str(item['seed']), size=size))
    assert len(runs) == len(manifest['variants']) * len(manifest['seeds'])
    aggregate = {}
    for variant in manifest['variants']:
        rows = [r for r in runs if r['variant'] == variant]
        ranks = [r for r in group_ranks['records'] if r['variant'] == variant and
                 r['color'] == 'purple' and r['date'] in EXAMPLES]
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
            directly_supported=dict(appearances=len(ranks), top1=mean(r['rank'] == 1 for r in ranks),
                                    top4=mean(r['rank'] <= 4 for r in ranks),
                                    mrr=mean(1 / r['rank'] for r in ranks)),
            targeted_by_operation={op: dict(
                appearances=sum(EXAMPLES[r['date']][0] == op for r in ranks),
                top4=mean(r['rank'] <= 4 for r in ranks if EXAMPLES[r['date']][0] == op),
                mrr=mean(1/r['rank'] for r in ranks if EXAMPLES[r['date']][0] == op))
                for op in ('substitution', 'deletion')},
            target_harness={key: sum(r['target_harness'][key] for r in rows)
                            for key in ('attempts', 'solved', 'recovered')})
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs,
                   by_mechanism=group_ranks['aggregate'], feature_coverage=coverage['summary'],
                   latency=coverage['latency'], preparation=json.loads((ROOT / 'feature-preparation.json').read_text()))
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    variants = manifest['variants']
    lines = ['# One-letter substitution and deletion semantics on GINE', '',
             'Three matched seeds (10–12), 15,000 updates per arm, batch 16, validation only. '
             'Same tuned standalone GINE, original 15 channels, frozen Qwen text embeddings, '
             'fully trainable active LayerNorm/projection/GINE/readout, probability-mass loss '
             'and balanced shuffled s2/s3/s4 board-size updates as the baseline. No new curriculum, '
             'phonetic features, auxiliary loss, LR schedule, early stopping or tuning. '
             'Notebooks and test set untouched.', '',
             '## Added evidence', '',
             'For each single ASCII alphabetic input, enumerate exact one-letter substitutions '
             'and deletions whose outputs are WordNet English vocabulary words. The source need '
             'not be a valid word. Target readings must have at least three letters; source length '
             'is capped at 32. No insertion, multi-letter edit, multilingual translation, '
             'entity lookup or category labels are used.', '',
             'Each operation has first-letter, last-letter, aligned internal position from the '
             'start, aligned internal position from the end, and anywhere channels. Internal '
             'aligned maxima only compare edits at matching offsets. Anywhere permits different '
             'offsets, needed for taxonomic ranks such as `border/classy/foamily/genius`.', '',
             'Each alignment supplies two channels: maximum nonnegative cosine between frozen '
             'embeddings of distinct transformed words, and reading specificity '
             '`1/(1+ln(n))`, where n counts distinct aligned ordered reading pairs. Specificity '
             'is a lexical ambiguity descriptor, not a calibrated probability. It is separate '
             'from the semantic maximum, so the model learns how to use it. Identical transformed '
             'targets are excluded to avoid trivial shared-spelling shortcuts; `wind/wine/wing/wink` '
             '→ `win` is intentionally not new semantic evidence. Empty/unsupported pairs are zero.', '',
             'Twenty optional channels are appended to the original 15. Every arm has identical '
             '35-channel parameter shapes and identical shared initial weights; added input '
             'columns start at zero. Control masks all additions, substitution/deletion masks '
             'the other family, and both enables all. New columns can learn only in enabled '
             'arms. Shared lexical lookups, frozen vectors, and per-provider board matrices '
             'are cached. Resources, parameters, channel order and model revision are fingerprinted '
             'for reconstruction by the normal checkpoint loader and Streamlit.', '',
             'All runs complete the full budget, evaluating every 250 updates. Select by '
             'validation full-board Hit@1 with MRR tie-break. These are fresh matched controls, '
             'not older reported scores. Seeds reuse the same 121 validation puzzles; repeated '
             'validation selection makes absolute scores optimistic.', '',
             '## Aggregate results', '', '| Metric | Control | Substitution | Deletion | Both |',
             '|---|---:|---:|---:|---:|']
    for size in ('s4', 's3', 's2'):
        for metric in ('hit1', 'mrr', 'hit4'):
            values = [aggregate[v]['ranking'][size][metric] for v in variants]
            cells = [f'{x:.4f}' if metric == 'mrr' else f'{x:.2%}' for x in values]
            lines.append(f"| {size} {metric} | " + ' | '.join(cells) + ' |')
    for metric in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups'):
        cells = [f"{aggregate[v]['harness'][metric]:.3f}" if metric == 'mean_correct_groups' else
                 f"{aggregate[v]['harness'][metric]:.2%}" for v in variants]
        lines.append(f"| Harness {metric} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Harness: the same 121 validation games, beam 20, original full-board scoring, '
              'four allowed mistakes, shuffle seed 0, standard feedback pruning. All checkpoints '
              'were reconstructed through the shared loader. s2/s3 ranking uses oracle reduced '
              'states, not feedback trajectories.', '',
              '| Seed | Variant | s4 Hit@1 | Solves /121 | Selected update |',
              '|---:|---|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | "
                     f"{r['harness']['solved']} | {r['best_update']} |")
    lines += ['', '| Color-specific group Top4 | Control | Substitution | Deletion | Both |',
              '|---|---:|---:|---:|---:|']
    for color in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f'| {color} | ' + ' | '.join(f"{aggregate[v]['color_top4'][color]:.2%}" for v in variants) + ' |')
    lines += ['', '| Harness color recovery | Control | Substitution | Deletion | Both |',
              '|---|---:|---:|---:|---:|']
    for color in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f'| {color} | ' + ' | '.join(f"{aggregate[v]['color_recovery'][color]:.2%}" for v in variants) + ' |')
    lines += ['', '## Specific letter-edit groups', '',
              'The existing coarse inventory identifies 12 letter-edit groups among 484 '
              'validation groups. Eight Purple groups directly fit these operations: two '
              'substitution groups and six deletion groups. Other labels require insertion '
              'or multi-letter edits, or reduce all entries to one identical target. The '
              'broader letter-edit bucket is not the same as directly supported coverage.', '',
              '| Metric on the eight directly supported groups | Control | Substitution | Deletion | Both |',
              '|---|---:|---:|---:|---:|']
    for metric in ('top1', 'top4', 'mrr'):
        cells = [f"{aggregate[v]['directly_supported'][metric]:.4f}" if metric == 'mrr' else
                 f"{aggregate[v]['directly_supported'][metric]:.2%}" for v in variants]
        lines.append(f'| Specific-group {metric} | ' + ' | '.join(cells) + ' |')
    for op in ('substitution', 'deletion'):
        for metric in ('top4', 'mrr'):
            cells = [f"{aggregate[v]['targeted_by_operation'][op][metric]:.4f}" if metric == 'mrr' else
                     f"{aggregate[v]['targeted_by_operation'][op][metric]:.2%}" for v in variants]
            lines.append(f'| {op} target-group {metric} | ' + ' | '.join(cells) + ' |')
    for label, key in [('Full solves on the eight target puzzles', 'solved'),
                       ('Purple target groups recovered in harness', 'recovered')]:
        cells = [f"{aggregate[v]['target_harness'][key]}/{aggregate[v]['target_harness']['attempts']}" for v in variants]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'These are the same eight groups repeated across three seeds (24 appearances), '
              'not 24 independent categories. Ranks refer to that specific group among all '
              '1,820 quads on its original board, rather than whether another correct group '
              'reaches the top. Labels/intended readings are used for analysis only.', '',
              '| Date | Category | Intended readings retrieved | Control mean rank | Substitution | Deletion | Both |',
              '|---|---|---:|---:|---:|---:|---:|']
    for e in sorted(coverage['examples'], key=lambda x: x['date']):
        values = [mean(r['rank'] for r in group_ranks['records'] if r['variant'] == v and
                       r['date'] == e['date'] and r['color'] == 'purple') for v in variants]
        lines.append(f"| {e['date']} | {e['category']} | {sum(e['intended_retrieved'])}/4 | " +
                     ' | '.join(f'{x:.1f}' for x in values) + ' |')
    lines += ['', '## Static evidence and limitations', '',
              'For each true group, mean edge evidence is ranked against the 1,816 incorrect '
              'quads on the same board, with half credit for ties. These are single-channel '
              'diagnostics, not model accuracy or proof of a consistent four-word explanation. '
              'Each pair can maximize over different lexical readings. The JSON retains '
              'maximizing witness words for the eight examples.', '',
              '| Semantic channel | Eight supported | Two substitution targets | Six deletion targets | All Purple |',
              '|---|---:|---:|---:|---:|']
    names = ['prefix_semantic_similarity', 'suffix_semantic_similarity'] + [
        name for name in coverage['summary']['directly_supported']['channels']
        if name.startswith(('substitution_', 'deletion_')) and name.endswith('_semantic')]
    for name in names:
        lines.append(f"| {name} | {coverage['summary']['directly_supported']['channels'][name]['incorrect_percentile']:.3f} | "
                     f"{coverage['summary']['substitution_targets']['channels'][name]['incorrect_percentile']:.3f} | "
                     f"{coverage['summary']['deletion_targets']['channels'][name]['incorrect_percentile']:.3f} | "
                     f"{coverage['summary']['all_purple']['channels'][name]['incorrect_percentile']:.3f} |")
    lines += ['', 'First/last deletion partly overlaps the existing valid prefix/suffix '
              'semantic channels. Substitution and internal deletion add genuinely different '
              'readings. Improvements cannot be assumed just from finding an intended target '
              'in the dictionary; alternative readings and high maxima can introduce noise.', '',
              '## Interpretation', '',
              'Substitution alone leaves overall solves almost unchanged (88→89/363), '
              'while deletion rises to 94/363 and both to 93/363. Deletion has the best '
              'mean solve rate, 25.90% versus 24.24%, with paired solves 29→33, 32→31, '
              '27→30. All additions lower mean full-board Hit@1: control 34.44%, '
              'substitution 33.33%, deletion 33.06%, both 32.23%. The small solve gains '
              'do not justify promoting a new baseline from this reused validation set.', '',
              'The stronger finding is a gap between supplied evidence and selected-model '
              'retrieval. All 32 intended readings are retrieved. The appropriate positional '
              'channel ranks fruit evidence at the 99.72nd percentile versus incorrect quads; '
              'numbers and astronomical terms reach 100%. Nevertheless, none of the eight '
              'specific groups reaches Top4 in any selected model/seed. Substitution makes '
              'the fruit group mean rank worse (196→651) and numbers worse (467→905). '
              'Both improves wood-related ranks (839→341) but they remain far from the top.', '',
              'On the eight target puzzles, the harness fully solves 3/24 control attempts, '
              '2/24 substitution, 3/24 deletion and 4/24 both. It recovers the Purple target '
              'on 3/24, 2/24, 3/24 and 5/24 attempts respectively. Feedback and forced final '
              'groups can recover targets absent from initial Top4; these small counts still '
              'do not establish reliable success on the intended mechanism.', '',
              'The features therefore add demonstrably selective evidence for these cases, '
              'but the present training/selection setup does not convert it into reliable '
              'target-group decisions. The any-correct-group objective can ignore difficult '
              'groups, making all-group or partition training a useful next controlled test. '
              'This experiment does not isolate the cause: noisy lexical maxima, representation '
              'limitations and rare training examples remain possible contributors.', '',
              'Maxima need scrutiny too. In the unrestricted fruit channel, only one of six '
              'maximizing reading pairs matches the intended fruits; alternatives include '
              '`amble/rote`. Positional alignment restores much more selective fruit evidence. '
              'Thus broad maxima are not equivalent to one consistent explanation, even '
              'when the intended readings exist. Preserve the opt-in implementation and '
              'keep the durable notebook baseline unchanged.', '',
              '## Feature cost', '']
    prep = summary['preparation']
    lines.append(f"Train/validation union: {prep['input_words']} inputs, {prep['alternative_words']} distinct "
                 f"readings, median {prep['median_alternatives']} and maximum {prep['max_alternatives']} "
                 f"readings per input. Incremental prefetch took {prep['prefetch_seconds']:.1f}s and graph "
                 f"preparation {prep['dataset_preparation_seconds']:.1f}s with the existing cache.")
    latency = coverage['latency']
    for name in ('warm_resource_uncached_family', 'all_families_warm_matrix_cache'):
        t = latency[name]
        lines.append(f"\n{name}: median {t['median_ms']:.2f} ms, p95 {t['p95_ms']:.2f} ms per full board "
                     f"on {latency['cpu_threads']} CPU threads.")
    lines += ['', 'These timings exclude Qwen inference for unseen alternatives and initial model/'
              'corpus loading. Cached matrices can be sliced for all subgraphs. Cold live boards '
              'can still require embedding new lexical readings; no fully cold latency claim '
              'is made.', '',
              'Raw results and checkpoints: [summary.json](gine-letter-edits/summary.json). '
              'Per-mechanism ranks: [group-ranks.json](gine-letter-edits/group-ranks.json). '
              'Witnesses/coverage: [feature-coverage.json](gine-letter-edits/feature-coverage.json). '
              'Curves: [learning-curves.html](gine-letter-edits/learning-curves.html).', '',
              'Reproduce: `nix develop -c uv run python scripts/run_letter_edit_experiment.py`, '
              'then `nix develop -c uv run python scripts/audit_letter_edit_coverage.py`, '
              '`nix develop -c uv run python scripts/audit_phonetic_group_ranks.py --results '
              'experiments/gine-letter-edits`, and '
              '`nix develop -c uv run python scripts/summarize_letter_edit_experiment.py`.']
    Path('experiments/gine-letter-edits.md').write_text('\n'.join(lines) + '\n')
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
