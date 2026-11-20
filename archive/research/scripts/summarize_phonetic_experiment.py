"""Aggregate matched pronunciation experiments without selecting on test data."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    output = Path('experiments/gine-phonetics')
    manifest = json.loads((output / 'manifest.json').read_text())
    runs = []
    for item in json.loads((output / 'runs.json').read_text()):
        result = json.loads(Path(item['result']).read_text())
        harness = json.loads(Path(item['harness']).read_text())
        runs.append(dict(variant=item['variant'], seed=item['seed'], checkpoint=result['checkpoint'],
                         ranking=result['training']['validation'], best_update=result['training']['best_update'],
                         updates=result['training']['updates_ran'], harness=harness['summary'],
                         recovered_colors=dict(Counter(c for g in harness['games'] for c in g['solved_colors']))))
    assert len(runs) == len(manifest['seeds']) * len(manifest['variants'])
    assert all(r['updates'] == 15000 for r in runs)
    aggregate = {}
    for variant in manifest['variants']:
        rows = [r for r in runs if r['variant'] == variant]
        aggregate[variant] = dict(
            ranking={size: {m: mean(r['ranking'][size][m] for r in rows) for m in ('hit1', 'mrr', 'hit4')}
                     for size in ('s4', 's3', 's2')},
            s4_hit1_sd=stdev(r['ranking']['s4']['hit1'] for r in rows),
            harness={m: mean(r['harness'][m] for r in rows) for m in
                     ('solved', 'solve_rate', 'first_guess_accuracy', 'mean_correct_groups')},
            color_top4={c: mean(r['ranking']['s4']['by_color'][c]['top4'] for r in rows)
                        for c in ('yellow', 'green', 'blue', 'purple')},
            recovered_colors={c: sum(r['recovered_colors'].get(c, 0) for r in rows)
                              for c in ('yellow', 'green', 'blue', 'purple')})
    summary = dict(manifest=manifest, aggregate=aggregate, runs=runs)
    group_ranks = json.loads((output / 'group-ranks.json').read_text()) if (output / 'group-ranks.json').exists() else None
    coverage = json.loads((output / 'feature-coverage.json').read_text()) if (output / 'feature-coverage.json').exists() else None
    if group_ranks:
        summary['by_mechanism'] = group_ranks['aggregate']
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Pronunciation edges on tuned standalone GINE', '',
             'Three matched seeds (10–12), validation only, 15,000 updates each, batch 16. '
             'Same frozen Qwen inputs, fully trainable active LayerNorm/projection/GINE/readout, '
             'tuned hidden width 128, two GINE layers, readout 128, dropout 0, LR '
             f"{manifest['params']['lr']}, AdamW weight decay {manifest['params']['weight_decay']}. "
             'No auxiliary loss, LR schedule, early stopping, or Optuna in this experiment. '
             'Checkpoint selection uses validation s4 Hit@1, with MRR only breaking checkpoint ties. '
             'Notebooks and test puzzles were not used.', '',
             '## Added features', '',
             '1. Exact homophones from any CMUdict pronunciation, ignoring lexical stress.',
             '2. Maximum normalized phoneme-token edit similarity over pronunciation alternatives.',
             '3. Shared rhyme from the last primary/secondary stressed vowel onward.',
             '4. Maximum frozen semantic similarity between dictionary-attested homophone alternatives '
             'on both sides. This targets groups such as `base/liar/loot/symbol` → '
             '`bass/lyre/lute/cymbal`, rather than requiring the original four words to sound alike.',
             '5. Maximum phonetic similarity of valid proper WordNet prefixes.',
             '6. The corresponding proper-suffix channel.', '',
             'Substring extraction matches the existing semantic channels: alphabetic normalization, '
             'minimum three letters, proper prefix/suffix only, and short-fragment confidence penalties. '
             'This does not compare arbitrary internal substrings or all phoneme subsequences. '
             'Pronunciations of lexical fragments are looked up independently and may differ from '
             'their sounds inside the full word. Unknown pronunciations contribute zero and are '
             'reported as unknown in the audit, not known negative evidence. Multiple pronunciations '
             'are retained; multiword composition is capped deterministically at 32 alternatives.', '',
             'NLTK CMUdict resource and lexical/embedding identities are fingerprinted in feature metadata. '
             'Sound features are opt-in (`include_phonetics=True`), preserving existing 15-channel '
             'checkpoint identities and behavior. No new dependency or corpus download was required. '
             '[CMUdict documentation](https://www.nltk.org/api/nltk.corpus.reader.cmudict.html).', '',
             '## Matched comparison', '',
             'All arms have identical 21-channel architecture and shared initial weights. Added edge-input '
             'columns start at zero; the control masks all six additions, whole-word masks only the '
             'two substring channels, and the full arm enables all six. Every old parameter is copied '
             'from the same per-seed original-width initialization. Thus random initialization and '
             'extra parameter count do not explain differences between arms. The control is retrained, '
             'not substituted with an older reported score.', '',
             '| Metric | Control | Whole-word sound | Whole-word + substrings |', '|---|---:|---:|---:|']
    for size in ('s4', 's3', 's2'):
        for metric in ('hit1', 'mrr', 'hit4'):
            values = [aggregate[v]['ranking'][size][metric] for v in manifest['variants']]
            formatted = [f'{value:.4f}' if metric == 'mrr' else f'{value:.2%}' for value in values]
            lines.append(f"| {size} {metric} | {' | '.join(formatted)} |")
    for metric in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups'):
        values = [aggregate[v]['harness'][metric] for v in manifest['variants']]
        formatted = [f'{value:.3f}' if metric == 'mean_correct_groups' else f'{value:.2%}' for value in values]
        lines.append(f"| Harness {metric} | {' | '.join(formatted)} |")
    lines += ['', 'Harness: same 121 validation puzzles, beam width 20, full-board scoring, four allowed '
              'mistakes, shuffle seed 0, existing public-feedback pruning. Saved checkpoints were '
              'reconstructed by the normal loader for every harness run. s3/s2 ranking uses oracle '
              'reduced states, not feedback trajectories.', '',
              '| Seed | Variant | s4 Hit@1 | s4 MRR | Solves /121 | Selected update |',
              '|---:|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | "
                     f"{r['ranking']['s4']['mrr']:.4f} | {r['harness']['solved']} | {r['best_update']} |")
    lines += ['', '| Color | Control s4 group Top4 | Whole-word | With substrings |', '|---|---:|---:|---:|']
    for color in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f"| {color} | " + ' | '.join(f"{aggregate[v]['color_top4'][color]:.2%}" for v in manifest['variants']) + ' |')
    lines += ['', '| Color recovered in harness | Control | Whole-word | With substrings |', '|---|---:|---:|---:|']
    for color in ('yellow', 'green', 'blue', 'purple'):
        lines.append(f"| {color} | " + ' | '.join(
            f"{aggregate[v]['recovered_colors'][color]}/{121*len(manifest['seeds'])}" for v in manifest['variants']) + ' |')
    if group_ranks:
        lines += ['', '## Specific group ranks by audited mechanism', '',
                  'These measure the rank of the specific true group on the original board, not whether '
                  'some other correct group is chosen. Mechanisms are inferred from labels for analysis '
                  'only; definitions and every assignment are in the inventory JSON.', '',
                  '| Mechanism | Groups per seed | Control group Top4 | Whole-word | With substrings |',
                  '|---|---:|---:|---:|---:|']
        for mechanism, row in group_ranks['aggregate']['control'].items():
            values = [group_ranks['aggregate'][v][mechanism]['top4'] for v in manifest['variants']]
            lines.append(f"| {mechanism} | {row['appearances']//len(manifest['seeds'])} | " +
                         ' | '.join(f'{value:.2%}' for value in values) + ' |')
        lines += ['', '### Sound-labelled examples', '',
                  '| Date | Category | Control mean rank | Whole-word mean rank | With substrings mean rank |',
                  '|---|---|---:|---:|---:|']
        examples = sorted({(r['date'], r['category']) for r in group_ranks['records'] if r['mechanism'] == 'sound'})
        for date, category in examples:
            values = [mean(r['rank'] for r in group_ranks['records'] if r['date'] == date and
                           r['category'] == category and r['variant'] == v) for v in manifest['variants']]
            lines.append(f"| {date} | {category} | " + ' | '.join(f'{value:.1f}' for value in values) + ' |')
    if coverage:
        lines += ['', '## Static feature coverage on sound-labelled groups', '',
                  '16 sound-labelled groups across colors (including homophones, rhymes, heteronyms '
                  'and silent-letter groups). Mean edge evidence is compared with all 1,816 incorrect '
                  'quads on the same board; ties get half credit. This is a univariate diagnostic, '
                  'not a model performance estimate or a proof of explanatory coverage.', '',
                  '| Channel | Any pair | All six pairs | Mean true-vs-incorrect percentile |',
                  '|---|---:|---:|---:|']
        for name, row in coverage['summary']['sound']['channels'].items():
            lines.append(f"| {name} | {row['any_pair']:.2%} | {row['all_pairs']:.2%} | {row['incorrect_percentile']:.3f} |")
    lines += ['', '## Interpretation', '',
              'Whole-word features give a small mean harness improvement (23.69% → 24.79%), '
              'with solve counts 30→30, 25→26, 31→34. s4 Hit@1 is essentially unchanged '
              '(34.44% → 34.16%); MRR and Hit@4 improve slightly. This is not sufficient '
              'evidence to promote the feature set as a new baseline after only three seeds on '
              'the reused validation puzzles.', '',
              'Substring sound features do not add a consistent gain: mean solve rate falls to '
              '23.14%, with paired solve counts 30→31, 25→29, 31→24. They remain opt-in '
              'experimental features rather than default inputs.', '',
              'The strongest specific success is `gnu/knew/new/nu`: control ranks 981, 520, 912 '
              'become 1, 6, 1 with whole-word features (2, 7, 2 including substrings). '
              'Most homophones-of-category groups remain far below the top candidates. '
              'For example, instrument homophones stay around mean rank 1697 despite stronger '
              'edge evidence. CMUdict gives `liar` the alternative `leier` rather than `lyre`; '
              'it gives `keto` alternatives such as `cato` rather than the intended city `Quito`. '
              'Dictionary support and correct intended readings must be audited separately.', '',
              'There is also a learning/use-of-evidence gap: the music-genre group '
              '`indy/mettle/seoul/wrap` has homophone-alternative mean evidence at the 99.2nd '
              'percentile versus incorrect quads, yet the whole-word model ranks it about 1265 '
              'on average. This does not prove the edge is a complete explanation, but it cautions '
              'against assuming every failure needs another feature. The any-correct-group objective '
              'permits prioritizing easier groups; whether that contributes to neglecting these '
              'signals is a hypothesis, not isolated by this experiment.', '',
              'No baseline notebook, harness policy, or existing checkpoint feature definition was '
              'changed. For a next small feature experiment, the audit favors semantic comparisons '
              'after one-letter transformations, then alternative-anagram semantics. These still '
              'need matched controls and validation evaluation.', '',
              'Pronunciation resources and word-pair computations are shared/cached. New live boards '
              'may require additional frozen embeddings of homophone alternatives; no cold-start '
              'latency improvement is claimed from cached harness timings.']
    lines += ['', 'Repeated seeds play the same puzzles; these are not independent held-out datasets. '
              'Color Top4 measures the rank of that specific group, while Hit@1 accepts any correct group. '
              'Homophone-alternative semantic similarity is a noisy maximum over dictionary senses; '
              'a high edge does not establish one shared four-word explanation.', '',
              'Coverage and next-feature hypotheses: [mechanism audit](connection-mechanism-gaps.md). '
              'Raw metrics/checkpoints/histories: [summary.json](gine-phonetics/summary.json).', '',
              'Reproduce: `nix develop -c uv run python scripts/run_phonetic_experiment.py`, then '
              '`nix develop -c uv run python scripts/summarize_phonetic_experiment.py`.']
    Path('experiments/gine-phonetics.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
