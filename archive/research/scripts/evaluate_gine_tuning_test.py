"""Evaluate fixed, validation-selected GINE reruns on the existing test split."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

import run_graph_feature_experiment as experiment
from nyt_connections_solver.data_loader import collate_keep_wordlists, make_state_loaders
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


def main():
    directory = Path('experiments/gine-tuning')
    output = directory / 'test'
    output.mkdir(exist_ok=True)
    runs = json.loads((directory / 'final-runs.json').read_text())
    results = {item['result']: json.loads(Path(item['result']).read_text()) for item in runs}
    # Choose the primary checkpoint before observing any test outcomes.
    primary = max((r for r in runs if r['model'] == 'tuned'),
                  key=lambda r: results[r['result']]['training']['validation']['s4']['hit1'])
    (output / 'selection.json').write_text(json.dumps(dict(
        primary=primary, selection='Highest validation s4 Hit@1 among fresh tuned reruns',
        aggregate='All four fixed seeds per configuration; no test-based selection',
        split='test', beam_width=20, score_mode='full_board', max_mistakes=4,
        shuffle_seed=0), indent=2) + '\n')
    reference = torch.load(results[primary['result']]['checkpoint'], map_location='cpu', weights_only=True)
    dates = reference['split_dates']
    assert not (set(dates['test']) & (set(dates['train']) | set(dates['val'])))
    frame = experiment.load_connections(Path(reference['dataset_path']))
    subset = frame[frame['date'].isin(dates['test'])]
    assert subset['date'].nunique() == len(dates['test'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options = reference['feature_metadata']['options']
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                 cache_dir='data/feature_cache/graph', **options)
    assert metadata == reference['feature_metadata']
    assert list(edges.feature_names) == reference['edge_feature_names']
    datasets, _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                    feature_builder=nodes, edge_providers=edges, cache=edges.cache)
    loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
               for s, d in datasets.items()}
    embedder.to('cpu')
    artifacts = experiment.make_artifacts()
    rows = []
    for item in runs:
        result = results[item['result']]
        checkpoint = torch.load(result['checkpoint'], map_location='cpu', weights_only=True)
        assert checkpoint['split_dates'] == dates
        assert checkpoint['feature_metadata'] == metadata
        model = StandaloneCandidateGINE(group_idx=artifacts[4]['idx'], **checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = experiment.evaluate(model, loaders, artifacts, device, by_color=True)
        del model
        torch.cuda.empty_cache()
        ranking = {f's{s}': values for s, values in metrics.items()}
        harness_path = output / f"harness-{item['model']}-seed-{item['seed']}.json"
        if not harness_path.exists():
            with harness_path.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', result['checkpoint'],
                                '--split', 'test', '--beam-width', '20', '--score-mode', 'full_board',
                                '--output', str(harness_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
        harness = json.loads(harness_path.read_text())
        assert harness['checkpoint'] == result['checkpoint'] and harness['split'] == 'test'
        assert len(harness['games']) == len(dates['test'])
        row = dict(model=item['model'], seed=item['seed'], checkpoint=result['checkpoint'],
                   ranking=ranking, harness=harness['summary'],
                   recovered_colors=dict(Counter(c for g in harness['games'] for c in g['solved_colors'])))
        rows.append(row)
        (output / f"metrics-{item['model']}-seed-{item['seed']}.json").write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    aggregate = {}
    for name in ('baseline', 'tuned'):
        selected = [r for r in rows if r['model'] == name]
        aggregate[name] = dict(
            ranking={s: {m: mean(r['ranking'][s][m] for r in selected) for m in ('hit1', 'mrr', 'hit4')}
                     for s in ('s4', 's3', 's2')},
            harness={m: mean(r['harness'][m] for r in selected) for m in
                     ('solved', 'solve_rate', 'first_guess_accuracy', 'mean_correct_groups')},
            recovered_colors={c: sum(r['recovered_colors'].get(c, 0) for r in selected)
                              for c in ('yellow', 'green', 'blue', 'purple')})
    summary = dict(test_puzzles=len(dates['test']), primary_seed=primary['seed'],
                   aggregate=aggregate, runs=rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    validation = json.loads((directory / 'summary.json').read_text())['aggregate']
    primary_row = next(r for r in rows if r['model'] == 'tuned' and r['seed'] == primary['seed'])
    lines = [
        '# Tuned GINE: fixed-checkpoint test evaluation', '',
        'Evaluated the already-selected parameters and saved validation-selected weights without retraining, '
        'changing the harness, or selecting on test outcomes. Primary checkpoint: fresh tuned seed '
        f"{primary['seed']}, chosen by validation s4 Hit@1 before test evaluation. "
        'All four previously fixed seeds are also reported for both baseline and tuned configurations.', '',
        f"Existing test split: {len(dates['test'])} puzzles, disjoint from training and validation. "
        'This test split was inspected in earlier project experiments, so it is not a pristine final holdout; '
        'newly ingested puzzles remain appropriate for a future final evaluation.', '',
        'Harness: beam 20, full-board scoring, four allowed mistakes, shuffle seed 0, existing feedback pruning. '
        's3/s2 ranking metrics use oracle reduced states, not feedback trajectories. '
        'Seed averages repeat the same puzzles and are not independent test datasets.', '',
        '| Metric | Baseline test mean | Tuned test mean | Tuned validation mean |',
        '|---|---:|---:|---:|',
    ]
    for size in ('s4', 's3', 's2'):
        for metric in ('hit1', 'mrr', 'hit4'):
            values = [aggregate[n]['ranking'][size][metric] for n in ('baseline', 'tuned')]
            values.append(validation['tuned']['ranking'][size][metric])
            formatted = [f'{v:.4f}' if metric == 'mrr' else f'{v:.2%}' for v in values]
            lines.append(f"| {size} {metric} | {' | '.join(formatted)} |")
    for metric in ('solve_rate', 'first_guess_accuracy', 'mean_correct_groups'):
        values = [aggregate[n]['harness'][metric] for n in ('baseline', 'tuned')]
        values.append(validation['tuned']['harness'][metric])
        formatted = [f'{v:.3f}' if metric == 'mean_correct_groups' else f'{v:.2%}' for v in values]
        lines.append(f"| Harness {metric} | {' | '.join(formatted)} |")
    lines += ['', '| Seed | Baseline s4 Hit@1 | Tuned s4 Hit@1 | Baseline solves | Tuned solves |',
              '|---:|---:|---:|---:|---:|']
    for seed in sorted({r['seed'] for r in rows}):
        b, t = [next(r for r in rows if r['seed'] == seed and r['model'] == name)
                for name in ('baseline', 'tuned')]
        lines.append(f"| {seed} | {b['ranking']['s4']['hit1']:.2%} | {t['ranking']['s4']['hit1']:.2%} | "
                     f"{b['harness']['solved']} | {t['harness']['solved']} |")
    lines += ['', f"Primary checkpoint (seed {primary['seed']}): s4 Hit@1 "
              f"{primary_row['ranking']['s4']['hit1']:.2%}, MRR {primary_row['ranking']['s4']['mrr']:.4f}, "
              f"{primary_row['harness']['solved']}/{len(dates['test'])} solves "
              f"({primary_row['harness']['solve_rate']:.2%}).", '',
              '| Color | Baseline recovered | Tuned recovered |', '|---|---:|---:|']
    opportunities = len(dates['test']) * 4
    for color in ('yellow', 'green', 'blue', 'purple'):
        b, t = [aggregate[n]['recovered_colors'][color] for n in ('baseline', 'tuned')]
        lines.append(f'| {color} | {b}/{opportunities} ({b/opportunities:.2%}) | '
                     f'{t}/{opportunities} ({t/opportunities:.2%}) |')
    lines += ['', 'Detailed metrics and per-guess histories: [test/summary.json](gine-tuning/test/summary.json).',
              'Reproduce: `nix develop -c uv run python scripts/evaluate_gine_tuning_test.py`. '
              'Existing completed harness replays are reused; no training runs.', '']
    Path('experiments/gine-tuning-test.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
