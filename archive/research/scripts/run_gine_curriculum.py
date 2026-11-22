"""Matched validation-only GINE state-size curriculum experiment."""
from collections import Counter
import random
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

import run_graph_feature_experiment as experiment
from tune_gine import build_model
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


def make_schedule(variant, seed, updates=15000):
    """Equal counts at each state size; only presentation order differs."""
    if updates < 6 or updates % 6:
        raise ValueError('Budget must be positive and divisible by six')
    rng = random.Random(seed)
    schedule = []
    if variant == 'mixed':
        blocks = [[2, 3, 4] for _ in range(updates // 3)]
    elif variant == 'curriculum':
        schedule = [2] * (updates // 6) + [3] * (updates // 6)
        blocks = [[2, 3, 4, 4] for _ in range(updates // 6)]
    else:
        raise ValueError(variant)
    for block in blocks:
        rng.shuffle(block)
        schedule.extend(block)
    assert len(schedule) == updates
    assert Counter(schedule) == {2: updates // 3, 3: updates // 3, 4: updates // 3}
    return schedule


def main():
    output = Path('experiments/gine-curriculum')
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    params = json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    dates = reference['split_dates']
    manifest = dict(reference=str(reference_path), params=params, seeds=[10, 11, 12],
                    split='val', updates=15000, eval_every=250, batch_size=16,
                    variants=['mixed', 'curriculum'],
                    curriculum='2500 s2, 2500 s3, then 10000 mixed s2:s3:s4=1:1:2',
                    exposure='exactly 5000 updates per size in each arm',
                    initialization='identical per-seed tuned GINE and shared text projection',
                    selection='s4 Hit@1 with MRR checkpoint tie-break; no early stopping or pruning')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing results have a different configuration')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame = experiment.load_connections(Path(reference['dataset_path']))
    options = {**reference['feature_metadata']['options'], 'use_spelling': False, 'include_phonetics': False}
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                  cache_dir='data/feature_cache/graph', **options)
    datasets = {}
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(dates[split])]
        if subset.date.nunique() != len(dates[split]):
            raise ValueError('Split no longer matches reference')
        datasets[split], _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                               feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        print('PREPARED', split, {s: len(d) for s, d in datasets[split].items()}, flush=True)
    embedder.to('cpu')
    torch.cuda.empty_cache()
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    old_dim = reference['model_config']['edge_dim']
    assert list(edges.feature_names[:old_dim]) == reference['edge_feature_names']
    assert edges.dim == old_dim
    runs = []
    for seed in manifest['seeds']:
        for variant in manifest['variants']:
            path = output / f'{variant}-seed-{seed}.json'
            payload = json.loads(path.read_text()) if path.exists() else None
            if not payload or payload.get('status') != 'complete':
                model, config = build_model(reference, params, artifacts[4]['idx'], seed)
                schedule = make_schedule(variant, seed)
                (output / f'{variant}-seed-{seed}-schedule.json').write_text(json.dumps(schedule))
                loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
                                         generator=torch.Generator().manual_seed(seed+s))
                           for s, d in datasets['train'].items()}
                def progress(step, metrics, best, row):
                    (output / 'progress.json').write_text(json.dumps(dict(
                        seed=seed, variant=variant, update=step, best_hit1=best[0], metrics=metrics), indent=2))
                model, summary = experiment.train(
                    model, loaders, val_loaders, artifacts, device, lr=params['lr'], projection_lr=None,
                    weight_decay=params['weight_decay'], updates=15000, eval_every=250,
                    min_updates_before_stop=15000, patience=1000, min_delta=1e-3, evaluation_callback=progress,
                    state_schedule=schedule)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                weights = Path('weights/experiments') / f'{stamp}-curriculum-{variant}-seed-{seed}.pt'
                checkpoint = dict(model_class='StandaloneCandidateGINE', model_config=config,
                                  model_state_dict=model.state_dict(), feature_metadata=metadata,
                                  edge_feature_names=list(edges.feature_names), split_dates=dates,
                                  dataset_path=reference['dataset_path'], embedding_model_name=reference['embedding_model_name'],
                                  training_config=dict(variant=f'curriculum-{variant}', seed=seed, params=params,
                                                       reference_checkpoint=str(reference_path), updates=15000,
                                                       state_schedule=schedule),
                                  training_summary=summary)
                torch.save(checkpoint, weights)
                payload = dict(status='complete', seed=seed, variant=variant, checkpoint=str(weights),
                               training=summary, model_config=config)
                path.write_text(json.dumps(payload, indent=2) + '\n')
                del model, loaders
                gc.collect()
                torch.cuda.empty_cache()
            harness_path = output / f'harness-{variant}-seed-{seed}-val.json'
            if not harness_path.exists():
                with harness_path.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                                    '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                                    '--output', str(harness_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
            runs.append(dict(variant=variant, seed=seed, result=str(path), harness=str(harness_path)))
            (output / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
            print('COMPLETE', variant, seed, payload['training']['validation']['s4']['hit1'], flush=True)
    print('ALL RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
