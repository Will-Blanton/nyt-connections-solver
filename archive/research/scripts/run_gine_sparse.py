"""Matched validation-only sparse GINE connectivity experiment."""
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
from run_gine_curriculum import make_schedule
from graph_group_loss import ProbabilityFloorLoss
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


def main():
    output = Path('experiments/gine-sparse')
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    params = json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    dates = reference['split_dates']
    manifest = dict(reference=str(reference_path), params=params, seeds=[10, 11, 12],
                    split='val', updates=15000, eval_every=250, batch_size=16,
                    variants=['sparse-q90', 'sparse-q99'], quantiles={'sparse-q90': .9, 'sparse-q99': .99},
                    penalty_weight=.1, beta=.1,
                    objective='probability mass + weight * mean_correct relu(log(beta/m) - log_probability)',
                    controls='existing gine-probability-floor/floor-0.1-seed-{seed}.json',
                    connectivity='union of positive channels >= training-only off-diagonal channel quantiles; no fallback',
                    feature_set='original 15, no phonetics or letter edits',
                    exposure='exactly 5000 updates per size in each arm',
                    initialization='identical per-seed tuned GINE and shared text projection',
                    selection='s4 Hit@1 with MRR checkpoint tie-break; no early stopping or pruning')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing results have a different configuration')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Full experiment requires a GPU')
    frame = experiment.load_connections(Path(reference['dataset_path']))
    options = {**reference['feature_metadata']['options'], 'use_spelling': False, 'include_phonetics': False}
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                  cache_dir='data/feature_cache/graph', **options)
    datasets = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(dates[split])]
        if subset.date.nunique() != len(dates[split]):
            raise ValueError('Split no longer matches reference')
        datasets[split], _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                               feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        print('PREPARED', split, {s: len(d) for s, d in datasets[split].items()}, flush=True)
    torch.set_num_threads(old_threads)
    embedder.to('cpu')
    torch.cuda.empty_cache()
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    old_dim = reference['model_config']['edge_dim']
    assert list(edges.feature_names[:old_dim]) == reference['edge_feature_names']
    assert edges.dim == old_dim
    # Full boards only: avoid repeating a training pair across partial states.
    triangle = torch.triu_indices(16, 16, offset=1)
    train_edges = torch.stack([d['edge_features'] for d in datasets['train'][4]])
    train_pairs = train_edges[:, triangle[0], triangle[1]].reshape(-1, edges.dim)
    thresholds = {v: torch.quantile(train_pairs, q, dim=0).tolist() for v, q in manifest['quantiles'].items()}
    connectivity = dict(thresholds=thresholds, feature_names=list(edges.feature_names), splits={})
    for split in ('train', 'val'):
        full = torch.stack([d['edge_features'] for d in datasets[split][4]])
        same_group = triangle[0] // 4 == triangle[1] // 4
        stats = {}
        for variant, values in thresholds.items():
            keep = ((full > 0) & (full >= torch.tensor(values))).any(-1)
            pairs = keep[:, triangle[0], triangle[1]]
            diagonal = torch.eye(16, dtype=torch.bool)
            degrees = (keep & ~diagonal).sum(-1)
            # Label use is diagnostic only, after fitting; never changes thresholds.
            stats[variant] = dict(pair_fraction=pairs.float().mean().item(),
                true_group_pair_fraction=pairs[:, same_group].float().mean().item(),
                cross_group_pair_fraction=pairs[:, ~same_group].float().mean().item(),
                isolated_node_fraction=(degrees == 0).float().mean().item(),
                per_color_retained=[keep[:, 4*i:4*i+4, 4*i:4*i+4][:, ~torch.eye(4, dtype=torch.bool)].float().mean().item() for i in range(4)])
        connectivity['splits'][split] = stats
    (output / 'connectivity.json').write_text(json.dumps(connectivity, indent=2)+'\n')
    print('CONNECTIVITY', json.dumps(connectivity), flush=True)
    runs = []
    for seed in manifest['seeds']:
        runs.append(dict(variant='complete', seed=seed,
                         result=f'experiments/gine-probability-floor/floor-0.1-seed-{seed}.json',
                         harness=f'experiments/gine-probability-floor/harness-floor-0.1-seed-{seed}-val.json'))
        for variant in manifest['variants']:
            criterion = ProbabilityFloorLoss(weight=manifest['penalty_weight'], beta=manifest['beta'])
            path = output / f'{variant}-seed-{seed}.json'
            payload = json.loads(path.read_text()) if path.exists() else None
            if not payload or payload.get('status') != 'complete':
                model, config = build_model(reference, params, artifacts[4]['idx'], seed)
                config['sparse_edge_thresholds'] = thresholds[variant]
                # No random draws or parameter changes relative to the complete control.
                model.sparse_edge_thresholds = torch.tensor(thresholds[variant], dtype=torch.float32)
                schedule = make_schedule('mixed', seed)
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
                    state_schedule=schedule, ranking_loss=criterion)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                weights = Path('weights/experiments') / f'{stamp}-sparse-{variant}-seed-{seed}.pt'
                checkpoint = dict(model_class='StandaloneCandidateGINE', model_config=config,
                                  model_state_dict=model.state_dict(), feature_metadata=metadata,
                                  edge_feature_names=list(edges.feature_names), split_dates=dates,
                                  dataset_path=reference['dataset_path'], embedding_model_name=reference['embedding_model_name'],
                                  training_config=dict(variant=variant, beta=manifest['beta'], penalty_weight=manifest['penalty_weight'], seed=seed, params=params,
                                                       reference_checkpoint=str(reference_path), updates=15000,
                                                       state_schedule=schedule, ranking_loss='probability_mass_with_floor'),
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
