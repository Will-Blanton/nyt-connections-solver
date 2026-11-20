"""Validation-only frozen-logit residual GINE, with existing matched controls."""
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
from run_gine_curriculum import make_schedule
from tune_gine import build_model
from nyt_connections_solver.additive_models import AdditiveGINE
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features


OUTPUT = Path('experiments/gine-two-stage')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    params = json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    correction_params = {**params, 'hidden_width': 64, 'readout_width': 64}
    controls = {seed: json.loads(Path(f'experiments/gine-curriculum/mixed-seed-{seed}.json').read_text())
                for seed in (10, 11, 12)}
    manifest = dict(seeds=[10, 11, 12], split='val', updates=15000, eval_every=250,
                    batch_size=16, base_params=params, correction_params=correction_params,
                    correction_scale=.25, correction_initialization='text-only reference, fresh GINE, zero final output',
                    correction_seeds=[110, 111, 112], ensemble_pairs=[[10, 11], [11, 12], [12, 10]],
                    selection='s4 Hit@1, MRR tie-break; includes update 0; no early stopping',
                    base_results={str(s): f'experiments/gine-curriculum/mixed-seed-{s}.json' for s in controls},
                    features='original 15; no phonetics or letter edits',
                    loss='existing correct-choice probability mass; no auxiliary losses or schedules')
    path = OUTPUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Manifest mismatch')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('GPU required for the full experiment budget')
    frame = experiment.load_connections(Path(reference['dataset_path']))
    options = {**reference['feature_metadata']['options'], 'use_spelling': False, 'include_phonetics': False}
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                 cache_dir='data/feature_cache/graph', **options)
    assert edges.dim == 15
    datasets = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(reference['split_dates'][split])]
        assert subset.date.nunique() == len(reference['split_dates'][split])
        datasets[split], _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                               feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        print('PREPARED', split, flush=True)
    torch.set_num_threads(old_threads)
    embedder.to('cpu')
    torch.cuda.empty_cache()
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    runs = []
    for seed in manifest['seeds']:
        baseline = controls[seed]
        base = torch.load(baseline['checkpoint'], map_location='cpu', weights_only=True)
        assert base['feature_metadata'] == metadata and base['split_dates'] == reference['split_dates']
        assert base['edge_feature_names'] == list(edges.feature_names)
        runs.append(dict(variant='baseline', seed=seed,
                         result=manifest['base_results'][str(seed)],
                         harness=f'experiments/gine-curriculum/harness-mixed-seed-{seed}-val.json'))
        for variant in ('residual', 'ensemble'):
            result_path = OUTPUT / f'{variant}-seed-{seed}.json'
            payload = json.loads(result_path.read_text()) if result_path.exists() else None
            if not payload or payload.get('status') != 'complete':
                if variant == 'residual':
                    correction, correction_config = build_model(reference, correction_params, artifacts[4]['idx'], seed+100)
                    torch.nn.init.zeros_(correction.group_readout[-1].weight)
                    torch.nn.init.zeros_(correction.group_readout[-1].bias)
                    config = dict(base_config=base['model_config'], correction_config=correction_config,
                                  correction_scale=.25, base_scale=1., train_correction=True)
                else:
                    other_seed = {10: 11, 11: 12, 12: 10}[seed]
                    other = torch.load(controls[other_seed]['checkpoint'], map_location='cpu', weights_only=True)
                    assert other['feature_metadata'] == metadata and other['split_dates'] == base['split_dates']
                    config = dict(base_config=base['model_config'], correction_config=other['model_config'],
                                  base_scale=.5, correction_scale=.5, train_correction=False)
                model = AdditiveGINE(group_idx=artifacts[4]['idx'], **config)
                config.update(in_dim=nodes.dim, edge_dim=edges.dim)
                model.base.load_state_dict(base['model_state_dict'])
                model.correction.load_state_dict(correction.state_dict() if variant == 'residual' else other['model_state_dict'])
                if variant == 'residual':
                    del correction
                    initial_state = deepcopy(model.state_dict())
                    initial = experiment.evaluate(model.to(device), val_loaders, artifacts, device, by_color=True)
                    for size in (2, 3, 4):
                        assert abs(initial[size]['hit1'] - baseline['training']['validation'][f's{size}']['hit1']) < 1e-9
                    # Reset after constructing the wrapper, and pair state exposure with the control.
                    experiment.seed_everything(seed)
                    loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
                                             generator=torch.Generator().manual_seed(seed+s))
                               for s, d in datasets['train'].items()}
                    def progress(step, metrics, best, row):
                        (OUTPUT / 'progress.json').write_text(json.dumps(dict(seed=seed, update=step,
                                                                            metrics=metrics, best=best), indent=2))
                    model, summary = experiment.train(
                        model, loaders, val_loaders, artifacts, device,
                        lr=params['lr'], projection_lr=None, weight_decay=params['weight_decay'],
                        updates=15000, eval_every=250, min_updates_before_stop=15000,
                        patience=1000, min_delta=1e-3, state_schedule=make_schedule('mixed', seed),
                        evaluation_callback=progress)
                    for name, tensor in model.base.state_dict().items():
                        assert torch.equal(tensor, base['model_state_dict'][name]), f'Frozen base changed: {name}'
                    best_key = tuple(summary['validation']['s4'][k] for k in ('hit1', 'mrr'))
                    initial_key = tuple(initial[4][k] for k in ('hit1', 'mrr'))
                    summary['baseline_validation'] = {f's{s}': m for s, m in initial.items()}
                    summary['frozen_base_verified'] = True
                    if initial_key >= best_key:
                        model.load_state_dict(initial_state)
                        summary.update(best_update=0, validation=summary['baseline_validation'])
                        summary['validation_macro'] = {k: sum(initial[s][k] for s in initial)/len(initial)
                                                       for k in ('hit1', 'mrr', 'hit4', 'map', 'ndcg4')}
                    del loaders, initial_state
                else:
                    metrics = experiment.evaluate(model.to(device), val_loaders, artifacts, device, by_color=True)
                    model.cpu()
                    summary = dict(validation={f's{s}': m for s, m in metrics.items()}, updates_ran=0,
                                   best_update=None, components=[baseline['checkpoint'], controls[other_seed]['checkpoint']])
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                weights = Path('weights/experiments') / f'{stamp}-two-stage-{variant}-seed-{seed}.pt'
                checkpoint = {k: base[k] for k in ('feature_metadata', 'edge_feature_names', 'split_dates',
                                                  'dataset_path', 'embedding_model_name')}
                checkpoint.update(model_class='AdditiveGINE', model_config=config, model_state_dict=model.state_dict(),
                                  training_config=dict(variant=variant, seed=seed, manifest=manifest,
                                                       base_checkpoint=baseline['checkpoint']), training_summary=summary)
                torch.save(checkpoint, weights)
                payload = dict(status='complete', variant=variant, seed=seed, checkpoint=str(weights), training=summary)
                result_path.write_text(json.dumps(payload, indent=2)+'\n')
                del model
                gc.collect()
                torch.cuda.empty_cache()
            harness_path = OUTPUT / f'harness-{variant}-seed-{seed}-val.json'
            if not harness_path.exists():
                with harness_path.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                                    '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                                    '--output', str(harness_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
            runs.append(dict(variant=variant, seed=seed, result=str(result_path), harness=str(harness_path)))
            (OUTPUT / 'runs.json').write_text(json.dumps(runs, indent=2)+'\n')
            print('COMPLETE', variant, seed, payload['training']['validation']['s4']['hit1'], flush=True)
    print('ALL RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
