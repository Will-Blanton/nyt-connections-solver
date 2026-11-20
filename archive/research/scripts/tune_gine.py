#!/usr/bin/env python3
"""A bounded, validation-only GINE study followed by matched fresh-seed reruns."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import optuna
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

import run_graph_feature_experiment as experiment
from nyt_connections_solver.data_loader import collate_keep_wordlists, make_state_loaders
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


BASELINE = dict(hidden_width=32, gine_layers=2, readout_width=128,
                dropout=0., lr=1e-4, weight_decay=1e-6)


def suggest(trial):
    return dict(hidden_width=trial.suggest_categorical('hidden_width', [32, 64, 128]),
                gine_layers=trial.suggest_int('gine_layers', 1, 3),
                readout_width=trial.suggest_categorical('readout_width', [64, 128, 256]),
                dropout=trial.suggest_categorical('dropout', [0., .1, .2]),
                lr=trial.suggest_float('lr', 3e-5, 5e-4, log=True),
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True))


def build_model(reference, params, idx, seed):
    experiment.seed_everything(seed)
    config = deepcopy(reference['model_config'])
    config.update(edge_hidden=params['hidden_width'], gine_layers=params['gine_layers'],
                  readout_width=params['readout_width'], gine_dropout=params['dropout'],
                  edge_bias_scale=0., train_text_projection=True, train_input_norm=True)
    model = StandaloneCandidateGINE(group_idx=idx, **config)
    missing, unexpected = model.load_state_dict(reference['model_state_dict'], strict=False)
    if unexpected or any(not n.startswith(('edge_encoder.', 'edge_gate.', 'edge_correction.', 'group_readout.'))
                         for n in missing):
        raise ValueError('Unexpected mismatch loading the shared text-only initialization')
    return model, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('weights/experiments/20261023T150230Z-text-only.pt'))
    parser.add_argument('--output', type=Path, default=Path('experiments/gine-tuning'))
    parser.add_argument('--trials', type=int, default=16)
    parser.add_argument('--updates', type=int, default=15000)
    parser.add_argument('--prune-after', type=int, default=8000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 11, 12, 13])
    args = parser.parse_args()
    if args.prune_after < 8000 or args.updates < 10000 or args.prune_after >= args.updates:
        raise ValueError('Give trials >=10000 updates and defer pruning until >=8000')
    if args.trials < 6:
        raise ValueError('At least six full startup trials are required')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'runs').mkdir(exist_ok=True)
    reference = torch.load(args.reference, map_location='cpu', weights_only=True)
    if reference['training_config']['variant'] != 'text-only':
        raise ValueError('Use the shared text-only initialization checkpoint')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame = experiment.load_connections(Path(reference['dataset_path']))
    dates = reference['split_dates']
    options = {**reference['feature_metadata']['options'], 'use_spelling': False}
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                  cache_dir='data/feature_cache/graph', **options)
    datasets = {}
    for split in ('train', 'val'):
        subset = frame[frame['date'].isin(dates[split])]
        if subset['date'].nunique() != len(dates[split]):
            raise ValueError('Dataset does not reproduce the shared split')
        datasets[split], _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                               feature_builder=nodes, edge_providers=edges, cache=edges.cache)
    # Encoding is complete/cached; free model memory before sequential GPU trials.
    embedder.to('cpu')
    torch.cuda.empty_cache()
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    study = optuna.create_study(
        study_name='gine_basic_20261025', storage=f"sqlite:///{(args.output / 'study.sqlite3').resolve()}",
        direction='maximize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=20261025, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=args.prune_after,
                                          interval_steps=1000, n_min_trials=4))
    if not study.trials:
        study.enqueue_trial(BASELINE)
        study.enqueue_trial({**BASELINE, 'hidden_width': 64})
        study.enqueue_trial({**BASELINE, 'hidden_width': 64, 'dropout': .1, 'weight_decay': 1e-4})
    manifest = dict(reference=str(args.reference), split='val', updates=args.updates,
                    pruning='MedianPruner: 6 startup trials, >=8000 updates, 1000-step intervals, >=4 comparisons',
                    selection='s4 Hit@1 only', final_seeds=args.seeds, batch_size=16,
                    auxiliary_loss=False, lr_schedule=False, baseline=BASELINE,
                    train_dates=len(dates['train']), validation_dates=len(dates['val']))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run(params, seed, label, trial=None):
        started = perf_counter()
        model, config = build_model(reference, params, artifacts[4]['idx'], seed)
        loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
                                 generator=torch.Generator().manual_seed(seed + s))
                   for s, d in datasets['train'].items()}
        progress = []
        path = args.output / 'runs' / f'{label}.json'

        def callback(step, metrics, best, row):
            progress.append(row)
            path.write_text(json.dumps(dict(label=label, params=params, seed=seed, status='running',
                                             history=progress), indent=2) + '\n')
            if trial is not None:
                trial.report(best[0], step)
                trial.set_user_attr('updates_ran', step)
                trial.set_user_attr('best_hit1', best[0])
                if step >= args.prune_after and trial.should_prune():
                    raise optuna.TrialPruned(f'Pruned at {step}, best validation Hit@1={best[0]:.4f}')

        try:
            model, summary = experiment.train(
                model, loaders, val_loaders, artifacts, device, lr=params['lr'], projection_lr=None,
                weight_decay=params['weight_decay'], updates=args.updates, eval_every=args.eval_every,
                min_updates_before_stop=args.updates, patience=1000, min_delta=1e-3,
                evaluation_callback=callback)
            training_config = dict(variant='gine-basic-tuning', params=params, seed=seed,
                                   reference_checkpoint=str(args.reference), updates=args.updates,
                                   batch_size=16, metric='s4 Hit@1', auxiliary_loss=False, lr_schedule=False)
            checkpoint = dict(model_class='StandaloneCandidateGINE', model_config=config,
                              model_state_dict=model.state_dict(), embedding_model_name=reference['embedding_model_name'],
                              feature_metadata=metadata, edge_feature_names=list(edges.feature_names),
                              split_dates=dates, dataset_path=reference['dataset_path'],
                              training_config=training_config, training_summary=summary)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            weights = Path('weights/experiments') / f'{stamp}-tuning-{label}.pt'
            torch.save(checkpoint, weights)
            payload = dict(label=label, params=params, seed=seed, status='complete', training=summary,
                           checkpoint=str(weights), elapsed_seconds=perf_counter() - started,
                           trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            path.write_text(json.dumps(payload, indent=2) + '\n')
            return payload
        except optuna.TrialPruned:
            path.write_text(json.dumps(dict(label=label, params=params, seed=seed, status='pruned',
                                             history=progress, elapsed_seconds=perf_counter() - started), indent=2) + '\n')
            raise
        finally:
            del model, loaders
            gc.collect()
            torch.cuda.empty_cache()

    def objective(trial):
        payload = run(suggest(trial), 0, f'trial-{trial.number:02d}', trial)
        trial.set_user_attr('checkpoint', payload['checkpoint'])
        return payload['training']['validation']['s4']['hit1']

    finished = sum(t.state.is_finished() for t in study.trials)
    study.optimize(objective, n_trials=max(0, args.trials - finished), gc_after_trial=True)
    study.trials_dataframe().to_csv(args.output / 'trials.csv', index=False)
    best = dict(trial=study.best_trial.number, value=study.best_value, params=study.best_params)
    (args.output / 'best.json').write_text(json.dumps(best, indent=2) + '\n')
    print('SELECTED', json.dumps(best), flush=True)
    final = []
    for seed in args.seeds:
        for name, params in [('baseline', BASELINE), ('tuned', best['params'])]:
            label = f'final-{name}-seed-{seed}'
            path = args.output / 'runs' / f'{label}.json'
            payload = json.loads(path.read_text()) if path.exists() else None
            if not payload or payload.get('status') != 'complete':
                payload = run(params, seed, label)
            output = args.output / f'harness-{name}-seed-{seed}-val.json'
            if not output.exists():
                with (args.output / f'harness-{name}-seed-{seed}.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                                    '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                                    '--output', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
            final.append(dict(model=name, seed=seed, result=str(path), harness=str(output)))
            print('FINAL', name, seed, payload['training']['validation']['s4']['hit1'],
                  json.loads(output.read_text())['summary']['solved'], flush=True)
    (args.output / 'final-runs.json').write_text(json.dumps(final, indent=2) + '\n')


if __name__ == '__main__':
    main()
