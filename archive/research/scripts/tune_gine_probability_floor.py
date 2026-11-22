"""Bounded GINE/floor-loss tuning, selected by one harness replay per trial."""
import argparse
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
from graph_group_loss import ProbabilityFloorLoss
from run_gine_curriculum import make_schedule
from tune_gine import build_model
from nyt_connections_solver.data_loader import collate_keep_wordlists, make_state_loaders
from nyt_connections_solver.graph_features import build_graph_features


STUDY_NAME = 'gine_probability_floor_v1'
BETA = .1


def suggest(trial):
    return dict(hidden_width=trial.suggest_categorical('hidden_width', [64, 128, 256]),
                gine_layers=trial.suggest_int('gine_layers', 1, 3),
                readout_width=trial.suggest_categorical('readout_width', [64, 128, 256]),
                dropout=trial.suggest_categorical('dropout', [0., .05, .1, .2]),
                lr=trial.suggest_float('lr', 3e-5, 6e-4, log=True),
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 3e-3, log=True),
                floor_weight=trial.suggest_float('floor_weight', .01, .3, log=True))


def make_pruner(prune_after=8000):
    if prune_after < 8000:
        raise ValueError('Do not prune before 8000 updates')
    return optuna.pruners.PercentilePruner(
        75., n_startup_trials=6, n_warmup_steps=prune_after, interval_steps=1000, n_min_trials=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('weights/experiments/20261023T150230Z-text-only.pt'))
    parser.add_argument('--output', type=Path, default=Path('experiments/gine-floor-tuning'))
    parser.add_argument('--trials', type=int, default=16)
    parser.add_argument('--updates', type=int, default=15000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--prune-after', type=int, default=8000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22])
    args = parser.parse_args()
    if args.trials < 6 or args.updates < 15000 or not 8000 <= args.prune_after < args.updates:
        raise ValueError('Require >=6 trials, >=15000 updates, pruning between 8000 and the budget')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'runs').mkdir(exist_ok=True)
    baseline = {**json.loads(Path('experiments/gine-tuning/best.json').read_text())['params'], 'floor_weight': .1}
    reference = torch.load(args.reference, map_location='cpu', weights_only=True)
    assert reference['training_config']['variant'] == 'text-only'
    dates = reference['split_dates']
    manifest = dict(study_name=STUDY_NAME, reference=str(args.reference), split='val',
                    trials=args.trials, updates=args.updates, eval_every=args.eval_every,
                    batch_size=16, beta=BETA, baseline=baseline, final_seeds=args.seeds,
                    search_seed=0, sampler_seed=20261031,
                    selection='checkpoint: s4 Hit@1 with MRR tie-break; configuration: harness solve rate',
                    harness='one replay per completed trial; beam20, full_board, four mistakes, shuffle0',
                    pruning=dict(percentile=25, optuna_keep_percentile=75, startup_trials=6, min_updates=args.prune_after,
                                 interval=1000, min_comparisons=4, metric='best s4 Hit@1'),
                    feature_set='original 15 edges; Qwen frozen; active projection/norm/GINE/readout trainable',
                    search_space=dict(hidden_width=[64, 128, 256], gine_layers=[1, 3], readout_width=[64, 128, 256],
                                      dropout=[0., .05, .1, .2], lr=[3e-5, 6e-4], weight_decay=[1e-6, 3e-3],
                                      floor_weight=[.01, .3]),
                    lr_schedule=False, train_dates=len(dates['train']), val_dates=len(dates['val']))
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing tuning configuration differs')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('GPU required for the full tuning budget')
    frame = experiment.load_connections(Path(reference['dataset_path']))
    options = {**reference['feature_metadata']['options'], 'use_spelling': False, 'include_phonetics': False}
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                 cache_dir='data/feature_cache/graph', **options)
    assert edges.dim == 15 and list(edges.feature_names) == reference['edge_feature_names']
    datasets = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(dates[split])]
        assert subset.date.nunique() == len(dates[split])
        datasets[split], _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                               feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        print('PREPARED', split, flush=True)
    torch.set_num_threads(old_threads)
    embedder.to('cpu')
    torch.cuda.empty_cache()
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    study = optuna.create_study(study_name=STUDY_NAME,
        storage=f"sqlite:///{(args.output / 'study.sqlite3').resolve()}",
        direction='maximize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=20261031, n_startup_trials=6),
        pruner=make_pruner(args.prune_after))
    # Recover from interrupted sessions without treating stale RUNNING trials as active.
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    recovery_path = args.output / 'recovery.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
    for number in recovery.get('retry_trials', []):
        if not any(t.user_attrs.get('retry_of_trial') == number for t in study.trials):
            original = next(t for t in study.trials if t.number == number)
            study.enqueue_trial(original.params, user_attrs={'retry_of_trial': number})
    if not study.trials:
        study.enqueue_trial(baseline)
        study.enqueue_trial({**baseline, 'hidden_width': 64, 'dropout': .05})
        study.enqueue_trial({**baseline, 'hidden_width': 256, 'dropout': .05})

    def replay(payload, label):
        output = args.output / f'harness-{label}-val.json'
        if not output.exists():
            with output.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                                '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                                '--output', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
        harness = json.loads(output.read_text())
        assert harness['split'] == 'val' and len(harness['games']) == len(dates['val'])
        return output, harness

    def run(params, seed, label, trial=None):
        path = args.output / 'runs' / f'{label}.json'
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing and existing.get('status') == 'complete':
            assert existing['params'] == params and existing['seed'] == seed
            return existing
        started = perf_counter()
        model, config = build_model(reference, params, artifacts[4]['idx'], seed)
        loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
                                 generator=torch.Generator().manual_seed(seed+s))
                   for s, d in datasets['train'].items()}
        schedule = make_schedule('mixed', seed, args.updates)
        history = []
        def callback(step, metrics, best, row):
            history.append(row)
            progress = dict(label=label, params=params, seed=seed, status='running', history=history)
            path.write_text(json.dumps(progress, indent=2)+'\n')
            (args.output / 'progress.json').write_text(json.dumps(dict(label=label, update=step,
                best_hit1=best[0], params=params, metrics=metrics), indent=2)+'\n')
            if trial is not None:
                trial.report(best[0], step)
                trial.set_user_attr('updates_ran', step)
                trial.set_user_attr('best_hit1', best[0])
                if step >= args.prune_after and trial.should_prune():
                    raise optuna.TrialPruned(f'Bottom-quartile best Hit@1 at update {step}')
        try:
            model, summary = experiment.train(
                model, loaders, val_loaders, artifacts, device,
                lr=params['lr'], projection_lr=None, weight_decay=params['weight_decay'],
                updates=args.updates, eval_every=args.eval_every, min_updates_before_stop=args.updates,
                patience=1000, min_delta=1e-3, state_schedule=schedule,
                ranking_loss=ProbabilityFloorLoss(weight=params['floor_weight'], beta=BETA),
                evaluation_callback=callback)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            weights = Path('weights/experiments') / f'{stamp}-floor-tuning-{label}.pt'
            checkpoint = dict(model_class='StandaloneCandidateGINE', model_config=config,
                model_state_dict=model.state_dict(), feature_metadata=metadata, edge_feature_names=list(edges.feature_names),
                split_dates=dates, dataset_path=reference['dataset_path'], embedding_model_name=reference['embedding_model_name'],
                training_config=dict(variant='gine-floor-tuning', params=params, seed=seed, beta=BETA,
                    ranking_loss='probability_mass_with_floor', updates=args.updates, batch_size=16,
                    reference_checkpoint=str(args.reference), state_schedule=schedule), training_summary=summary)
            torch.save(checkpoint, weights)
            payload = dict(label=label, params=params, seed=seed, status='complete', model_config=config,
                           training=summary, checkpoint=str(weights), elapsed_seconds=perf_counter()-started,
                           trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            path.write_text(json.dumps(payload, indent=2)+'\n')
            return payload
        except optuna.TrialPruned:
            path.write_text(json.dumps(dict(label=label, params=params, seed=seed, status='pruned',
                history=history, elapsed_seconds=perf_counter()-started), indent=2)+'\n')
            raise
        finally:
            del model, loaders
            gc.collect()
            torch.cuda.empty_cache()

    def objective(trial):
        label = f'trial-{trial.number:02d}'
        payload = run(suggest(trial), 0, label, trial)
        output, harness = replay(payload, label)
        trial.set_user_attr('checkpoint', payload['checkpoint'])
        trial.set_user_attr('harness', str(output))
        trial.set_user_attr('selected_hit1', payload['training']['validation']['s4']['hit1'])
        trial.set_user_attr('solved', harness['summary']['solved'])
        print('TRIAL HARNESS', label, harness['summary']['solved'], flush=True)
        return harness['summary']['solve_rate']

    finished = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
                   and t.number not in recovery.get('invalid_pruning_trials', []) for t in study.trials)
    study.optimize(objective, n_trials=max(0, args.trials-finished), gc_after_trial=True)
    study.trials_dataframe().to_csv(args.output / 'trials.csv', index=False)
    best = dict(trial=study.best_trial.number, value=study.best_value, metric='validation harness solve rate', params=study.best_params)
    (args.output / 'best.json').write_text(json.dumps(best, indent=2)+'\n')
    print('SELECTED', json.dumps(best), flush=True)
    final = []
    for seed in args.seeds:
        for name, params in (('current-floor', baseline), ('tuned-floor', best['params'])):
            label = f'final-{name}-seed-{seed}'
            payload = run(params, seed, label)
            output, harness = replay(payload, label)
            final.append(dict(variant=name, seed=seed, result=str(args.output / 'runs' / f'{label}.json'), harness=str(output)))
            (args.output / 'final-runs.json').write_text(json.dumps(final, indent=2)+'\n')
            (args.output / 'runs.json').write_text(json.dumps(final, indent=2)+'\n')
            print('FINAL', name, seed, payload['training']['validation']['s4']['hit1'], harness['summary']['solved'], flush=True)
    print('ALL TUNING AND RERUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
