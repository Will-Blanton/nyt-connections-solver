"""Validation-only optimization study for the full learned-transformation edge model.

Run sequentially with nix develop -c uv run python -u scripts/tune_gine_transformations.py.
Architecture and loss stay fixed; only LR, weight decay and GINE dropout vary.
"""
import argparse
from copy import deepcopy
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import optuna
import torch
from torch.utils.data import DataLoader, Subset

import run_graph_feature_experiment as experiment
from run_gine_curriculum import make_schedule
from run_gine_transformations import build
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def queue_dropout_coverage(root):
    """After the first adaptive comparisons, ensure all proposed dropouts are tried."""
    study = optuna.load_study(study_name='transformation_optimizer_v1',
        storage=f'sqlite:///{(root / "study.sqlite3").resolve()}')
    if any(t.user_attrs.get('dropout_coverage') for t in study.trials):
        return
    params = [dict(lr=3e-4, weight_decay=.001, dropout=d) for d in (.05, .2)]
    for p in params:
        study.enqueue_trial(p, user_attrs={'dropout_coverage': True})
    write(root/'dropout-coverage-queue.json', dict(params=params,
        reason='After trial7, explicitly cover .05 and .2 rather than repeatedly sampling .1'))


@torch.inference_mode()
def diagnostic_metrics(model, loaders, artifacts, device):
    """Eval-mode objective/ranking, so dropout does not bias the train/val comparison."""
    model.eval()
    result = {}
    for size, loader in loaders.items():
        idx, y = artifacts[size]['idx'].to(device), artifacts[size]['y'].to(device)
        totals = dict(loss=0., hit1=0., mrr=0.)
        count = 0
        for batch in loader:
            scores = experiment.score_batch(model, batch, idx, device)
            n = len(scores)
            totals['loss'] += experiment.state_loss(scores, y).item() * n
            metrics = experiment.ranking_metrics(scores, y)
            for key in ('hit1', 'mrr'):
                totals[key] += metrics[key].sum().item()
            count += n
        result[f's{size}'] = {**{k: v / count for k, v in totals.items()}, 'states': count}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/gine-transformation-tuning'))
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--updates', type=int, default=30000)
    parser.add_argument('--queue-dropout-coverage', action='store_true')
    parser.add_argument('--long-check', action='store_true',
        help='After matched reruns, retrain the latest-peaking full model for 60k updates')
    args = parser.parse_args()
    if args.queue_dropout_coverage:
        queue_dropout_coverage(args.output)
        return
    if args.updates < 30000 or args.updates % 6 or args.trials < 6:
        raise ValueError('At least 30k updates (divisible by six) and six startup trials required')
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root / 'runs').mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    base = json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    manifest = dict(reference=str(reference_path), baseline=base, updates=args.updates,
        trials=args.trials, split='val', tuning_seed=0, final_seeds=[10, 11, 12],
        batch_size=16, eval_every=500, cpu_threads=4, architecture='edges-text',
        search=dict(lr=[1e-5, 5e-4], weight_decay=[1e-5, 1e-2], dropout=[0., .05, .1, .2]),
        pruning='Median, best-so-far Hit@1; six full startup trials; >=20k updates; >=4 references',
        selection='s4 Hit@1; within-run checkpoints use MRR tie-break',
        loss='probability_mass', scheduler=False, early_stopping=False,
        diagnostics='full validation and fixed 96 train states/size every 2k plus first/last; sparse gradient probe',
        final_arms=['tuned', 'tuned-original-only'],
        comparison='existing edges-text, original-only and ordinary GINE seeds10/11/12 at identical 30k budget')
    path = root / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing manifest differs')
    write(path, manifest)
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self): return reference['model_config']['in_dim']
        def encode(self, *a, **kw): raise RuntimeError('Expected cached frozen Qwen features')
    encoder = CacheOnly()
    nodes, edges, metadata = build_graph_features(encoder, reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph', **{**reference['feature_metadata']['options'],
        'use_spelling': False, 'transformation_nodes': True, 'max_transformations': 8,
        'transformation_min_length': 3})
    frame = experiment.load_connections(Path(reference['dataset_path']))
    dates = reference['split_dates']
    datasets = {}
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(dates[split])]
        assert subset.date.nunique() == len(dates[split])
        datasets[split], _ = make_state_loaders(subset, encoder, None, 16, shuffle=False,
            feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        print('PREPARED', split, {s: len(d) for s, d in datasets[split].items()}, flush=True)
    assert list(edges.feature_names) == reference['edge_feature_names']
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
        for s, d in datasets['val'].items()}
    generator = torch.Generator().manual_seed(20261101)
    diagnostic_indices = {s: torch.randperm(len(d), generator=generator)[:96].tolist()
        for s, d in datasets['train'].items()}
    write(root / 'diagnostic-indices.json', diagnostic_indices)
    train_eval = {s: DataLoader(Subset(d, diagnostic_indices[s]), batch_size=16, shuffle=False,
        collate_fn=collate_keep_wordlists) for s, d in datasets['train'].items()}
    artifacts = experiment.make_artifacts()

    def run(params, seed, label, original_only=False, trial=None, update_budget=None):
        budget = update_budget or args.updates
        path = root / 'runs' / f'{label}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['status'] == 'complete':
                assert saved['params'] == params and saved['seed'] == seed
                assert saved['training']['updates_ran'] == budget
                return saved
        model, config = build(reference, params, artifacts[4]['idx'], seed, nodes.dim, 'edges-text')
        if original_only:
            model.original_only = True
            config['original_only'] = True
        loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
            generator=torch.Generator().manual_seed(seed+s)) for s, d in datasets['train'].items()}
        history, diagnostics = [], []
        prefix_state = None
        prefix_key = (-float('inf'), -float('inf'))
        prefix_update = 0
        started = perf_counter()
        def callback(step, metrics, best, row):
            nonlocal prefix_state, prefix_key, prefix_update
            history.append(row)
            if update_budget is not None and step <= args.updates:
                key=(metrics[4]['hit1'],metrics[4]['mrr'])
                if key > prefix_key:
                    prefix_key, prefix_update = key, step
                    prefix_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
            if step == 500 or step % 2000 == 0 or step == budget:
                diagnostics.append(dict(update=step,
                    train=diagnostic_metrics(model, train_eval, artifacts, device),
                    validation=diagnostic_metrics(model, val_loaders, artifacts, device)))
            write(path, dict(status='running', label=label, params=params, seed=seed,
                history=history, diagnostics=diagnostics))
            write(root / 'progress.json', dict(label=label, update=step, best_hit1=best[0], params=params))
            if trial is not None:
                trial.report(best[0], step)
                if step >= 20000 and trial.should_prune():
                    raise optuna.TrialPruned(f'best Hit@1={best[0]:.4f} after {step} updates')
        try:
            model, summary = experiment.train(model, loaders, val_loaders, artifacts, device,
                lr=params['lr'], projection_lr=None, weight_decay=params['weight_decay'],
                updates=budget, eval_every=500, min_updates_before_stop=budget,
                patience=10000, min_delta=1e-3, evaluation_callback=callback,
                state_schedule=make_schedule('mixed', seed, updates=budget), diagnose_gradients=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            weights = Path('weights/experiments') / f'{stamp}-transformation-tuning-{label}.pt'
            prefix_result = None
            if update_budget is not None:
                prefix_model=deepcopy(model)
                prefix_model.load_state_dict(prefix_state)
                prefix_metrics=experiment.evaluate(prefix_model.to(device), val_loaders, artifacts, device, by_color=True)
                prefix_model.to('cpu')
                prefix_summary={**summary, 'updates_ran':args.updates, 'best_update':prefix_update,
                    'history':[h for h in summary['history'] if h['update']<=args.updates],
                    'gradient_diagnostics':[g for g in summary['gradient_diagnostics'] if g['update']<=args.updates],
                    'validation':{f's{s}':m for s,m in prefix_metrics.items()},
                    'validation_macro':{k:sum(m[k] for m in prefix_metrics.values())/len(prefix_metrics)
                        for k in ('hit1','mrr','hit4','map','ndcg4')}}
                prefix_weights=weights.with_name(weights.stem+'-prefix30k.pt')
                torch.save(dict(model_class='TransformationGINE',model_config=config,
                    model_state_dict=prefix_model.state_dict(),feature_metadata=metadata,
                    edge_feature_names=list(edges.feature_names),split_dates=dates,
                    dataset_path=reference['dataset_path'],embedding_model_name=reference['embedding_model_name'],
                    training_config=dict(variant=label+'-prefix30k',params=params,seed=seed,updates=args.updates,
                        reference_checkpoint=str(reference_path),ranking_loss='probability_mass',batch_size=16,
                        scheduler=False,original_only=original_only),training_summary=prefix_summary),prefix_weights)
                prefix_result=root/'runs'/f'{label}-prefix30k.json'
                write(prefix_result,dict(status='complete',label=label+'-prefix30k',params=params,seed=seed,
                    checkpoint=str(prefix_weights),model_config=config,training=prefix_summary,
                    diagnostics=[d for d in diagnostics if d['update']<=args.updates]))
                del prefix_model
            torch.save(dict(model_class='TransformationGINE', model_config=config,
                model_state_dict=model.state_dict(), feature_metadata=metadata,
                edge_feature_names=list(edges.feature_names), split_dates=dates,
                dataset_path=reference['dataset_path'], embedding_model_name=reference['embedding_model_name'],
                training_config=dict(variant=label, params=params, seed=seed, updates=budget,
                    reference_checkpoint=str(reference_path), ranking_loss='probability_mass',
                    batch_size=16, scheduler=False, original_only=original_only),
                training_summary=summary), weights)
            payload = dict(status='complete', label=label, params=params, seed=seed,
                checkpoint=str(weights), model_config=config, training=summary,
                diagnostics=diagnostics, elapsed_seconds=perf_counter()-started,
                prefix_result=str(prefix_result) if prefix_result else None,
                parameters=dict(total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            write(path, payload)
            return payload
        except optuna.TrialPruned:
            write(path, dict(status='pruned', label=label, params=params, seed=seed,
                history=history, diagnostics=diagnostics, elapsed_seconds=perf_counter()-started))
            raise
        finally:
            # Pruning raises inside the shared trainer, before its normal hook cleanup.
            # Drop the entire model; no model instance is reused between trials.
            del model, loaders
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(study_name='transformation_optimizer_v1',
        storage=f'sqlite:///{(root / "study.sqlite3").resolve()}', load_if_exists=True,
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=20261101, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=20000,
            interval_steps=2000, n_min_trials=4))
    if not study.trials:
        for overrides in ({}, {'lr': 1e-4}, {'lr': 3e-5},
            {'lr': 1e-4, 'dropout': .1, 'weight_decay': .001},
            {'lr': 3e-5, 'dropout': .1, 'weight_decay': .001},
            {'dropout': .1, 'weight_decay': .001}):
            p = {**base, **overrides}
            study.enqueue_trial({k: p[k] for k in ('lr', 'dropout', 'weight_decay')})
    def objective(trial):
        params = {**base, 'lr': trial.suggest_float('lr', 1e-5, 5e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'dropout': trial.suggest_categorical('dropout', [0., .05, .1, .2])}
        result = run(params, 0, f'trial-{trial.number:02d}', trial=trial)
        trial.set_user_attr('checkpoint', result['checkpoint'])
        trial.set_user_attr('best_update', result['training']['best_update'])
        if trial.number == 7:
            queue_dropout_coverage(root)
        return result['training']['validation']['s4']['hit1']
    finished = sum(t.state.is_finished() for t in study.trials)
    study.optimize(objective, n_trials=max(0, args.trials-finished), gc_after_trial=True)
    study.trials_dataframe().to_csv(root / 'trials.csv', index=False)
    best = dict(trial=study.best_trial.number, value=study.best_value,
        params={**base, **study.best_params})
    write(root / 'best.json', best)
    print('SELECTED', best, flush=True)
    entries = []
    for seed in (10, 11, 12):
        for variant, folder in (('control', 'gine-transformations'),
            ('edges-text', 'gine-transformations'), ('original-only', 'gine-original-only')):
            entries.append(dict(variant=variant, seed=seed,
                result=f'experiments/{folder}/{variant}-seed-{seed}.json',
                harness=f'experiments/{folder}/harness-{variant}-seed-{seed}-val.json'))
        for variant in ('tuned', 'tuned-original-only'):
            label = f'{variant}-seed-{seed}'
            payload = run(best['params'], seed, label, original_only=variant.endswith('original-only'))
            harness = root / f'harness-{label}-val.json'
            if not harness.exists():
                with harness.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                        '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                        '--output', str(harness)], stdout=log, stderr=subprocess.STDOUT, check=True)
            entries.append(dict(variant=variant, seed=seed, result=str(root/'runs'/f'{label}.json'),
                harness=str(harness)))
            write(root / 'runs.json', entries)
            print('COMPLETE', label, payload['training']['validation']['s4']['hit1'], flush=True)
    if args.long_check:
        finalists=[e for e in entries if e['variant']=='tuned']
        entry=max(finalists,key=lambda e:json.loads(Path(e['result']).read_text())['training']['best_update'])
        seed=entry['seed']
        payload=run(best['params'], seed, f'long-tuned-seed-{seed}', update_budget=60000)
        harness=root/f'harness-long-tuned-seed-{seed}-val.json'
        if not harness.exists():
            with harness.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                    '--split','val','--beam-width','20','--score-mode','full_board',
                    '--output',str(harness)],stdout=log,stderr=subprocess.STDOUT,check=True)
        prefix_path=Path(payload['prefix_result'])
        prefix=json.loads(prefix_path.read_text())
        prefix_harness=root/f'harness-long-tuned-seed-{seed}-prefix30k-val.json'
        if not prefix_harness.exists():
            with prefix_harness.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable,'scripts/evaluate_graph_feedback.py',prefix['checkpoint'],
                    '--split','val','--beam-width','20','--score-mode','full_board',
                    '--output',str(prefix_harness)],stdout=log,stderr=subprocess.STDOUT,check=True)
        write(root/'long-check.json',dict(seed=seed,
            selection='latest best checkpoint among the three tuned full-model reruns',
            matched_30k=entry, result=str(root/'runs'/f'long-tuned-seed-{seed}.json'),
            harness=str(harness),updates=60000,
            within_run_30k=dict(result=str(prefix_path),harness=str(prefix_harness)),
            training='one uninterrupted 60k run, compare best in first30k with best in60k; no optimizer reset',
            restart_note='Same seed did not reproduce prior stochastic trajectory across processes; incomplete initial extension archived'))
    print('ALL RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
