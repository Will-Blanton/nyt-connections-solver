"""Matched five-fold tuning of fresh GINE with/without quartet edge amplification.

Run with nix develop -c uv run python -u scripts/tune_gine_coherence_cv.py.
--background detaches a resumable worker and writes its PID and log under --output.
No learned weights are loaded from the metadata reference checkpoint.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import optuna
import torch
from torch.utils.data import DataLoader, Subset

import run_graph_feature_experiment as experiment
from graph_group_loss import ProbabilityFloorLoss
from run_gine_curriculum import make_schedule
from nyt_connections_solver.data_loader import ConnectionsDataset, GraphConnectionsDataset, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


ARMS = ('baseline', 'coherence')
FOLD_SEED = 20261106
STARTUP = [
    dict(hidden_width=128, gine_layers=2, readout_width=128, projection_width=256,
         dropout=0., lr=.0003016684667937828, weight_decay=.00009500890359387789),
    dict(hidden_width=64, gine_layers=2, readout_width=128, projection_width=256,
         dropout=.1, lr=3e-4, weight_decay=1e-4),
    dict(hidden_width=128, gine_layers=2, readout_width=128, projection_width=256,
         dropout=.2, lr=3e-4, weight_decay=.001),
    dict(hidden_width=64, gine_layers=1, readout_width=64, projection_width=128,
         dropout=.2, lr=5e-4, weight_decay=.001),
    dict(hidden_width=128, gine_layers=3, readout_width=256, projection_width=256,
         dropout=.3, lr=1e-4, weight_decay=.001),
    dict(hidden_width=32, gine_layers=2, readout_width=64, projection_width=128,
         dropout=.4, lr=3e-4, weight_decay=.01),
]


def write(path, value):
    """Atomic JSON replacement so progress remains readable after interruptions."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def make_folds(split_dates, seed=FOLD_SEED, count=5):
    """Use original train+val dates only; group every state by puzzle date."""
    train, val, excluded = (set(split_dates[k]) for k in ('train', 'val', 'test'))
    if train & val or (train | val) & excluded:
        raise ValueError('Reference date splits overlap')
    pool = sorted(train | val)
    if len(pool) < count:
        raise ValueError('Not enough puzzles for folds')
    random.Random(seed).shuffle(pool)
    folds = []
    for i in range(count):
        heldout = set(pool[i::count])
        folds.append(dict(fold=i, seed=1000+i,
            train=sorted(set(pool)-heldout), val=sorted(heldout)))
    assert set().union(*(set(f['val']) for f in folds)) == set(pool)
    assert sum(len(f['val']) for f in folds) == len(pool)
    return folds


def suggest(trial, arm):
    params = dict(
        hidden_width=trial.suggest_categorical('hidden_width', [32, 64, 128]),
        gine_layers=trial.suggest_int('gine_layers', 1, 3),
        readout_width=trial.suggest_categorical('readout_width', [64, 128, 256]),
        projection_width=trial.suggest_categorical('projection_width', [128, 256]),
        dropout=trial.suggest_categorical('dropout', [0., .1, .2, .3, .4]),
        lr=trial.suggest_float('lr', 3e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True))
    params['edge_coherence_scale'] = (0. if arm == 'baseline' else
        trial.suggest_float('edge_coherence_scale', .1, 3., log=True))
    return params


def make_pruner():
    # Steps are FULLY completed folds, never training updates or partial folds.
    return optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=3,
                                      interval_steps=1, n_min_trials=4)


def build_fresh_model(base_config, params, idx, seed):
    """Only shapes come from base_config; no state_dict/pretrained adapter input."""
    experiment.seed_everything(seed)
    config = deepcopy(base_config)
    config.update(edge_hidden=params['hidden_width'], gine_layers=params['gine_layers'],
        readout_width=params['readout_width'], projection_width=params['projection_width'],
        gine_dropout=params['dropout'], edge_bias_scale=0.,
        edge_coherence_scale=params['edge_coherence_scale'], train_text_projection=True,
        train_input_norm=True)
    model = StandaloneCandidateGINE(group_idx=idx, **config)
    assert all(p.requires_grad for module in (model.input_norm, model.proj, model.edge_encoder,
                                             model.group_readout) for p in module.parameters())
    return model, config


def fold_subsets(datasets, fold):
    """Index views share immutable precomputed data, never learned fold state."""
    sides = {}
    for side in ('train', 'val'):
        dates = set(fold[side])
        sides[side] = {s: Subset(d, [i for i, sample in enumerate(d) if sample['date'] in dates])
                       for s, d in datasets.items()}
        for s, subset in sides[side].items():
            expected = {2: 6, 3: 4, 4: 1}[s] * len(dates)
            if len(subset) != expected:
                raise ValueError('Dataset does not match fold puzzle dates')
    return sides


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/gine-coherence-cv'))
    parser.add_argument('--trials', type=int, default=16, help='Equal number of configurations per arm')
    parser.add_argument('--updates', type=int, default=30000, help='Updates per completed fold')
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.trials < 8 or args.updates < 15000 or args.updates % 6 or args.eval_every < 1:
        raise ValueError('Require >=8 trials, >=15000 updates divisible by six, positive eval interval')
    return args


def main():
    args = parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if args.background:
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   *[a for a in sys.argv[1:] if a != '--background']]
        with (root / 'run.log').open('a') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        write(root / 'launcher.json', dict(pid=child.pid, command=command,
                                         started=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(pid=child.pid, log=str(root/'run.log'), progress=str(root/'progress.json'))))
        return
    lock = (root / 'worker.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A CV worker already holds this output directory')
    write(root / 'worker.json', dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for the full CV sweep')
    device = torch.device('cuda')
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    # Drop all learned weights before constructing any model or fold dataset.
    reference.pop('model_state_dict')
    folds = make_folds(reference['split_dates'])
    manifest = dict(version=1, reference_metadata=str(reference_path), trials_per_arm=args.trials,
        updates_per_fold=args.updates, eval_every=args.eval_every, folds=folds, fold_seed=FOLD_SEED,
        pool='original train + validation; historical test dates excluded',
        pool_puzzles=sum(len(f['val']) for f in folds), initialization='all learned parameters fresh per fold',
        frozen='cached original Qwen embeddings only; inactive inherited attention modules unused',
        objective='mean of five fold best s4 Hit@1 values; MRR breaks checkpoint ties',
        selection_caveat='folds used for checkpoint/hyperparameter selection; not unbiased final test estimates',
        arms=list(ARMS), batch_size=16, cpu_threads=4, state_schedule='balanced mixed s2/s3/s4',
        loss='probability mass plus fixed floor weight .1, beta .1',
        features='original 15 edge channels; text-only Qwen nodes',
        pruning='median after six completed configurations and >=3 full folds; no within-fold pruning',
        search_space=dict(hidden_width=[32,64,128], gine_layers=[1,3], readout_width=[64,128,256],
            projection_width=[128,256], dropout=[0.,.1,.2,.3,.4], lr=[3e-5,1e-3],
            weight_decay=[1e-6,1e-2], edge_coherence_scale=[.1,3.]),
        startup_common=STARTUP, harness=dict(beam_width=20, score_mode='full_board', max_mistakes=4, seed=0))
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text()) != manifest:
        raise ValueError('Existing CV experiment has a different manifest')
    write(root/'manifest.json', manifest)
    write(root/'status.json', dict(status='preparing', pid=os.getpid()))
    frame = experiment.load_connections(Path(reference['dataset_path']))
    pool_dates = set().union(*(set(f['val']) for f in folds))
    frame = frame[frame.date.isin(pool_dates)]
    assert frame.date.nunique() == len(pool_dates)
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):
            return reference['model_config']['in_dim']

        def encode(self, *args, **kwargs):
            raise RuntimeError('Expected existing frozen Qwen feature cache')
    encoder = CacheOnly()
    nodes, edges, metadata = build_graph_features(encoder, reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph', **{**reference['feature_metadata']['options'],
        'use_spelling':False, 'include_phonetics':False, 'include_letter_edits':False,
        'phonetic_nodes':False, 'transformation_nodes':False})
    assert list(edges.feature_names) == reference['edge_feature_names']
    boards = ConnectionsDataset.prepare(frame, encoder, None, nodes, edges)
    datasets = {s: GraphConnectionsDataset(None, num_sets_remaining=s, prepared_boards=boards)
                for s in (2,3,4)}
    subsets = {f['fold']: fold_subsets(datasets, f) for f in folds}
    artifacts = experiment.make_artifacts()
    print('PREPARED', len(pool_dates), 'puzzles; folds', [len(f['val']) for f in folds], flush=True)
    studies = {}
    for arm in ARMS:
        study = optuna.create_study(study_name=arm, storage=f'sqlite:///{(root/"study.sqlite3").resolve()}',
            direction='maximize', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=FOLD_SEED, n_startup_trials=6), pruner=make_pruner())
        if not study.trials:
            for i, common in enumerate(STARTUP):
                params = dict(common)
                if arm == 'coherence':
                    params['edge_coherence_scale'] = [1.,.5,2.,1.,.25,1.][i]
                study.enqueue_trial(params)
        studies[arm] = study

    def run_fold(arm, trial, params, fold):
        directory = root/arm/f'trial-{trial.number:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f'fold-{fold["fold"]}.json'
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get('status') == 'complete':
                assert prior['params'] == params and prior['fold'] == fold['fold']
                return prior
        model, config = build_fresh_model(reference['model_config'], params, artifacts[4]['idx'], fold['seed'])
        views = subsets[fold['fold']]
        train_loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
            generator=torch.Generator().manual_seed(fold['seed']+s)) for s,d in views['train'].items()}
        val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                       for s,d in views['val'].items()}
        history = []
        started = perf_counter()
        print('START', arm, trial.number, 'fold', fold['fold'], json.dumps(params), flush=True)
        def callback(step, metrics, best, row):
            history.append(row)
            write(path, dict(status='running', arm=arm, trial=trial.number, fold=fold['fold'],
                params=params, history=history))
            write(root/'progress.json', dict(arm=arm, trial=trial.number, fold=fold['fold'],
                update=step, updates=args.updates, best_hit1=best[0],
                completed_fold_seconds=perf_counter()-started))
        try:
            model, summary = experiment.train(model, train_loaders, val_loaders, artifacts, device,
                lr=params['lr'], projection_lr=None, weight_decay=params['weight_decay'],
                updates=args.updates, eval_every=args.eval_every, min_updates_before_stop=args.updates,
                patience=100000, min_delta=1e-3, evaluation_callback=callback,
                state_schedule=make_schedule('mixed', fold['seed'], updates=args.updates),
                ranking_loss=ProbabilityFloorLoss(weight=.1, beta=.1))
            checkpoint = directory/f'fold-{fold["fold"]}.pt'
            torch.save(dict(model_class='StandaloneCandidateGINE', model_config=config,
                model_state_dict=model.state_dict(), embedding_model_name=reference['embedding_model_name'],
                feature_metadata=metadata, edge_feature_names=list(edges.feature_names),
                split_dates=dict(train=fold['train'], val=fold['val'], test=[]),
                dataset_path=reference['dataset_path'],
                training_config=dict(variant='five-fold-'+arm, params=params, seed=fold['seed'],
                    initialization='fresh; no learned checkpoint weights', updates=args.updates,
                    cv_fold=fold['fold'], cv_manifest=str(root/'manifest.json'),
                    ranking_loss='probability_mass_with_floor', penalty_weight=.1, beta=.1),
                training_summary=summary), checkpoint)
            payload = dict(status='complete', arm=arm, trial=trial.number, fold=fold['fold'], params=params,
                checkpoint=str(checkpoint), training=summary, elapsed_seconds=perf_counter()-started)
            write(path, payload)
            print('FOLD COMPLETE', arm, trial.number, fold['fold'],
                  summary['validation']['s4']['hit1'], flush=True)
            return payload
        finally:
            del model, train_loaders, val_loaders
            gc.collect()
            torch.cuda.empty_cache()

    def objective(arm, trial):
        params = suggest(trial, arm)
        scores, paths = [], []
        for fold in folds:
            payload = run_fold(arm, trial, params, fold)
            scores.append(payload['training']['validation']['s4']['hit1'])
            paths.append(str(root/arm/f'trial-{trial.number:02d}'/f'fold-{fold["fold"]}.json'))
            trial.set_user_attr('fold_results', paths)
            trial.set_user_attr('fold_scores', scores)
            trial.report(sum(scores)/len(scores), len(scores))
            if 3 <= len(scores) < 5 and trial.should_prune():
                raise optuna.TrialPruned(f'{len(scores)} complete folds: mean Hit@1 {sum(scores)/len(scores):.4f}')
        return sum(scores)/len(scores)

    # Alternate configurations across arms instead of completing one whole search first.
    write(root/'status.json', dict(status='tuning', pid=os.getpid()))
    subprocess.run([sys.executable, 'scripts/summarize_gine_coherence_cv.py', '--output', str(root)], check=True)
    for _ in range(args.trials):
        for arm, study in studies.items():
            finished = [t for t in study.trials if t.state.is_finished()]
            if len(finished) >= args.trials:
                continue
            running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
            if len(running) > 1:
                raise RuntimeError('Unexpected multiple running trials')
            trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
            try:
                value = objective(arm, trial)
                study.tell(trial, value)
                print('TRIAL COMPLETE', arm, trial.number, value, flush=True)
            except optuna.TrialPruned as reason:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print('TRIAL PRUNED', arm, trial.number, str(reason), flush=True)
            study.trials_dataframe().to_csv(root/f'trials-{arm}.csv', index=False)
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                best = study.best_trial
                write(root/f'best-{arm}.json', dict(trial=best.number, mean_hit1=best.value,
                    params=best.params, fold_results=best.user_attrs['fold_results']))
            subprocess.run([sys.executable, 'scripts/summarize_gine_coherence_cv.py', '--output', str(root)], check=True)

    write(root/'status.json', dict(status='harness', pid=os.getpid()))
    final = []
    for arm, study in studies.items():
        best = study.best_trial
        for result_path in best.user_attrs['fold_results']:
            payload = json.loads(Path(result_path).read_text())
            fold = folds[payload['fold']]
            checkpoint = torch.load(payload['checkpoint'], map_location='cpu', weights_only=True)
            assert checkpoint['split_dates']['train'] == fold['train']
            assert checkpoint['split_dates']['val'] == fold['val'] and checkpoint['split_dates']['test'] == []
            model = StandaloneCandidateGINE(group_idx=artifacts[4]['idx'], **checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            loader = DataLoader(subsets[fold['fold']]['val'][4], batch_size=8,
                                shuffle=False, collate_fn=collate_keep_wordlists)
            checked = experiment.evaluate(model.to(device), {4:loader}, artifacts, device)[4]
            assert abs(checked['hit1']-payload['training']['validation']['s4']['hit1']) < 1e-8
            del model, checkpoint
            torch.cuda.empty_cache()
            harness = root/f'harness-{arm}-fold-{fold["fold"]}.json'
            if not harness.exists():
                with harness.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                        '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                        '--output', str(harness)], stdout=log, stderr=subprocess.STDOUT, check=True)
            final.append(dict(arm=arm, fold=fold['fold'], trial=best.number,
                result=result_path, harness=str(harness), audited_hit1=checked['hit1']))
            write(root/'final-runs.json', final)
    write(root/'status.json', dict(status='complete', pid=os.getpid(),
                                 finished=datetime.now(timezone.utc).isoformat()))
    subprocess.run([sys.executable, 'scripts/summarize_gine_coherence_cv.py', '--output', str(root)], check=True)
    print('ALL CV TUNING AND HARNESS EVALUATIONS COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # A failure must not leave a dead worker looking like an active sweep.
        root = parse_args().output
        worker = root/'worker.json'
        if worker.exists() and json.loads(worker.read_text()).get('pid') == os.getpid():
            write(root/'status.json', dict(status='failed', pid=os.getpid(), error=repr(error)))
        raise
