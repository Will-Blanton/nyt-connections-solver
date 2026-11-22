"""Sequential matched validation-only test of candidate-local edge amplification."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch
from torch.utils.data import DataLoader

import run_graph_feature_experiment as experiment
from graph_group_loss import ProbabilityFloorLoss
from run_gine_curriculum import make_schedule
from tune_gine import build_model
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features


ROOT = Path('experiments/gine-edge-coherence')
SCALES = {'control': 0., 'amplified': 1.}


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    params = json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    manifest = dict(reference=str(reference_path), params=params, seeds=[10, 11, 12],
        variants=SCALES, split='val', updates=15000, eval_every=250, batch_size=16, cpu_threads=4,
        feature_set='original 15; no new providers or node features',
        modification='e_ij,k * (1 + scale * min_i(mean_j!=i(e_ij,k))) within each candidate',
        loss='probability_mass_with_floor', penalty_weight=.1, beta=.1,
        exposure='5000 updates per size, shared mixed schedule and per-size batch permutations',
        selection='s4 Hit@1 with MRR tie-break; no early stopping, pruning, or scheduler',
        initialization='fresh matched control and intervention per seed; same state_dict before training',
        frozen='only cached Qwen embeddings; active LN, text projection, GINE and readout trainable',
        harness=dict(beam_width=20, score_mode='full_board', split='val'))
    if (ROOT / 'manifest.json').exists() and json.loads((ROOT / 'manifest.json').read_text()) != manifest:
        raise ValueError('Existing experiment has a different configuration')
    write(ROOT / 'manifest.json', manifest)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for the full experiment')
    device = torch.device('cuda')
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):
            return reference['model_config']['in_dim']

        def encode(self, *args, **kwargs):
            raise RuntimeError('Expected existing frozen feature cache')
    encoder = CacheOnly()
    nodes, edges, metadata = build_graph_features(encoder, reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph', **{**reference['feature_metadata']['options'],
        'use_spelling': False, 'include_phonetics': False, 'include_letter_edits': False,
        'phonetic_nodes': False, 'transformation_nodes': False})
    assert list(edges.feature_names) == reference['edge_feature_names']
    frame = experiment.load_connections(Path(reference['dataset_path']))
    dates = reference['split_dates']
    datasets = {}
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(dates[split])]
        assert subset.date.nunique() == len(dates[split])
        datasets[split], _ = make_state_loaders(subset, encoder, None, 16, shuffle=False,
            feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        print('PREPARED', split, {s: len(d) for s, d in datasets[split].items()}, flush=True)
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    # Read-only diagnostics: does coherence separate true groups and near misses?
    diagnostics = {}
    for split in ('train', 'val'):
        values = {key: [] for key in ('gold', 'near_miss', 'other')}
        idx, gold, near = (artifacts[4][key] for key in ('idx', 'y', 'near_miss'))
        for board in datasets[split][4]:
            candidate = board['edge_features'][idx[:, :, None], idx[:, None, :]]
            diagonal = torch.eye(4, dtype=torch.bool)
            coherence = (candidate.masked_fill(diagonal[..., None], 0).sum(-2) / 3).amin(-2)
            for key, mask in (('gold', gold), ('near_miss', near), ('other', ~gold & ~near)):
                values[key].append(coherence[mask].mean(0))
        diagnostics[split] = {key: torch.stack(rows).mean(0).tolist() for key, rows in values.items()}
    write(ROOT / 'coherence-diagnostics.json', dict(feature_names=list(edges.feature_names), splits=diagnostics))
    runs = []
    for seed in manifest['seeds']:
        schedule = make_schedule('mixed', seed)
        write(ROOT / f'schedule-seed-{seed}.json', schedule)
        for variant, scale in SCALES.items():
            path = ROOT / f'{variant}-seed-{seed}.json'
            payload = json.loads(path.read_text()) if path.exists() else None
            if not payload or payload.get('status') != 'complete':
                model, config = build_model(reference, params, artifacts[4]['idx'], seed)
                # No random draws or parameter changes relative to the fresh control.
                model.edge_coherence_scale = scale
                config['edge_coherence_scale'] = scale
                print('START', variant, seed, flush=True)
                loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
                    generator=torch.Generator().manual_seed(seed+s)) for s, d in datasets['train'].items()}
                history = []
                def progress(step, metrics, best, row):
                    history.append(row)
                    write(path, dict(status='running', variant=variant, seed=seed, history=history))
                    write(ROOT / 'progress.json', dict(variant=variant, seed=seed,
                        update=step, best_hit1=best[0], metrics=metrics))
                started = perf_counter()
                model, summary = experiment.train(model, loaders, val_loaders, artifacts, device,
                    lr=params['lr'], projection_lr=None, weight_decay=params['weight_decay'],
                    updates=15000, eval_every=250, min_updates_before_stop=15000,
                    patience=1000, min_delta=1e-3, evaluation_callback=progress,
                    state_schedule=schedule, ranking_loss=ProbabilityFloorLoss(weight=.1, beta=.1))
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                weights = Path('weights/experiments') / f'{stamp}-edge-coherence-{variant}-seed-{seed}.pt'
                torch.save(dict(model_class='StandaloneCandidateGINE', model_config=config,
                    model_state_dict=model.state_dict(), feature_metadata=metadata,
                    edge_feature_names=list(edges.feature_names), split_dates=dates,
                    dataset_path=reference['dataset_path'], embedding_model_name=reference['embedding_model_name'],
                    training_config=dict(variant=variant, seed=seed, params=params, updates=15000,
                        reference_checkpoint=str(reference_path), ranking_loss='probability_mass_with_floor',
                        penalty_weight=.1, beta=.1, edge_coherence_scale=scale), training_summary=summary), weights)
                payload = dict(status='complete', variant=variant, seed=seed, checkpoint=str(weights),
                    model_config=config, training=summary, elapsed_seconds=perf_counter()-started)
                write(path, payload)
                del model, loaders
                gc.collect()
                torch.cuda.empty_cache()
            harness = ROOT / f'harness-{variant}-seed-{seed}-val.json'
            if not harness.exists():
                with harness.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', payload['checkpoint'],
                        '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                        '--output', str(harness)], stdout=log, stderr=subprocess.STDOUT, check=True)
            runs.append(dict(variant=variant, seed=seed, result=str(path), harness=str(harness)))
            write(ROOT / 'runs.json', runs)
            print('COMPLETE', variant, seed, payload['training']['validation']['s4']['hit1'], flush=True)
    print('ALL RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
