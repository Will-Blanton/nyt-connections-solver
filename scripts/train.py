#!/usr/bin/env python3
"""Train the selected GINE baseline from scratch, without research artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from nyt_connections_solver.data_loader import load_connections, make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.training import build_model, candidate_artifacts, train_fixed, validate_splits


class CacheOnlyEncoder(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, *args, **kwargs):
        raise RuntimeError('Missing frozen Qwen vector in cache; rerun without --cache-only')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/gine-baseline.json')
    parser.add_argument('--dataset', type=Path, help='Override the configured historical CSV')
    parser.add_argument('--output-dir', type=Path, help='New directory; existing directories are rejected')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'data/feature_cache/graph')
    parser.add_argument('--cache-only', action='store_true', help='Fail on cache misses; do not load Qwen')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, help='Default: selected deployment seed, 10')
    parser.add_argument('--updates', type=int, help='Override fixed budget; positive multiple of three')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-every', type=int, default=500)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    splits = config['split_dates']
    validate_splits(splits)
    training = dict(config['training'])
    for key in ('seed', 'updates', 'batch_size'):
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    if training['updates'] < 3 or training['updates'] % 3 or training['batch_size'] < 1 or args.threads < 1 or args.log_every < 1:
        parser.error('Require positive batch size/threads/log interval and updates divisible by three')
    dataset = args.dataset or ROOT / config['dataset_path']
    frame = load_connections(dataset)
    selected = frame[frame.date.isin(splits['train'])]
    if set(selected.date) != set(splits['train']):
        raise ValueError('Dataset is missing configured training dates')
    resolved = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(resolved)
    torch.set_num_threads(args.threads)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir or ROOT / 'weights' / f'{stamp}-gine-seed-{training["seed"]}'
    output.mkdir(parents=True, exist_ok=False)
    effective = {**config, 'training': training, 'dataset_path': str(dataset.resolve())}
    write_json(output / 'config.json', effective)
    write_json(output / 'status.json', {'status': 'preparing'})
    try:
        if args.cache_only:
            encoder = CacheOnlyEncoder(config['model_config']['in_dim'])
        else:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(config['embedding_model_name'],
                revision=config['feature_options']['revision'], device=str(device))
        nodes, edges, metadata = build_graph_features(
            encoder, config['embedding_model_name'], cache_dir=args.cache_dir,
            **config['feature_options'])
        if (metadata != config['feature_metadata'] or list(edges.feature_names) != config['edge_feature_names']
                or nodes.dim != config['model_config']['in_dim'] or edges.dim != config['model_config']['edge_dim']):
            raise ValueError('Feature definitions/resources differ from the selected baseline')
        datasets, _ = make_state_loaders(selected, encoder, None, training['batch_size'],
            shuffle=False, feature_builder=nodes, edge_providers=edges)
        encoder.to('cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        artifacts = candidate_artifacts()
        model = build_model(config['model_config'], artifacts[4]['idx'], training['seed'])
        loaders = {size: DataLoader(data, batch_size=training['batch_size'], shuffle=True,
            collate_fn=collate_keep_wordlists,
            generator=torch.Generator().manual_seed(training['seed'] + size))
            for size, data in datasets.items()}
        print(f'Training {len(splits["train"])} puzzles; states '
              f'{ {s: len(d) for s, d in datasets.items()} }; device {device}', flush=True)
        print('Fresh learned layers; frozen Qwen; fixed final checkpoint; no held-out evaluation.', flush=True)
        model, summary = train_fixed(model, loaders, artifacts, device,
            seed=training['seed'], updates=training['updates'], lr=training['lr'],
            weight_decay=training['weight_decay'], floor_weight=training['floor_weight'],
            floor_beta=training['floor_beta'], log_every=args.log_every,
            callback=lambda row: write_json(output / 'status.json', {'status': 'training', **row}))
        checkpoint = output / 'model.pt'
        with dataset.open('rb') as stream:
            dataset_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        torch.save(dict(
            model_class='StandaloneCandidateGINE', model_config=config['model_config'],
            model_state_dict=model.state_dict(), embedding_model_name=config['embedding_model_name'],
            feature_metadata=metadata, edge_feature_names=list(edges.feature_names),
            dataset_path=str(dataset.resolve()), split_dates=splits,
            training_config={**training, 'recipe': config['recipe'], 'dataset_sha256': dataset_sha,
                'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                'harness': config['harness']}, training_summary=summary), checkpoint)
        with checkpoint.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        write_json(output / 'training-summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete', 'checkpoint': str(checkpoint),
                                           'sha256': checksum})
        print(f'Saved {checkpoint}\nSHA-256: {checksum}', flush=True)
    except Exception as error:
        write_json(output / 'status.json', {'status': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
