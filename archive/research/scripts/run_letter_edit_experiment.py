"""Matched GINE substitution/deletion feature experiments, validation only."""
from time import perf_counter
from statistics import median
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
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features


from run_phonetic_experiment import expanded_model


def main():
    output = Path('experiments/gine-letter-edits')
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    params = json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    dates = reference['split_dates']
    manifest = dict(reference=str(reference_path), params=params, seeds=[10, 11, 12],
                    split='val', updates=15000, eval_every=250, batch_size=16,
                    variants=['control', 'substitution', 'deletion', 'both'],
                    additions='20 channels: 2 operations x 5 alignments x semantic/specificity',
                    initialization='identical shared weights, zero added edge columns',
                    selection='s4 Hit@1 with MRR checkpoint tie-break; no early stopping or pruning')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing results have a different configuration')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame = experiment.load_connections(Path(reference['dataset_path']))
    options = {**reference['feature_metadata']['options'], 'use_spelling': False, 'include_phonetics': False, 'include_letter_edits': True}
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                  cache_dir='data/feature_cache/graph', **options)
    provider = edges.providers[-1]
    used = frame[frame.date.isin(dates['train'] + dates['val'])]
    words = sorted({word for group in used.connections for word in group})
    started = perf_counter()
    vocabulary = provider.vocabulary(words)
    sizes = [len(provider.vocabulary([word])) for word in words]
    print('LEXICAL ALTERNATIVES', len(vocabulary), 'words', len(words),
          'median', median(sizes), 'max', max(sizes), flush=True)
    for start in range(0, len(vocabulary), 256):
        provider.embeddings.encode(vocabulary[start:start+256])
        if start % 2560 == 0:
            print('PREFETCH', start, '/', len(vocabulary), flush=True)
    preparation = dict(alternative_words=len(vocabulary), input_words=len(words),
                       median_alternatives=median(sizes), max_alternatives=max(sizes),
                       prefetch_seconds=perf_counter()-started)
    (output / 'feature-preparation.json').write_text(json.dumps(preparation, indent=2))
    print('PREFETCH COMPLETE', preparation, flush=True)
    started = perf_counter()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, original_threads))
    datasets = {}
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(dates[split])]
        if subset.date.nunique() != len(dates[split]):
            raise ValueError('Split no longer matches reference')
        datasets[split], _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                               feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        print('PREPARED', split, {s: len(d) for s, d in datasets[split].items()}, flush=True)
    preparation['dataset_preparation_seconds'] = perf_counter() - started
    (output / 'feature-preparation.json').write_text(json.dumps(preparation, indent=2))
    torch.set_num_threads(original_threads)
    embedder.to('cpu')
    torch.cuda.empty_cache()
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    old_dim = reference['model_config']['edge_dim']
    assert list(edges.feature_names[:old_dim]) == reference['edge_feature_names']
    assert edges.dim == old_dim + 20
    runs = []
    for seed in manifest['seeds']:
        for variant, channels in [('control', list(range(old_dim))),
                                  ('substitution', list(range(old_dim+10))),
                                  ('deletion', list(range(old_dim)) + list(range(old_dim+10, edges.dim))),
                                  ('both', list(range(edges.dim)))]:
            path = output / f'{variant}-seed-{seed}.json'
            payload = json.loads(path.read_text()) if path.exists() else None
            if not payload or payload.get('status') != 'complete':
                model, config = expanded_model(reference, params, artifacts[4]['idx'], seed, edges.dim, channels)
                loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
                                         generator=torch.Generator().manual_seed(seed+s))
                           for s, d in datasets['train'].items()}
                def progress(step, metrics, best, row):
                    (output / 'progress.json').write_text(json.dumps(dict(
                        seed=seed, variant=variant, update=step, best_hit1=best[0], metrics=metrics), indent=2))
                model, summary = experiment.train(
                    model, loaders, val_loaders, artifacts, device, lr=params['lr'], projection_lr=None,
                    weight_decay=params['weight_decay'], updates=15000, eval_every=250,
                    min_updates_before_stop=15000, patience=1000, min_delta=1e-3, evaluation_callback=progress)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                weights = Path('weights/experiments') / f'{stamp}-letter-edits-{variant}-seed-{seed}.pt'
                checkpoint = dict(model_class='StandaloneCandidateGINE', model_config=config,
                                  model_state_dict=model.state_dict(), feature_metadata=metadata,
                                  edge_feature_names=list(edges.feature_names), split_dates=dates,
                                  dataset_path=reference['dataset_path'], embedding_model_name=reference['embedding_model_name'],
                                  training_config=dict(variant=f'letter-edits-{variant}', seed=seed, params=params,
                                                       reference_checkpoint=str(reference_path), updates=15000),
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
