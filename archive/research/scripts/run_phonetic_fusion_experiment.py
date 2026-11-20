"""Validation-only phonetic node representation/fusion sweep on tuned GINE."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

import run_graph_feature_experiment as experiment
from run_phonetic_experiment import expanded_model
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.phonetic_models import PhoneticFusionGINE


VARIANTS = {
    'edge_only': dict(phonetic_encoding='ngram', phonetic_fusion='linear', phonetic_enabled=False),
    'ngram_linear': dict(phonetic_encoding='ngram', phonetic_fusion='linear', phonetic_enabled=True),
    'ngram_gated': dict(phonetic_encoding='ngram', phonetic_fusion='gated', phonetic_enabled=True),
    'cnn_linear': dict(phonetic_encoding='cnn', phonetic_fusion='linear', phonetic_enabled=True),
    'cnn_gated': dict(phonetic_encoding='cnn', phonetic_fusion='gated', phonetic_enabled=True),
}


def fusion_model(reference, params, idx, seed, nodes, edge_dim, variant, sound_edges=True):
    old_dim = reference['model_config']['edge_dim']
    active = range(old_dim+4 if sound_edges else old_dim)  # No substring-sound channels in this sweep.
    base, config = expanded_model(reference, params, idx, seed, edge_dim, active)
    config.update(in_dim=nodes.dim, text_dim=nodes.base.dim, phonetic_dim=nodes.ngram_dim,
                  max_pronunciations=nodes.max_pronunciations, max_phones=nodes.max_phones,
                  phone_vocab_size=len(nodes.vocabulary)+1, **variant)
    with torch.random.fork_rng():
        model = PhoneticFusionGINE(group_idx=idx, **config)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if unexpected or any(not name.startswith(('phone_', 'phonetic_')) for name in missing):
        raise ValueError('Unexpected mismatch initializing phonetic fusion')
    return model, config


def main():
    output = Path('experiments/gine-phonetic-fusion')
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    params = json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    manifest = dict(reference=str(reference_path), params=params, seeds=[10, 11, 12],
                    split='val', updates=15000, eval_every=250, batch_size=16, variants=VARIANTS,
                    sound_edges='original 15 + 4 whole-word sound channels; substring sound masked',
                    node_payload=dict(ngram_dim=128, max_pronunciations=8, max_phones=32),
                    selection='mean validation s4 Hit@1, mean MRR tie-break',
                    final_ablation='best enabled-node fusion with sound edges disabled, same three seeds',
                    initialization='same old model weights; zero phonetic projection; gated initial sigmoid=0.5')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing results have a different setup')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame = experiment.load_connections(Path(reference['dataset_path']))
    dates = reference['split_dates']
    options = {**reference['feature_metadata']['options'], 'use_spelling': False,
               'include_phonetics': True, 'phonetic_nodes': True}
    embedder = SentenceTransformer(reference['embedding_model_name'], revision=options['revision'], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference['embedding_model_name'],
                                                  cache_dir='data/feature_cache/graph', **options)
    datasets, coverage = {}, {}
    for split in ('train', 'val'):
        subset = frame[frame.date.isin(dates[split])]
        if subset.date.nunique() != len(dates[split]):
            raise ValueError('Split does not match the reference')
        datasets[split], _ = make_state_loaders(subset, embedder, None, 16, shuffle=False,
                                               feature_builder=nodes, edge_providers=edges, cache=edges.cache)
        words = sorted({w for group in subset.connections for w in group})
        pronunciations = [nodes.pronunciations.phones(w) for w in words]
        coverage[split] = dict(unique_words=len(words), known=sum(bool(p) for p in pronunciations),
                               capped_variants=sum(len(p) > nodes.max_pronunciations for p in pronunciations),
                               truncated_words=sum(any(len(v) > nodes.max_phones for v in p) for p in pronunciations))
        print('PREPARED', split, coverage[split], flush=True)
    (output / 'feature-coverage.json').write_text(json.dumps(dict(
        node_dim=nodes.dim, text_dim=nodes.base.dim, phone_vocab=list(nodes.vocabulary), splits=coverage,
        metadata=metadata), indent=2) + '\n')
    embedder.to('cpu')
    torch.cuda.empty_cache()
    val_loaders = {s: DataLoader(d, batch_size=16, shuffle=False, collate_fn=collate_keep_wordlists)
                   for s, d in datasets['val'].items()}
    artifacts = experiment.make_artifacts()
    runs = []

    def run_one(label, variant, seed, sound_edges=True):
        path = output / f'{label}-seed-{seed}.json'
        result = json.loads(path.read_text()) if path.exists() else None
        if not result or result.get('status') != 'complete':
            model, config = fusion_model(reference, params, artifacts[4]['idx'], seed, nodes,
                                          edges.dim, variant, sound_edges)
            loaders = {s: DataLoader(d, batch_size=16, shuffle=True, collate_fn=collate_keep_wordlists,
                                     generator=torch.Generator().manual_seed(seed+s))
                       for s, d in datasets['train'].items()}
            def progress(step, metrics, best, row):
                (output / 'progress.json').write_text(json.dumps(dict(
                    variant=label, seed=seed, update=step, best_hit1=best[0], metrics=metrics), indent=2))
            model, summary = experiment.train(
                model, loaders, val_loaders, artifacts, device, lr=params['lr'], projection_lr=None,
                weight_decay=params['weight_decay'], updates=15000, eval_every=250,
                min_updates_before_stop=15000, patience=1000, min_delta=1e-3, evaluation_callback=progress)
            gate_stats = []
            if variant['phonetic_enabled']:
                with torch.inference_mode():
                    for batch in val_loaders[4]:
                        _, projected, gate = model.fusion_parts(batch['x'])
                        _, known = model.encode_phonetics(batch['x'])
                        gate_stats.extend(gate[known].tolist())
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            weights = Path('weights/experiments') / f'{stamp}-phonetic-fusion-{label}-seed-{seed}.pt'
            training_config = dict(variant=label, seed=seed, params=params, updates=15000,
                                   reference_checkpoint=str(reference_path), sound_edges=sound_edges)
            checkpoint = dict(model_class='PhoneticFusionGINE', model_config=config,
                              model_state_dict=model.state_dict(), feature_metadata=metadata,
                              edge_feature_names=list(edges.feature_names), split_dates=dates,
                              dataset_path=reference['dataset_path'], embedding_model_name=reference['embedding_model_name'],
                              training_config=training_config, training_summary=summary)
            torch.save(checkpoint, weights)
            result = dict(status='complete', variant=label, seed=seed, checkpoint=str(weights),
                          training=summary, model_config=config,
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          mean_known_word_gate=mean(gate_stats) if gate_stats else None)
            path.write_text(json.dumps(result, indent=2) + '\n')
            del model, loaders
            gc.collect()
            torch.cuda.empty_cache()
        harness_path = output / f'harness-{label}-seed-{seed}-val.json'
        if not harness_path.exists():
            with harness_path.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, 'scripts/evaluate_graph_feedback.py', result['checkpoint'],
                                '--split', 'val', '--beam-width', '20', '--score-mode', 'full_board',
                                '--output', str(harness_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
        item = dict(variant=label, seed=seed, result=str(path), harness=str(harness_path))
        runs.append(item)
        (output / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
        print('COMPLETE', label, seed, result['training']['validation']['s4']['hit1'], flush=True)
        return result

    sweep = []
    for seed in manifest['seeds']:
        for label, variant in VARIANTS.items():
            sweep.append(run_one(label, variant, seed))
    candidates = [label for label in VARIANTS if label != 'edge_only']
    winner = max(candidates, key=lambda label: tuple(mean(r['training']['validation']['s4'][m]
                    for r in sweep if r['variant'] == label) for m in ('hit1', 'mrr')))
    selection = dict(variant=winner, metric=manifest['selection'], selection_split='val',
                     note='Exploratory validation selection; not an independent held-out comparison')
    (output / 'selected.json').write_text(json.dumps(selection, indent=2) + '\n')
    print('SELECTED', winner, flush=True)
    for seed in manifest['seeds']:
        run_one('selected_nodes_only', VARIANTS[winner], seed, sound_edges=False)
    print('ALL RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
