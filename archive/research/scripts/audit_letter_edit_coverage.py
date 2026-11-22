"""Validation-only lexical coverage, witnesses, and warm-resource feature latency."""
import itertools
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch
import torch.nn.functional as F

from run_graph_feature_experiment import load_connections, make_artifacts
from audit_graph_edge_coverage import candidate_evidence
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.data_loader import ConnectionsDataset
from nyt_connections_solver.edge_features import LetterEditSimilarity


EXAMPLES = {
    '2025-03-07': ('substitution', ['apple', 'date', 'lemon', 'peach']),
    '2026-01-04': ('deletion', ['rear', 'tail', 'rump', 'can']),
    '2026-01-13': ('deletion', ['toe', 'arch', 'ankle', 'heel']),
    '2026-02-28': ('deletion', ['comet', 'nova', 'star', 'sun']),
    '2026-05-01': ('substitution', ['five', 'six', 'four', 'eight']),
    '2026-05-12': ('deletion', ['franc', 'rand', 'real', 'won']),
    '2026-05-31': ('deletion', ['ash', 'oak', 'pine', 'teak']),
    '2026-09-25': ('deletion', ['order', 'class', 'family', 'genus']),
}


def main():
    torch.set_num_threads(4)
    root = Path('experiments/gine-letter-edits')
    reference = torch.load('weights/experiments/20261023T150230Z-text-only.pt', map_location='cpu', weights_only=True)

    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):
            return reference['model_config']['in_dim']

        def encode(self, *args, **kwargs):
            raise RuntimeError('Precompute feature embeddings before running the audit')

    encoder = CacheOnly()
    nodes, edges, _ = build_graph_features(
        encoder, reference['embedding_model_name'], cache_dir='data/feature_cache/graph',
        **{**reference['feature_metadata']['options'], 'include_letter_edits': True})
    provider = edges.providers[-1]
    frame = load_connections(Path(reference['dataset_path']))
    frame = frame[frame.date.isin(reference['split_dates']['val'])]
    boards = ConnectionsDataset.prepare(frame, encoder, None, nodes, edges)
    assert len(boards) == 121
    artifact = make_artifacts()[4]
    targets = artifact['y'].nonzero().flatten()
    inventory = json.loads(Path('experiments/gine-phonetics/mechanism-inventory.json').read_text())
    labels = {(r['date'], r['color']): r for r in inventory['groups_inventory']}
    rows, examples = [], []
    for board in boards:
        values, pooled, percentiles = candidate_evidence(board['edge_features'], artifact['idx'], artifact['y'])
        date = str(board['date'])
        for ci, color in enumerate(('purple', 'blue', 'green', 'yellow')):
            source = labels[(date, color)]
            row = dict(date=date, color=color, category=source['category'], mechanism=source['mechanism'],
                       words=source['words'], channels={name: dict(
                           any_pair=bool((values[targets[ci], :, j] > 0).any()),
                           all_pairs=bool((values[targets[ci], :, j] > 0).all()),
                           mean=float(pooled[targets[ci], j]), incorrect_percentile=float(percentiles[ci, j]))
                           for j, name in enumerate(edges.feature_names)})
            rows.append(row)
            if color != 'purple' or date not in EXAMPLES:
                continue
            operation, intended = EXAMPLES[date]
            words = board['words'][:4]
            found = [target in provider.readings(word).get((operation, 'anywhere', 0), ())
                     for word, target in zip(words, intended)]
            vocabulary = provider.vocabulary(words)
            vectors = F.normalize(provider.embeddings.encode(vocabulary), dim=-1)
            lookup = {word: i for i, word in enumerate(vocabulary)}
            witnesses = []
            for i, j in itertools.combinations(range(4), 2):
                pairs = provider.aligned_pairs(provider.readings(words[i]), provider.readings(words[j]),
                                               operation, 'anywhere')
                scores = torch.tensor([float((vectors[lookup[a]] * vectors[lookup[b]]).sum()) for a, b in pairs])
                a, b = pairs[int(scores.argmax())]
                witnesses.append(dict(left_word=words[i], right_word=words[j], left_reading=a,
                                      right_reading=b, cosine=float(scores.max()),
                                      intended_pair=(a == intended[i] and b == intended[j])))
            examples.append(dict(date=date, category=source['category'], operation=operation,
                                 words=words, intended=intended, intended_retrieved=found,
                                 max_witnesses=witnesses, channels=row['channels']))
    assert len(examples) == 8
    summary = {}
    for subset in ('all_purple', 'letter_edit', 'directly_supported', 'substitution_targets', 'deletion_targets'):
        selected = [r for r in rows if (r['color'] == 'purple' if subset == 'all_purple' else
                    r['mechanism'] == 'letter_edit' if subset == 'letter_edit' else
                    r['color'] == 'purple' and r['date'] in EXAMPLES and
                    (subset == 'directly_supported' or EXAMPLES[r['date']][0] == subset.removesuffix('_targets')))]
        summary[subset] = dict(groups=len(selected), channels={name: dict(
            any_pair=mean(r['channels'][name]['any_pair'] for r in selected),
            all_pairs=mean(r['channels'][name]['all_pairs'] for r in selected),
            incorrect_percentile=mean(r['channels'][name]['incorrect_percentile'] for r in selected))
            for name in edges.feature_names})
    fresh = LetterEditSimilarity(provider.lexicon, provider.embeddings)
    encoded_times, cache_times = [], []
    for board in boards:
        started = perf_counter()
        fresh.encode(board['words'])
        encoded_times.append(perf_counter() - started)
        started = perf_counter()
        edges.encode(board['words'])
        cache_times.append(perf_counter() - started)
    def timing(values):
        return dict(mean_ms=mean(values)*1000, median_ms=median(values)*1000,
                    p95_ms=sorted(values)[int(.95*(len(values)-1))]*1000)
    latency = dict(boards=121, cpu_threads=4, warm_resource_uncached_family=timing(encoded_times),
                   all_families_warm_matrix_cache=timing(cache_times),
                   excludes='Qwen inference for unseen alternatives, initial model/corpus loading')
    output = dict(summary=summary, examples=examples, groups=rows, latency=latency)
    (root / 'feature-coverage.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(intended_readings_retrieved=sum(sum(e['intended_retrieved']) for e in examples),
                         intended_readings_total=32, latency=latency), indent=2), flush=True)


if __name__ == '__main__':
    main()
