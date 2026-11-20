"""Validation-only near-miss diagnostics for saved graph experiment runs."""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from run_graph_feature_experiment import load_connections, make_artifacts
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.data_loader import ConnectionsDataset
from nyt_connections_solver.checkpoints import GRAPH_MODEL_TYPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('experiments/gine-fragment-states'))
    root = parser.parse_args().results
    runs = json.loads((root / 'runs.json').read_text())
    first = json.loads(Path(runs[0]['result']).read_text())
    reference = torch.load(first['checkpoint'], map_location='cpu', weights_only=True)
    identity = reference['feature_metadata']['nodes']
    while 'base' in identity:
        identity = identity['base']

    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):
            return identity['embedding']['dim']

        def encode(self, *args, **kwargs):
            raise RuntimeError('Audit requires existing feature caches')

    encoder = CacheOnly()
    nodes, edges, metadata = build_graph_features(
        encoder, reference['embedding_model_name'], cache_dir='data/feature_cache/graph',
        **reference['feature_metadata']['options'])
    frame = load_connections(Path(reference['dataset_path']))
    frame = frame[frame.date.isin(reference['split_dates']['val'])]
    boards = ConnectionsDataset.prepare(frame, encoder, None, nodes, edges)
    assert len(boards) == len(reference['split_dates']['val'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    artifact = {k: v.to(device) for k, v in make_artifacts()[4].items()}
    idx, positives, near = artifact['idx'], artifact['y'], artifact['near_miss']
    true = idx[positives]
    overlap = (idx[:, None, :, None] == true[None, :, None, :]).any(-1).sum(-1).max(-1).values
    records = []
    for run in runs:
        result = json.loads(Path(run['result']).read_text())
        checkpoint = torch.load(result['checkpoint'], map_location='cpu', weights_only=True)
        assert checkpoint['feature_metadata'] == metadata
        assert checkpoint['split_dates'] == reference['split_dates']
        model = GRAPH_MODEL_TYPES[checkpoint['model_class']](group_idx=idx, **checkpoint['model_config']).to(device).eval()
        model.load_state_dict(checkpoint['model_state_dict'])
        with torch.inference_mode():
            for start in range(0, len(boards), 8):
                batch = boards[start:start+8]
                scores = model(torch.stack([b['x'] for b in batch]).to(device), group_idx=idx,
                               edge_features=torch.stack([b['edge_features'] for b in batch]).to(device)).squeeze(-1).float()
                best = scores.argmax(-1)
                margin = scores[:, positives].max(-1).values - scores[:, near].max(-1).values
                mass = scores.softmax(-1)[:, near].sum(-1)
                for board, choice, gap, probability in zip(batch, best.tolist(), margin.tolist(), mass.tolist()):
                    records.append(dict(variant=run['variant'], seed=run['seed'], date=str(board['date']),
                                        correct=bool(positives[choice]), top1_max_overlap=int(overlap[choice]),
                                        best_positive_minus_best_near_miss=gap, near_miss_probability_mass=probability))
        del model
        torch.cuda.empty_cache()
        print('AUDITED', run['variant'], run['seed'], flush=True)
    aggregate = {}
    for variant in sorted({r['variant'] for r in runs}):
        rows = [r for r in records if r['variant'] == variant]
        wrong = [r for r in rows if not r['correct']]
        aggregate[variant] = dict(
            boards=len(rows), hit1=mean(r['correct'] for r in rows),
            wrong_top1_3over4_rate=mean(not r['correct'] and r['top1_max_overlap'] == 3 for r in rows),
            wrong_top1_3over4_given_wrong=mean(r['top1_max_overlap'] == 3 for r in wrong) if wrong else None,
            positive_beats_best_near_miss=mean(r['best_positive_minus_best_near_miss'] > 0 for r in rows),
            near_miss_probability_mass=mean(r['near_miss_probability_mass'] for r in rows))
    (root / 'near-miss-audit.json').write_text(json.dumps(dict(aggregate=aggregate, records=records), indent=2) + '\n')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
