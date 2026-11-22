"""Validation-only gold-group probabilities and ranks from saved checkpoints."""
import json
import argparse
from pathlib import Path

import torch

from run_graph_feature_experiment import load_connections, make_artifacts
from nyt_connections_solver.checkpoints import GRAPH_MODEL_TYPES
from nyt_connections_solver.data_loader import ConnectionsDataset
from nyt_connections_solver.graph_features import build_graph_features


ROOT = Path('experiments/gine-probability-floor')


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT)
    ROOT = parser.parse_args().results
    torch.set_num_threads(4)
    runs = json.loads((ROOT / 'runs.json').read_text())
    first = json.loads(Path(runs[0]['result']).read_text())
    reference = torch.load(first['checkpoint'], map_location='cpu', weights_only=True)
    identity = reference['feature_metadata']['nodes']
    while 'embedding' not in identity:
        identity = identity['base']

    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):
            return identity['embedding']['dim']

        def encode(self, *args, **kwargs):
            raise RuntimeError('Audit requires precomputed feature vectors')

    encoder = CacheOnly()
    nodes, edges, metadata = build_graph_features(
        encoder, reference['embedding_model_name'], cache_dir='data/feature_cache/graph',
        **reference['feature_metadata']['options'])
    frame = load_connections(Path(reference['dataset_path']))
    frame = frame[frame.date.isin(reference['split_dates']['val'])]
    boards = ConnectionsDataset.prepare(frame, encoder, None, nodes, edges)
    assert len(boards) == len(reference['split_dates']['val']) == 121
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    artifact = make_artifacts()[4]
    idx = artifact['idx'].to(device)
    positives = artifact['y'].nonzero().flatten()
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
                ranks = (scores.argsort(dim=1, descending=True).argsort(dim=1)+1)[:, positives].cpu().tolist()
                log_probs = scores.log_softmax(dim=1)[:, positives].cpu().tolist()
                for board, rank, logp in zip(batch, ranks, log_probs):
                    records.append(dict(variant=run['variant'], seed=run['seed'], date=str(board['date']),
                                        gold_ranks=rank, gold_log_probabilities=logp))
        del model
        torch.cuda.empty_cache()
        print('AUDITED', run['variant'], run['seed'], flush=True)
    (ROOT / 'probability-audit.json').write_text(json.dumps(dict(
        colors=['purple', 'blue', 'green', 'yellow'], floor=.025, records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
