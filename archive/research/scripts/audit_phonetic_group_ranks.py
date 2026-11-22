"""Selected-checkpoint validation ranks by audited mechanism, using cached features."""
from collections import defaultdict
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
    parser.add_argument('--results', type=Path, default=Path('experiments/gine-phonetics'))
    parser.add_argument('--inventory', type=Path, default=Path('experiments/gine-phonetics/mechanism-inventory.json'))
    args = parser.parse_args()
    directory = args.results
    runs = json.loads((directory / 'runs.json').read_text())
    first = json.loads(Path(runs[0]['result']).read_text())
    reference = torch.load(first['checkpoint'], map_location='cpu', weights_only=True)
    node_identity = reference['feature_metadata']['nodes']
    while 'base' in node_identity:
        node_identity = node_identity['base']

    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):
            return node_identity['embedding']['dim']

        def encode(self, *args, **kwargs):
            raise RuntimeError('Precompute experiment features before auditing group ranks')

    encoder = CacheOnly()
    nodes, edges, metadata = build_graph_features(
        encoder, reference['embedding_model_name'], cache_dir='data/feature_cache/graph',
        **reference['feature_metadata']['options'])
    frame = load_connections(Path(reference['dataset_path']))
    frame = frame[frame.date.isin(reference['split_dates']['val'])]
    boards = ConnectionsDataset.prepare(frame, encoder, None, nodes, edges)
    inventory = json.loads(args.inventory.read_text())
    labels = {(r['date'], r['color']): r for r in inventory['groups_inventory']}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    artifact = make_artifacts()[4]
    idx = artifact['idx'].to(device)
    targets = artifact['y'].nonzero().flatten()
    records = []
    fusion_diagnostics = []
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
                inputs = torch.stack([b['x'] for b in batch]).to(device)
                if hasattr(model, 'fusion_parts'):
                    semantic, projected, gate = model.fusion_parts(inputs)
                    _, known = model.encode_phonetics(inputs)
                    ratios = (gate * projected).norm(dim=-1) / semantic.norm(dim=-1).clamp_min(1e-8)
                    for board, gates, valid, values in zip(batch, gate.squeeze(-1), known.squeeze(-1), ratios):
                        fusion_diagnostics.append(dict(variant=run['variant'], seed=run['seed'],
                            date=str(board['date']), known_words=int(valid.sum()),
                            mean_gate=float(gates[valid].mean()) if valid.any() else 0.,
                            contribution_norm_ratio=float(values[valid].mean()) if valid.any() else 0.))
                logits = model(inputs, group_idx=idx,
                               edge_features=torch.stack([b['edge_features'] for b in batch]).to(device)).squeeze(-1)
                ranks = (logits.argsort(dim=1, descending=True).argsort(dim=1) + 1)[:, targets].cpu().tolist()
                for board, group_ranks in zip(batch, ranks):
                    for color, rank in zip(('purple', 'blue', 'green', 'yellow'), group_ranks):
                        label = labels[(str(board['date']), color)]
                        records.append(dict(variant=run['variant'], seed=run['seed'], date=str(board['date']),
                                            color=color, category=label['category'], mechanism=label['mechanism'], rank=rank))
        del model
        torch.cuda.empty_cache()
        print('AUDITED', run['variant'], run['seed'], flush=True)
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['variant'], row['mechanism'])].append(row['rank'])
    aggregate = {variant: {mechanism: dict(appearances=len(ranks), top1=mean(r == 1 for r in ranks),
                                         top4=mean(r <= 4 for r in ranks), mrr=mean(1/r for r in ranks))
                           for (v, mechanism), ranks in grouped.items() if v == variant}
                 for variant in sorted({r['variant'] for r in records})}
    (directory / 'group-ranks.json').write_text(json.dumps(dict(
        aggregate=aggregate, records=records, fusion_diagnostics=fusion_diagnostics), indent=2) + '\n')


if __name__ == '__main__':
    main()
