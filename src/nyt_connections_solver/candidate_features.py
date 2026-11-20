"""Small original-board candidate tables, reusable across subgraphs and retries."""
from functools import lru_cache
from itertools import combinations
import torch


@lru_cache(maxsize=8)
def _candidate_lookup(size):
    return {positions: row for row, positions in enumerate(combinations(range(size), 4))}


def slice_candidate_features(values, positions, *, original_size=16, group_idx=None):
    """Gather original quartet rows in the residual state's local candidate order.

    positions may be permuted; feature values are invariant to word order.
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().tolist()
    if len(set(positions)) != len(positions) or any(i < 0 or i >= original_size for i in positions):
        raise ValueError('Expected distinct original-board word positions')
    local = combinations(range(len(positions)),4) if group_idx is None else group_idx.detach().cpu().tolist()
    lookup = _candidate_lookup(original_size)
    rows = [lookup[tuple(sorted(positions[i] for i in quartet))] for quartet in local]
    return values[torch.tensor(rows, device=values.device, dtype=torch.long)]


def candidate_edge_kwargs(batch, device):
    """Optional model arguments shared by graph training/evaluation loops."""
    if 'candidate_edge_features' not in batch:
        return {}
    return {'candidate_edge_features': batch['candidate_edge_features'].to(device)}
