"""The residual must preserve initialization, symmetry, and frozen base weights."""
import itertools
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyt_connections_solver.graph_models import GraphSetTransformer, GatedEdgeResidual


class GatedEdgeTests(unittest.TestCase):
    def test_zero_edge_control_ignores_all_edges(self):
        torch.set_num_threads(1)
        model = GatedEdgeResidual(group_idx=torch.tensor([[0, 1, 2, 3]]),
                                 in_dim=8, edge_dim=3, edge_bias_scale=0.,
                                 attn_width=8, attn_heads=2, attn_layers=1,
                                 lin_width=16, lin_layers=1, dropout=0.,
                                 residual_edge_channels=[]).eval()
        with torch.no_grad():
            model.edge_correction.weight.fill_(.1)
        x = torch.randn(1, 4, 8)
        torch.testing.assert_close(model(x, edge_features=torch.zeros(1, 4, 4, 3)),
                                   model(x, edge_features=torch.rand(1, 4, 4, 3)),
                                   rtol=0, atol=0)

    def test_masked_channels_cannot_affect_correction(self):
        torch.set_num_threads(1)
        idx = torch.tensor([[0, 1, 2, 3]])
        model = GatedEdgeResidual(group_idx=idx, in_dim=8, edge_dim=3,
                                 edge_bias_scale=0., attn_width=8, attn_heads=2,
                                 attn_layers=1, lin_width=16, lin_layers=1, dropout=0.,
                                 residual_edge_channels=[0]).eval()
        with torch.no_grad():
            model.edge_correction.weight.fill_(.1)
        x = torch.randn(1, 4, 8)
        edges = torch.rand(1, 4, 4, 3)
        changed = edges.clone()
        changed[..., 1:] += 100
        torch.testing.assert_close(model(x, edge_features=edges),
                                   model(x, edge_features=changed), rtol=0, atol=0)

    def test_zero_start_frozen_base_and_candidate_permutation(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        idx = torch.tensor(list(itertools.combinations(range(8), 4)))
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        base = GraphSetTransformer(group_idx=idx, **config).eval()
        model = GatedEdgeResidual(group_idx=idx, **config)
        model.load_state_dict(base.state_dict(), strict=False)
        x = torch.randn(2, 8, 8)
        edges = torch.rand(2, 8, 8, 3)
        edges = (edges + edges.transpose(1, 2)) / 2
        expected = base(x, edge_features=edges).detach()
        torch.testing.assert_close(model(x, edge_features=edges), expected)
        original = {k: v.clone() for k, v in base.state_dict().items()}
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            scores = model(x, edge_features=edges)
            (-scores[:, 0].mean() + scores[:, 1].mean()).backward()
            optimizer.step()
        self.assertGreater(model.edge_encoder[0].weight.grad.abs().sum().item(), 0)
        for key, value in original.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        model.eval()
        torch.testing.assert_close(model(x, edge_features=edges),
                                   model(x, group_idx=idx.flip(1), edge_features=edges),
                                   rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.equal(model(x, edge_features=edges), expected))


if __name__ == "__main__":
    unittest.main()
