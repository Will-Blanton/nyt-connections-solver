import unittest
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.graph_models import CandidateGINE, StandaloneCandidateGINE


class SparseGINETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_mask_removes_whole_message_and_preserves_self_term(self):
        model = CandidateGINE(3, 2, hidden=4, layers=2)
        words, edges = torch.randn(1, 4, 3), torch.rand(1, 4, 4, 2)
        empty = torch.zeros(1, 4, 4, dtype=torch.bool)
        actual = model(words, edges, empty)
        h = model.node_projection(words)
        for i, update in enumerate(model.updates):
            h = update((1 + model.eps[i]) * h)
        expected = torch.cat([h.mean(-2), h.amax(-2)], -1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(model.node_projection.weight.grad.abs().sum(), 0)
        self.assertEqual(model.edge_projections[0].weight.grad.abs().sum(), 0)

    def test_complete_mask_matches_legacy_and_permutation(self):
        model = CandidateGINE(3, 2, hidden=4, layers=2)
        words, edges = torch.randn(2, 4, 3), torch.rand(2, 4, 4, 2)
        complete = torch.ones(2, 4, 4, dtype=torch.bool)
        torch.testing.assert_close(model(words, edges), model(words, edges, complete), rtol=0, atol=0)
        mask = edges[..., 0] > .5
        perm = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(model(words, edges, mask),
            model(words[:, perm], edges[:, perm][:, :, perm], mask[:, perm][:, :, perm]))

    def test_threshold_union_and_checkpoint_config(self):
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1,
                      dropout=0., edge_hidden=8, train_text_projection=True,
                      train_input_norm=True, sparse_edge_thresholds=[.8, 0., 1.])
        idx = torch.tensor([[0, 1, 2, 3]])
        model = StandaloneCandidateGINE(group_idx=idx, **config).eval()
        edges = torch.zeros(1, 4, 4, 3)
        edges[0, 0, 1, 0] = .81
        edges[0, 0, 2, 1] = .1
        edges[0, 0, 3, 2] = 1.
        edges[0, 1, 2, 0] = .79
        keep = model.edge_connectivity(edges)
        self.assertEqual(keep.sum(), 3)
        self.assertFalse(keep.diagonal(dim1=-2, dim2=-1).any())
        x = torch.randn(1, 4, 8)
        reloaded = StandaloneCandidateGINE(group_idx=idx, **config).eval()
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x, edge_features=edges), reloaded(x, edge_features=edges))
        model(x, edge_features=edges).sum().backward()
        self.assertGreater(model.proj[0].weight.grad.abs().sum(), 0)
        for bad in ([.1], [-1, 0, 1], [float('nan'), 0, 1]):
            with self.assertRaises(ValueError):
                StandaloneCandidateGINE(group_idx=idx, **{**config, 'sparse_edge_thresholds': bad})
