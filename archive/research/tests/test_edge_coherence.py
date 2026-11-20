import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


class EdgeCoherenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                           attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1,
                           dropout=0., edge_hidden=8, train_text_projection=True,
                           train_input_norm=True)
        self.idx = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])

    def model(self, scale=0.):
        return StandaloneCandidateGINE(group_idx=self.idx, edge_coherence_scale=scale,
                                       **self.config).eval()

    def test_coherent_channel_doubles_but_outsider_does_not(self):
        edges = torch.zeros(2, 4, 4, 3)
        edges[..., 0] = 1  # all four support this channel
        edges[..., :3, :3, 1] = 1  # only three support this one
        edges[..., 2] = .5
        edges.diagonal(dim1=-3, dim2=-2).fill_(99)  # self edges must not count
        original = edges.clone()
        modified = self.model(1).modify_candidate_edges(edges)
        offdiag = ~torch.eye(4, dtype=torch.bool)
        torch.testing.assert_close(modified[..., 0][:, offdiag], torch.full((2, 12), 2.))
        torch.testing.assert_close(modified[..., 1][:, offdiag], edges[..., 1][:, offdiag])
        torch.testing.assert_close(modified[..., 2][:, offdiag], torch.full((2, 12), .75))
        self.assertFalse(modified.diagonal(dim1=-3, dim2=-2).any())
        torch.testing.assert_close(edges, original, rtol=0, atol=0)

    def test_zero_scale_is_exact_legacy_and_initialization_is_unchanged(self):
        torch.manual_seed(3)
        legacy = StandaloneCandidateGINE(group_idx=self.idx, **self.config).eval()
        torch.manual_seed(3)
        explicit = self.model(0)
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        edges = torch.rand(2, 5, 5, 3)
        x = torch.randn(2, 5, 8)
        self.assertIs(explicit.modify_candidate_edges(edges), edges)
        torch.testing.assert_close(legacy(x, edge_features=edges),
                                   explicit(x, edge_features=edges), rtol=0, atol=0)

    def test_candidate_locality_permutation_reload_and_gradients(self):
        model = self.model(1)
        x = torch.randn(2, 5, 8)
        raw = torch.rand(2, 5, 5, 3)
        edges = ((raw + raw.transpose(1, 2)) / 2).requires_grad_()
        scores = model(x, edge_features=edges)
        for i, idx in enumerate(self.idx):
            isolated = model(x[:, idx], group_idx=torch.tensor([[0, 1, 2, 3]]),
                             edge_features=edges[:, idx][:, :, idx])
            torch.testing.assert_close(scores[:, i:i+1], isolated)
        perm = torch.tensor([2, 0, 3, 1])
        quartet = edges[:, :4, :4]
        torch.testing.assert_close(model.modify_candidate_edges(quartet)[:, perm][:, :, perm],
                                   model.modify_candidate_edges(quartet[:, perm][:, :, perm]))
        reloaded = self.model(1)
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(scores, reloaded(x, edge_features=edges))
        scores.square().sum().backward()
        self.assertGreater(edges.grad.abs().sum(), 0)
        self.assertGreater(model.proj[0].weight.grad.abs().sum(), 0)
        for scale in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.model(scale)


if __name__ == '__main__':
    unittest.main()
