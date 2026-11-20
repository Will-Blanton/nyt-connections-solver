"""Direct quartet-edge summaries preserve relation identity and candidate locality."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from nyt_connections_solver.graph_models import StandaloneCandidateGINE
from run_gine_relation_witnesses import fresh_expanded_model


class EdgeSummaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                           attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        self.params = dict(hidden_width=8, gine_layers=2, readout_width=8, dropout=0.)
        self.idx = torch.tensor([[0, 1, 2, 3]])

    def build(self, width=0):
        return fresh_expanded_model(self.config, self.params, self.idx, 10, 5,
                                    range(4), (3, 4), coherence_scale=0., edge_summary_width=width)

    def test_statistics_distinguish_clique_and_star_and_ignore_diagonal(self):
        e = torch.zeros(1, 4, 4, 3)
        e[..., 0] = .8
        e[:, :3, :3, 1] = 1
        e[:, 0, :, 2] = 1
        e[:, :, 0, 2] = 1
        e.diagonal(dim1=-3, dim2=-2).fill_(99)
        before = e.clone()
        summary = StandaloneCandidateGINE.summarize_candidate_edges(e)
        torch.testing.assert_close(summary, torch.tensor([[.8, .5, .5, .8, 0., 0., .8, 0., 1/3]]))
        perm = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(summary, StandaloneCandidateGINE.summarize_candidate_edges(e[:, perm][:, :, perm]))
        torch.testing.assert_close(e, before, rtol=0, atol=0)

    def test_initial_scores_match_control_and_branch_learns_after_zero_start(self):
        base, _ = self.build()
        model, _ = self.build(4)
        for name, value in base.state_dict().items():
            actual = model.state_dict()[name]
            if name == 'group_readout.0.weight':
                torch.testing.assert_close(actual[:, :value.shape[-1]], value, rtol=0, atol=0)
                self.assertFalse(actual[:, value.shape[-1]:].any())
            else:
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
        x = torch.randn(2, 4, 8)
        e = torch.rand(2, 4, 4, 5)
        c = torch.rand(2, 1, 2)
        inputs = dict(edge_features=e, candidate_edge_features=c)
        torch.testing.assert_close(base(x, **inputs), model(x, **inputs), rtol=0, atol=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        model(x, **inputs).sum().backward()
        self.assertGreater(model.group_readout[0].weight.grad[:, -4:].abs().sum(), 0)
        optimizer.step()
        optimizer.zero_grad()
        model(x, **inputs).sum().backward()
        self.assertGreater(sum(p.grad.abs().sum() for p in model.edge_summary_encoder.parameters()), 0)

    def test_witness_overrides_masks_locality_and_strict_reload(self):
        model, config = self.build(4)
        x = torch.randn(2, 5, 8)
        e = torch.rand(2, 5, 5, 5)
        c = torch.tensor([[[.7, .9]], [[.4, .8]]])
        captured = []
        hook = model.edge_summary_encoder.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
        scores = model(x, edge_features=e, candidate_edge_features=c)
        hook.remove()
        stats = captured[0].reshape(2, 3, 5)
        torch.testing.assert_close(stats[:, :, 3], c[:, 0, 0, None].expand(-1, 3))
        self.assertFalse(stats[:, :, 4].any())
        # Enable the branch after initialization to test its locality, not just the base GINE.
        with torch.no_grad():
            model.group_readout[0].weight[:, -4:].fill_(.1)
        x2, e2 = x.clone(), e.clone()
        x2[:, 4] = 999
        e2[:, 4] = 999
        e2[:, :, 4] = 999
        expected = model(x, edge_features=e, candidate_edge_features=c)
        torch.testing.assert_close(expected, model(x2, edge_features=e2, candidate_edge_features=c))
        reloaded = StandaloneCandidateGINE(group_idx=self.idx, **config)
        reloaded.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(expected, reloaded(x, edge_features=e, candidate_edge_features=c))


if __name__ == '__main__':
    unittest.main()
