"""Frozen-stage invariants and score semantics for residual/ensemble checkpoints."""
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nyt_connections_solver.additive_models import AdditiveGINE
from nyt_connections_solver.checkpoints import GRAPH_MODEL_TYPES


class AdditiveGINETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.idx = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5]])
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.,
                      edge_hidden=8, gine_layers=1, readout_width=8, gine_dropout=.2,
                      train_text_projection=True, train_input_norm=True)
        self.config = dict(base_config=config, correction_config=config, correction_scale=.25)
        self.config.update(in_dim=8, edge_dim=3)
        self.x, self.edges = torch.randn(2, 6, 8), torch.rand(2, 6, 6, 3)

    def test_zero_start_frozen_base_and_correction_gradients(self):
        model = AdditiveGINE(group_idx=self.idx, **self.config)
        torch.nn.init.zeros_(model.correction.group_readout[-1].weight)
        torch.nn.init.zeros_(model.correction.group_readout[-1].bias)
        snapshot = {k: v.clone() for k, v in model.base.state_dict().items()}
        model.train()
        self.assertFalse(model.base.training)
        self.assertTrue(model.correction.training)
        baseline = model.base(self.x, edge_features=self.edges)
        torch.testing.assert_close(model(self.x, edge_features=self.edges), baseline, rtol=0, atol=0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            scores = model(self.x, edge_features=self.edges)
            loss = -scores.log_softmax(dim=1)[:, 0].mean()
            loss.backward()
            optimizer.step()
        for name, value in model.base.state_dict().items():
            self.assertTrue(torch.equal(snapshot[name], value))
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertGreater(model.correction.proj[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.correction.input_norm.weight.grad.abs().sum().item(), 0)
        model.eval()
        clone = GRAPH_MODEL_TYPES['AdditiveGINE'](group_idx=self.idx, **self.config).eval()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(self.x, edge_features=self.edges), clone(self.x, edge_features=self.edges))

    def test_ensemble_is_frozen_mean_and_candidate_order_invariant(self):
        model = AdditiveGINE(group_idx=self.idx, **{**self.config, 'base_scale': .5,
                                                  'correction_scale': .5, 'train_correction': False})
        model.train()
        self.assertFalse(model.base.training or model.correction.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        expected = (model.base(self.x, edge_features=self.edges) +
                    model.correction(self.x, edge_features=self.edges)) / 2
        torch.testing.assert_close(model(self.x, edge_features=self.edges), expected)
        torch.testing.assert_close(model(self.x, group_idx=self.idx[:, [2, 0, 3, 1]], edge_features=self.edges), expected)

    def test_mismatched_component_features_are_rejected(self):
        wrong = {**self.config['correction_config'], 'edge_dim': 4}
        with self.assertRaisesRegex(ValueError, 'edge_dim'):
            AdditiveGINE(group_idx=self.idx, **{**self.config, 'correction_config': wrong})
