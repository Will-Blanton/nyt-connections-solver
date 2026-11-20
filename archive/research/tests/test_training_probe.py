"""Measurements must preserve training and distinguish blocked gradients."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from graph_training_probe import TrainingProbe
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


class ProbeTests(unittest.TestCase):
    def test_probe_preserves_optimizer_step_and_reports_frozen_parameters(self):
        torch.set_num_threads(1)
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        idx = torch.tensor([[0, 1, 2, 3]])
        model = StandaloneCandidateGINE(group_idx=idx, **config)
        reference = StandaloneCandidateGINE(group_idx=idx, **config)
        reference.load_state_dict(model.state_dict())
        probe = TrainingProbe(model)
        x, edges = torch.randn(2, 4, 8), torch.rand(2, 4, 4, 3)
        for current in (model, reference):
            optimizer = torch.optim.AdamW((p for p in current.parameters() if p.requires_grad), lr=.001)
            if current is model:
                probe.begin(1, 4)
            loss = current(x, edge_features=edges).sum()
            loss.backward()
            if current is model:
                probe.before_step()
            optimizer.step()
            if current is model:
                probe.after_step(loss)
        probe.close()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
        row = probe.records[0]['parameters']
        self.assertFalse(row['proj']['trainable'])
        self.assertEqual(row['proj']['update_to_weight'], 0)
        self.assertGreater(row['group_readout']['gradient_norm'], 0)
        self.assertGreater(row['group_readout']['update_to_weight'], 0)
