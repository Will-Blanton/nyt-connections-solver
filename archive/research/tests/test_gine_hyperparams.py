import sys
from pathlib import Path
import unittest
import torch
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tune_gine import build_model, BASELINE
from nyt_connections_solver.graph_models import GraphSetTransformer, StandaloneCandidateGINE


class GINEHyperparameterTests(unittest.TestCase):
    def test_tuned_dimensions_dropout_gradients_and_reload(self):
        torch.set_num_threads(1)
        idx = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5]])
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        base = GraphSetTransformer(group_idx=idx, **config)
        reference = dict(model_config=config, model_state_dict=base.state_dict())
        params = {**BASELINE, 'hidden_width': 64, 'gine_layers': 3, 'readout_width': 48, 'dropout': .2}
        model, saved_config = build_model(reference, params, idx, 1)
        self.assertEqual(model.edge_encoder.node_projection.out_features, 64)
        self.assertEqual(len(model.edge_encoder.updates), 3)
        self.assertEqual(model.group_readout[0].out_features, 48)
        self.assertEqual(sum(isinstance(m, torch.nn.Dropout) for m in model.edge_encoder.modules()), 6)
        self.assertTrue(model.proj[0].weight.requires_grad and model.input_norm.weight.requires_grad)
        model.train()
        x, edges = torch.randn(2, 6, 8), torch.rand(2, 6, 6, 3)
        scores = model(x, edge_features=edges)
        (scores[:, 0] - scores[:, 1]).sum().backward()
        self.assertGreater(model.proj[0].weight.grad.abs().sum().item(), 0)
        model.eval()
        reloaded = StandaloneCandidateGINE(group_idx=idx, **saved_config).eval()
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x, edge_features=edges), reloaded(x, edge_features=edges))

    def test_median_pruning_does_not_cut_short_runs(self):
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(
            n_startup_trials=6, n_warmup_steps=8000, interval_steps=1000, n_min_trials=4))
        for _ in range(6):
            trial = study.ask()
            for step in range(250, 15001, 250):
                trial.report(.3, step)
            study.tell(trial, .3)
        trial = study.ask()
        for step in range(250, 8000, 250):
            trial.report(.1, step)
            self.assertFalse(trial.should_prune())
        trial.report(.1, 8000)
        self.assertTrue(trial.should_prune())
