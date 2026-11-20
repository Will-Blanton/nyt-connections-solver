"""GINE formula, frozen-base behavior, permutation invariance, and reload."""
import itertools
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyt_connections_solver.graph_models import CandidateGINE, GatedGINEResidual, GraphSetTransformer, StandaloneCandidateGINE


class GINETests(unittest.TestCase):
    def test_single_layer_adapter_formula_gradient_and_reload(self):
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.,
                      single_layer_adapter=True, train_input_norm=True)
        idx = torch.tensor([[0, 1, 2, 3]])
        model = StandaloneCandidateGINE(group_idx=idx, **config)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.embedding_adapter), 1)
        x, edges = torch.randn(2, 4, 8), torch.rand(2, 4, 4, 3)
        captured = []
        hook = model.edge_encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        scores = model(x, edge_features=edges)
        hook.remove()
        z = model.input_norm(x)
        expected = z + torch.relu(model.embedding_adapter[0](z))
        torch.testing.assert_close(captured[0], expected)
        scores.sum().backward()
        self.assertGreater(model.embedding_adapter[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.input_norm.weight.grad.abs().sum().item(), 0)
        reloaded = StandaloneCandidateGINE(group_idx=idx, **config)
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(scores, reloaded(x, edge_features=edges))

    def test_adapter_identity_start_gradient_and_reload(self):
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        idx = torch.tensor([[0, 1, 2, 3]])
        torch.manual_seed(9)
        control = StandaloneCandidateGINE(group_idx=idx, adapter_hidden=0, **config)
        torch.manual_seed(9)
        model = StandaloneCandidateGINE(group_idx=idx, adapter_hidden=4, **config)
        x, edges = torch.randn(2, 4, 8), torch.rand(2, 4, 4, 3)
        torch.testing.assert_close(control(x, edge_features=edges), model(x, edge_features=edges))
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            model(x, edge_features=edges).sum().backward()
            optimizer.step()
        self.assertGreater(model.embedding_adapter[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.input_norm.weight.grad.abs().sum().item(), 0)
        reloaded = StandaloneCandidateGINE(group_idx=idx, adapter_hidden=4, **config)
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x, edge_features=edges), reloaded(x, edge_features=edges))

    def test_standalone_projection_can_train_without_unfreezing_attention(self):
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        idx = torch.tensor([[0, 1, 2, 3]])
        frozen = StandaloneCandidateGINE(group_idx=idx, **config)
        learned = StandaloneCandidateGINE(group_idx=idx, train_text_projection=True, **config)
        learned.load_state_dict(frozen.state_dict())
        x = torch.randn(2, 4, 8)
        edges = torch.rand(2, 4, 4, 3)
        torch.testing.assert_close(frozen(x, edge_features=edges), learned(x, edge_features=edges))
        self.assertFalse(frozen.proj[0].weight.requires_grad)
        self.assertTrue(learned.proj[0].weight.requires_grad)
        self.assertFalse(learned.input_norm.weight.requires_grad)
        self.assertFalse(learned.full_encoder.mabs[0].attn.in_proj_weight.requires_grad)
        learned(x, edge_features=edges).sum().backward()
        self.assertGreater(learned.proj[0].weight.grad.abs().sum().item(), 0)

    def test_standalone_raw_path_learns_direct_embedding_projection(self):
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        model = StandaloneCandidateGINE(group_idx=torch.tensor([[0, 1, 2, 3]]),
                                        raw_text_embeddings=True, **config)
        x, edges = torch.randn(2, 4, 8), torch.rand(2, 4, 4, 3)
        model(x, edge_features=edges).sum().backward()
        self.assertEqual(model.edge_encoder.node_projection.in_features, 8)
        self.assertGreater(model.edge_encoder.node_projection.weight.grad.abs().sum().item(), 0)
        self.assertFalse(model.proj[0].weight.requires_grad)
        self.assertIsNone(model.proj[0].weight.grad)

    def test_standalone_narrow_projection_reload_and_gradient(self):
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.,
                      projection_width=4, train_text_projection=True, train_input_norm=True)
        idx = torch.tensor([[0, 1, 2, 3]])
        model = StandaloneCandidateGINE(group_idx=idx, **config)
        x, edges = torch.randn(2, 4, 8), torch.rand(2, 4, 4, 3)
        scores = model(x, edge_features=edges)
        scores.sum().backward()
        self.assertEqual(model.edge_encoder.node_projection.in_features, 4)
        self.assertGreater(model.proj[0].weight.grad.abs().sum().item(), 0)
        reloaded = StandaloneCandidateGINE(group_idx=idx, **config)
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(scores, reloaded(x, edge_features=edges))

    def test_freezing_random_projection_does_not_change_initial_tensors(self):
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        idx = torch.tensor([[0, 1, 2, 3]])
        torch.manual_seed(12)
        frozen = StandaloneCandidateGINE(group_idx=idx, **config)
        torch.manual_seed(12)
        trainable = StandaloneCandidateGINE(group_idx=idx, train_text_projection=True, **config)
        torch.testing.assert_close(frozen.input_norm.weight, trainable.input_norm.weight)
        torch.testing.assert_close(frozen.proj[0].weight, trainable.proj[0].weight)
        torch.testing.assert_close(frozen.proj[0].bias, trainable.proj[0].bias)

    def test_input_stages_preserve_frozen_score_and_capture_expected_words(self):
        torch.set_num_threads(1)
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        idx = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 5]])
        model = GatedGINEResidual(group_idx=idx, **config).eval()
        x, edges = torch.randn(1, 8, 8), torch.rand(1, 8, 8, 3)
        words, _, stages = model.encode_candidate_words(x, edge_features=edges, return_stages=True)
        projected = model.proj(model.input_norm(x))
        torch.testing.assert_close(stages["projected"], model.grouper(projected))
        torch.testing.assert_close(stages["global"], model.grouper(model.full_encoder(projected, edges)))
        torch.testing.assert_close(stages["candidate"], words)
        expected = model(x, edge_features=edges)
        for stage in ("projected", "global", "candidate"):
            model.gine_input_stage = stage
            torch.testing.assert_close(model(x, edge_features=edges), expected, rtol=0, atol=0)
            captured = []
            hook = model.edge_encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
            model(x, edge_features=edges)
            hook.remove()
            torch.testing.assert_close(captured[0], stages[stage])

    def test_zero_edge_control_retains_nodes_but_ignores_edges(self):
        torch.set_num_threads(1)
        model = GatedGINEResidual(group_idx=torch.tensor([[0, 1, 2, 3]]),
                                 in_dim=8, edge_dim=3, edge_bias_scale=0.,
                                 attn_width=8, attn_heads=2, attn_layers=1,
                                 lin_width=16, lin_layers=1, dropout=0.,
                                 residual_edge_channels=[]).eval()
        with torch.no_grad():
            model.edge_correction.weight.fill_(.1)
        x = torch.randn(1, 4, 8)
        expected = model(x, edge_features=torch.zeros(1, 4, 4, 3))
        torch.testing.assert_close(expected, model(x, edge_features=torch.rand(1, 4, 4, 3)),
                                   rtol=0, atol=0)
        self.assertFalse(torch.equal(expected,
                                    model(x + torch.randn_like(x), edge_features=torch.zeros(1, 4, 4, 3))))

    def test_message_formula_excludes_diagonal(self):
        network = CandidateGINE(1, 1, hidden=1, layers=1)
        network.node_projection = nn.Identity()
        network.edge_projections[0] = nn.Identity()
        network.updates[0] = nn.Identity()
        with torch.no_grad():
            network.eps.fill_(.5)
        words = torch.tensor([[[1.], [2.], [3.], [4.]]])
        edges = torch.ones(1, 4, 4, 1)
        edges[:, torch.arange(4), torch.arange(4)] = 1000
        # Every node receives sum(other h_j + 1), then adds 1.5 * h_i.
        expected_h = 13 + .5 * words
        expected = torch.cat([expected_h.mean(1), expected_h.amax(1)], -1)
        torch.testing.assert_close(network(words, edges), expected)

    def test_zero_start_gradients_frozen_base_permutation_and_reload(self):
        for nonlinear in (False, True):
            with self.subTest(nonlinear_edges=nonlinear):
                self.check_branch(nonlinear)

    def check_branch(self, nonlinear):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        idx = torch.tensor(list(itertools.combinations(range(8), 4)))
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        base = GraphSetTransformer(group_idx=idx, **config).eval()
        original = {key: value.clone() for key, value in base.state_dict().items()}
        model = GatedGINEResidual(group_idx=idx, nonlinear_edges=nonlinear, **config)
        model.load_state_dict(original, strict=False)
        x = torch.randn(2, 8, 8)
        edges = torch.rand(2, 8, 8, 3)
        edges = (edges + edges.transpose(1, 2)) / 2
        torch.testing.assert_close(model(x, edge_features=edges), base(x, edge_features=edges))
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            scores = model(x, edge_features=edges)
            (-scores[:, 0].mean() + scores[:, 1].mean()).backward()
            optimizer.step()
        projection = model.edge_encoder.edge_projections[0]
        if nonlinear:
            projection = projection[0]
        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        for key, value in original.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        model.eval()
        torch.testing.assert_close(model(x, edge_features=edges),
                                   model(x, group_idx=idx.flip(1), edge_features=edges),
                                   rtol=1e-5, atol=1e-6)
        reloaded = GatedGINEResidual(group_idx=idx, nonlinear_edges=nonlinear, **config).eval()
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x, edge_features=edges), reloaded(x, edge_features=edges))

    def test_nonlinear_zero_edge_encoder_has_no_constant_evidence(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        linear = CandidateGINE(8, 3)
        torch.manual_seed(3)
        nonlinear = CandidateGINE(8, 3, nonlinear_edges=True)
        words = torch.randn(2, 4, 8)
        zero_edges = torch.zeros(2, 4, 4, 3)
        torch.testing.assert_close(linear(words, zero_edges), nonlinear(words, zero_edges),
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
