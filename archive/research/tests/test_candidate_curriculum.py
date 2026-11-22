import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from collections import Counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_gine_candidate_curriculum import make_candidate_schedule, make_training_artifacts
import run_graph_feature_experiment as experiment
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


class CandidateCurriculumTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31415)

    def test_schedule_and_labels(self):
        schedule = make_candidate_schedule('pairs_triples_quads')
        self.assertEqual(Counter(schedule), {2: 3000, 3: 3000, 4: 9000})
        self.assertEqual(schedule[:3000], [2] * 3000)
        self.assertEqual(schedule[3000:6000], [3] * 3000)
        self.assertEqual(schedule[6000:], [4] * 9000)
        self.assertEqual(make_candidate_schedule('quad_only'), [4] * 15000)
        artifacts = make_training_artifacts()
        for k, count, positives in [(2, 120, 24), (3, 560, 16), (4, 1820, 4)]:
            idx, labels = artifacts[(4, k)]['idx'], artifacts[(4, k)]['y']
            self.assertEqual(idx.shape, (count, k))
            self.assertEqual(labels.sum().item(), positives)
            for group, label in zip(idx.tolist(), labels.tolist()):
                self.assertEqual(len({i // 4 for i in group}) == 1, label)

    def test_gine_pairs_triples_quads_gradients_and_quad_compatibility(self):
        artifacts = make_training_artifacts()
        model = StandaloneCandidateGINE(group_idx=artifacts[(4, 4)]['idx'], in_dim=8,
            edge_dim=3, attn_width=8, attn_heads=2, attn_layers=1, lin_width=16,
            lin_layers=1, dropout=0., edge_hidden=8, train_text_projection=True,
            train_input_norm=True).eval()
        x, edges = torch.randn(2, 16, 8), torch.randn(2, 16, 16, 3)
        for k in (2, 3, 4):
            artifact = artifacts[(4, k)]
            idx = artifact['idx']
            model.zero_grad()
            score = model(x, group_idx=idx, edge_features=edges)
            self.assertEqual(score.shape, (2, len(idx), 1))
            loss = experiment.state_loss(score, artifact['y'])
            loss.backward()
            for p in (model.proj[0].weight, model.input_norm.weight,
                      model.edge_encoder.edge_projections[0].weight, model.group_readout[0].weight):
                self.assertGreater(p.grad.abs().sum().item(), 0)
            if k == 4:
                # Exact pre-change four-word computation; parameter layout is unchanged.
                words = model.encode_input_words(x)
                candidates = model.grouper(words, group_idx=idx)
                candidate_edges = edges[:, idx[:, :, None], idx[:, None, :], :]
                pooled = model.edge_encoder(candidates.reshape(-1, 4, 8),
                    (candidate_edges * model.residual_edge_mask).reshape(-1, 4, 4, 3))
                legacy = model.group_readout(pooled).reshape(2, len(idx), 1)
                torch.testing.assert_close(score, legacy, rtol=0, atol=0)
                clone = StandaloneCandidateGINE(group_idx=idx, in_dim=8, edge_dim=3,
                    attn_width=8, attn_heads=2, attn_layers=1, lin_width=16,
                    lin_layers=1, dropout=0., edge_hidden=8, train_text_projection=True,
                    train_input_norm=True).eval()
                clone.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(score, clone(x, edge_features=edges))

    def test_trainer_changes_only_training_candidate_indices(self):
        artifacts = experiment.make_artifacts()
        training = make_training_artifacts()
        model = torch.nn.Linear(1, 1)
        observed = []
        metrics = {s: dict.fromkeys(('hit1', 'mrr', 'hit4', 'map', 'ndcg4'), .5)
                   for s in (2, 3, 4)}

        def score(model, batch, idx, device):
            observed.append(idx.shape[1])
            return model.weight.expand(2, len(idx))[..., None]

        def evaluate(model, loaders, supplied, device, **kwargs):
            self.assertIs(supplied, artifacts)
            self.assertTrue(all(a['idx'].shape[1] == 4 for a in supplied.values()))
            return metrics

        with patch.object(experiment, 'score_batch', score), patch.object(experiment, 'evaluate', evaluate):
            _, summary = experiment.train(model, {4: [None]}, {}, artifacts, torch.device('cpu'),
                lr=.001, projection_lr=None, weight_decay=0., updates=3, eval_every=1,
                min_updates_before_stop=3, patience=100, min_delta=0.,
                state_schedule=[4, 4, 4], candidate_schedule=[2, 3, 4], training_artifacts=training)
        self.assertEqual(observed, [2, 3, 4])
        self.assertEqual(summary['updates_ran'], 3)
