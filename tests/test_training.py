"""Production GINE training, split boundaries, and checkpoint reconstruction."""
import itertools
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nyt_connections_solver.training import (
    build_model, candidate_artifacts, mixed_state_schedule,
    probability_mass_with_floor, train_fixed, validate_splits,
)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checked_in_recipe_and_split_boundaries(self):
        config = json.loads((ROOT / 'config/gine-baseline.json').read_text())
        validate_splits(config['split_dates'])
        self.assertEqual(len(config['split_dates']['train']), 1089)
        self.assertEqual(len(config['split_dates']['test']), 121)
        self.assertFalse(config['split_dates']['val'])
        self.assertEqual(config['training']['updates'], 1878)
        self.assertEqual(len(config['edge_feature_names']), 15)
        self.assertFalse(config['feature_options']['use_spelling'])
        self.assertEqual(config['model_config']['edge_coherence_scale'], 0.)
        for splits in ({'train': [], 'val': [], 'test': []},
                       {'train': ['a'], 'val': ['a'], 'test': []},
                       {'train': ['a', 'a'], 'val': [], 'test': []}):
            with self.assertRaises(ValueError):
                validate_splits(splits)

    def test_state_sampling_is_balanced_from_start_and_labels_are_exact(self):
        schedule = mixed_state_schedule(10, 18)
        self.assertEqual(schedule, mixed_state_schedule(10, 18))
        for start in range(0, 18, 3):
            self.assertEqual(set(schedule[start:start+3]), {2, 3, 4})
        for size, artifact in candidate_artifacts().items():
            self.assertEqual(len(artifact['idx']), len(list(itertools.combinations(range(size*4), 4))))
            self.assertEqual(artifact['y'].sum(), size)
            for group in artifact['idx'][artifact['y']]:
                self.assertTrue((group // 4 == group[0] // 4).all())

    def test_trainable_projection_gine_readout_and_strict_reload(self):
        config = json.loads((ROOT / 'config/gine-baseline.json').read_text())['model_config']
        config = {**config, 'in_dim': 8}
        artifacts = candidate_artifacts()
        model = build_model(config, artifacts[4]['idx'], 10)
        x = torch.randn(1, 8, 8)
        edges = torch.rand(1, 8, 8, 15)
        edges = (edges + edges.transpose(1, 2)) / 2
        idx = artifacts[2]['idx']
        scores = model(x, group_idx=idx, edge_features=edges)
        probability_mass_with_floor(scores, artifacts[2]['y']).backward()
        for module in (model.input_norm, model.proj, model.edge_encoder, model.group_readout):
            self.assertTrue(all(p.requires_grad for p in module.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        loaded = build_model(config, artifacts[4]['idx'], 99)
        loaded.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(loaded(x, group_idx=idx, edge_features=edges), scores)
        perm = torch.tensor([7, 1, 4, 3, 0, 5, 2, 6])
        inverse = perm.argsort()
        reordered = loaded(x[:, perm], group_idx=inverse[idx], edge_features=edges[:, perm][:, :, perm])
        torch.testing.assert_close(reordered, scores, rtol=1e-5, atol=1e-6)

    def test_fixed_training_updates_all_active_layers_without_heldout_inputs(self):
        config = json.loads((ROOT / 'config/gine-baseline.json').read_text())['model_config']
        config = {**config, 'in_dim': 8}
        artifacts = candidate_artifacts()
        model = build_model(config, artifacts[4]['idx'], 10)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        loaders = {size: DataLoader([{'x': torch.randn(size*4, 8),
                    'edge_features': torch.rand(size*4, size*4, 15)}], batch_size=1)
                   for size in (2, 3, 4)}
        callbacks = []
        model, summary = train_fixed(model, loaders, artifacts, 'cpu', seed=10,
            updates=6, lr=.001, weight_decay=0, log_every=3, callback=callbacks.append)
        self.assertEqual(summary['state_updates'], {2: 2, 3: 2, 4: 2})
        self.assertEqual([r['update'] for r in callbacks], [3, 6])
        self.assertEqual(summary['updates_ran'], 6)
        self.assertFalse(model.training)
        after = model.state_dict()
        for prefix in ('input_norm.', 'proj.', 'edge_encoder.', 'group_readout.'):
            self.assertTrue(any(not torch.equal(v, after[k]) for k, v in before.items() if k.startswith(prefix)))
        for key, value in before.items():
            if key.startswith(('global_encoder.', 'group_encoder.', 'decoder.')):
                torch.testing.assert_close(after[key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
