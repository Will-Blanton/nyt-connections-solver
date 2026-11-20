import sys
from pathlib import Path
import unittest
from collections import Counter
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_gine_curriculum import make_schedule
import run_graph_feature_experiment as experiment


class CurriculumTests(unittest.TestCase):
    def test_equal_exposure_and_stages(self):
        for seed in (10, 11, 12):
            mixed = make_schedule('mixed', seed)
            curriculum = make_schedule('curriculum', seed)
            self.assertEqual(Counter(mixed), Counter(curriculum))
            self.assertEqual(Counter(curriculum), {2: 5000, 3: 5000, 4: 5000})
            self.assertEqual(curriculum[:2500], [2] * 2500)
            self.assertEqual(curriculum[2500:5000], [3] * 2500)
            for i in range(5000, 15000, 4):
                self.assertEqual(Counter(curriculum[i:i+4]), {2: 1, 3: 1, 4: 2})
            for i in range(0, 15000, 3):
                self.assertEqual(set(mixed[i:i+3]), {2, 3, 4})
            self.assertEqual(curriculum, make_schedule('curriculum', seed))
        self.assertNotEqual(make_schedule('curriculum', 10), make_schedule('curriculum', 11))

    def test_invalid_schedule_configuration(self):
        for variant, budget in [('bad', 15000), ('mixed', 5), ('curriculum', 0)]:
            with self.assertRaises(ValueError):
                make_schedule(variant, 10, budget)

    def test_training_consumes_requested_schedule(self):
        torch.set_num_threads(1)
        schedule = make_schedule('curriculum', 10, 12)
        observed = []
        model = torch.nn.Linear(1, 2)
        loaders = {size: [None] for size in (2, 3, 4)}
        artifacts = {size: {'idx': torch.tensor(size), 'y': torch.tensor([True, False])}
                     for size in loaders}
        metrics = {size: dict.fromkeys(('hit1', 'mrr', 'hit4', 'map', 'ndcg4'), .5)
                   for size in loaders}

        def scores(model, batch, idx, device):
            observed.append(idx.item())
            return model(torch.ones(1, 1))

        with patch.object(experiment, 'score_batch', scores), patch.object(
                experiment, 'evaluate', return_value=metrics):
            _, summary = experiment.train(
                model, loaders, loaders, artifacts, torch.device('cpu'),
                lr=.001, projection_lr=None, weight_decay=0., updates=12,
                eval_every=6, min_updates_before_stop=12, patience=100,
                min_delta=0., state_schedule=schedule)
        self.assertEqual(observed, schedule)
        self.assertEqual(summary['updates_ran'], 12)
