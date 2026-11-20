from collections import Counter
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fragment_states import FragmentStates
import run_graph_feature_experiment as experiment
from nyt_connections_solver.data_loader import collate_keep_wordlists


class FragmentStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.board = dict(x=torch.arange(32).reshape(16, 2).float(),
                          edge_features=torch.arange(16 * 16).reshape(16, 16, 1).float(),
                          word_list=[f'w{i}' for i in range(16)], date='train-date')

    def test_labels_composition_slicing_and_balance(self):
        for size in (2, 3):
            for fragments in (False, True):
                dataset = FragmentStates([self.board], size, 10, fragments=fragments)
                targets = Counter()
                for sample in dataset:
                    p = sample['original_positions']
                    targets[sample['target_group']] += 1
                    self.assertEqual(len(set(p.tolist())), size * 4)
                    counts = sorted(Counter((p // 4).tolist()).values())
                    self.assertEqual(counts, ([1, 1, 2, 4] if size == 2 else [2, 3, 3, 4])
                                     if fragments else [4] * size)
                    self.assertEqual(sample['relevance'].sum().item(), 1 if fragments else size)
                    for candidate, label in zip(dataset.candidates, sample['relevance']):
                        original = p[candidate].tolist()
                        official = any(set(original) == set(range(4*g, 4*g+4)) for g in range(4))
                        self.assertEqual(bool(label), official)
                    torch.testing.assert_close(sample['x'], self.board['x'][p])
                    torch.testing.assert_close(sample['edge_features'], self.board['edge_features'][p][:, p])
                    self.assertEqual(sample['word_list'], [f'w{i}' for i in p.tolist()])
                self.assertEqual(targets, dict.fromkeys(range(4), 3))

    def test_reproducibility_and_target_positions_match_control(self):
        for size in (2, 3):
            treatment = FragmentStates([self.board], size, 10, fragments=True)
            control = FragmentStates([self.board], size, 10, fragments=False)
            replica = FragmentStates([self.board], size, 10, fragments=True)
            other_seed = FragmentStates([self.board], size, 11, fragments=True)
            for i in range(12):
                a, b = treatment[i], control[i]
                torch.testing.assert_close(a['original_positions'], replica[i]['original_positions'])
                target = a['target_group']
                torch.testing.assert_close(a['original_positions'] // 4 == target,
                                           b['original_positions'] // 4 == target)
            self.assertFalse(torch.equal(treatment[0]['original_positions'], other_seed[0]['original_positions']))

    def test_batched_loss_is_correct_and_trainer_uses_sample_labels(self):
        dataset = FragmentStates([self.board], 2, 10, fragments=True)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_keep_wordlists)
        batch = next(iter(loader))
        logits = torch.randn(4, 70, requires_grad=True)
        labels = batch['relevance']
        expected = torch.nn.functional.cross_entropy(logits, labels.long().argmax(dim=1))
        torch.testing.assert_close(experiment.state_loss(logits[..., None], labels), expected)
        expected.backward()
        self.assertGreater(logits.grad.abs().sum(), 0)
        old_labels = torch.arange(70) < 2
        torch.testing.assert_close(experiment.state_loss(logits[..., None], old_labels),
                                   experiment.state_loss(logits[..., None], old_labels.expand(4, -1)))
        with self.assertRaises(ValueError):
            experiment.state_loss(logits[..., None], torch.zeros_like(labels))
        model = torch.nn.Linear(2, 70)
        metrics = {2: dict.fromkeys(('hit1', 'mrr', 'hit4', 'map', 'ndcg4'), .5)}
        artifacts = {2: {'idx': dataset.candidates, 'y': torch.zeros(70, dtype=torch.bool)}}
        seen = []
        original_loss = experiment.state_loss

        def loss(scores, relevance):
            seen.append(relevance.clone())
            return original_loss(scores, relevance)

        with patch.object(experiment, 'score_batch', lambda model, batch, idx, device:
                          model(batch['x'].mean(dim=1))[..., None]), patch.object(
                          experiment, 'evaluate', return_value=metrics), patch.object(
                          experiment, 'state_loss', side_effect=loss):
            # The shared loop reports all three validation state sizes.
            metrics[3] = metrics[2]
            metrics[4] = metrics[2]
            experiment.train(model, {2: loader}, {2: loader}, artifacts, torch.device('cpu'),
                             lr=.001, projection_lr=None, weight_decay=0., updates=1,
                             eval_every=1, min_updates_before_stop=1, patience=10,
                             min_delta=0., state_schedule=[2])
        torch.testing.assert_close(seen[0], labels)
