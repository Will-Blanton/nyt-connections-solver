"""Floor behavior, numerical stability, and preserved mass objective."""
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from graph_group_loss import probability_mass_with_floor
from run_graph_feature_experiment import state_loss


class ProbabilityFloorTests(unittest.TestCase):
    def test_zero_weight_and_already_satisfied_floor_preserve_mass(self):
        logits = torch.tensor([[3., 1., -2.]])
        labels = torch.tensor([True, True, False])
        expected = state_loss(logits[..., None], labels)
        torch.testing.assert_close(probability_mass_with_floor(logits, labels, weight=0), expected, rtol=0, atol=0)
        torch.testing.assert_close(probability_mass_with_floor(logits, labels, weight=.1), expected, rtol=0, atol=0)

    def test_only_correct_candidates_below_floor_are_penalized(self):
        probabilities = torch.tensor([[.97, .001, .029]])
        logits = probabilities.log().requires_grad_()
        labels = torch.tensor([True, True, False])
        difference = probability_mass_with_floor(logits, labels, weight=.1) - state_loss(logits[..., None], labels)
        self.assertAlmostEqual(difference.item(), .1 * math.log(.05/.001)/2, places=6)
        difference.backward()
        self.assertLess(logits.grad[0, 1].item(), -.049)
        # The unrelated incorrect candidate is not itself assigned a probability floor.
        self.assertGreater(logits.grad[0, 2].item(), 0)

    def test_variable_positive_counts_shift_permutation_and_extremes(self):
        logits = torch.tensor([[1000., -1000., 0., 1.], [1., 2., 3., 4.]], requires_grad=True)
        labels = torch.tensor([[True, True, False, False], [True, True, True, False]])
        value = probability_mass_with_floor(logits, labels)
        expected = sum(probability_mass_with_floor(s[None], p) for s, p in zip(logits, labels))/2
        torch.testing.assert_close(value, expected)
        perm = torch.tensor([3, 0, 2, 1])
        torch.testing.assert_close(value, probability_mass_with_floor(logits[:, perm]+50, labels[:, perm]))
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_settings_and_labels_rejected(self):
        logits, labels = torch.zeros(1, 3), torch.tensor([True, False, False])
        for kwargs in ({'beta': 0}, {'beta': 1}, {'weight': -1}, {'weight': float('nan')}):
            with self.assertRaises(ValueError):
                probability_mass_with_floor(logits, labels, **kwargs)
        with self.assertRaises(ValueError):
            probability_mass_with_floor(logits, torch.zeros(3, dtype=torch.bool))
