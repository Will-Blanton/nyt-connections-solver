"""Objective identities and gradients, independent of any GNN implementation."""
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from graph_group_loss import per_group_contrastive_loss
from run_graph_feature_experiment import state_loss


class GroupContrastiveLossTests(unittest.TestCase):
    def test_manual_value_and_single_positive_equivalence(self):
        scores = torch.tensor([[1., 2., 3., 4.]])
        labels = torch.tensor([True, True, False, False])
        negative_mass = math.exp(3) + math.exp(4)
        expected = sum(math.log1p(negative_mass / math.exp(v)) for v in (1, 2))/2
        self.assertAlmostEqual(per_group_contrastive_loss(scores, labels).item(), expected, places=6)
        labels = torch.tensor([True, False, False, False])
        torch.testing.assert_close(per_group_contrastive_loss(scores, labels), state_loss(scores[..., None], labels))

    def test_easy_positive_does_not_hide_missed_positive(self):
        labels = torch.tensor([True, True, False])
        logits = torch.tensor([[30., -10., 0.]], requires_grad=True)
        loss = per_group_contrastive_loss(logits, labels)
        loss.backward()
        self.assertGreater(loss.item(), 4.9)
        self.assertLess(logits.grad[0, 1].item(), -.49)
        self.assertGreater(logits.grad[0, 2].item(), .49)
        self.assertLess(state_loss(logits[..., None], labels).item(), 1e-5)

    def test_batched_masks_permutation_shift_and_extremes(self):
        scores = torch.tensor([[1000., -1000., 0., -2.], [0., 1., 2., 3.]], requires_grad=True)
        labels = torch.tensor([[True, True, False, False], [False, False, True, False]])
        loss = per_group_contrastive_loss(scores, labels)
        expected = sum(per_group_contrastive_loss(s[None], p) for s, p in zip(scores, labels))/2
        torch.testing.assert_close(loss, expected)
        permutation = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(loss, per_group_contrastive_loss(scores[:, permutation], labels[:, permutation]))
        torch.testing.assert_close(loss, per_group_contrastive_loss(scores + 50, labels))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue((scores.grad[labels] <= 0).all())
        self.assertTrue((scores.grad[~labels] >= 0).all())

    def test_no_negatives_and_invalid_labels(self):
        scores = torch.randn(2, 3, requires_grad=True)
        loss = per_group_contrastive_loss(scores, torch.ones(3, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))
        with self.assertRaises(ValueError):
            per_group_contrastive_loss(scores, torch.zeros(3, dtype=torch.bool))
