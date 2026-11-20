import math
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from graph_auxiliary_loss import puzzle_contrastive_loss


class AuxiliaryLossTests(unittest.TestCase):
    def test_uniform_similarity_has_no_self_positive(self):
        for n in (8, 12, 16):
            loss = puzzle_contrastive_loss(torch.ones(2, n, 4))
            self.assertAlmostEqual(loss.item(), math.log(n - 1), places=5)

    def test_group_geometry_and_puzzle_locality(self):
        words = torch.eye(4).repeat_interleave(4, dim=0).unsqueeze(0)
        coherent = puzzle_contrastive_loss(words)
        shuffled = puzzle_contrastive_loss(words[:, torch.arange(16).reshape(4, 4).T.flatten()])
        self.assertLess(coherent.item(), shuffled.item())
        other = torch.randn_like(words)
        together = puzzle_contrastive_loss(torch.cat([words, other]))
        torch.testing.assert_close(together, (coherent + puzzle_contrastive_loss(other)) / 2)

    def test_gradients_reach_projection(self):
        projection = torch.nn.Linear(8, 16)
        loss = puzzle_contrastive_loss(projection(torch.randn(2, 16, 8)))
        loss.backward()
        self.assertTrue(torch.isfinite(projection.weight.grad).all())
        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
