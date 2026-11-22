from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from graph_training_schedule import annealed_encoder_lr, cosine_to_zero


class EncoderAnnealingTests(unittest.TestCase):
    def test_cosine_schedule_endpoints_and_monotonicity(self):
        values = [cosine_to_zero(step, .001, 101) for step in range(1, 102)]
        self.assertEqual(values[0], .001)
        self.assertEqual(values[-1], 0.)
        self.assertAlmostEqual(values[50], .0005)
        self.assertTrue(all(a >= b >= 0 for a, b in zip(values, values[1:])))
        self.assertEqual(cosine_to_zero(120, .001, 101), 0.)
    def test_zero_lr_preserves_weights_despite_momentum_and_weight_decay(self):
        encoder = torch.nn.Parameter(torch.tensor([1.]))
        head = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([{'params': [head]}, {'params': [encoder]}], lr=.1, weight_decay=.1)
        before_freeze = None
        for step in range(1, 7):
            optimizer.param_groups[1]['lr'] = annealed_encoder_lr(step, .1, 3)
            optimizer.zero_grad()
            (encoder + head).square().sum().backward()
            old_head = head.detach().clone()
            optimizer.step()
            if step == 2:
                before_freeze = encoder.detach().clone()
            if step >= 3:
                torch.testing.assert_close(encoder, before_freeze, rtol=0, atol=0)
                self.assertFalse(torch.equal(head, old_head))
        self.assertEqual(annealed_encoder_lr(1, .1, 3), .1)
        self.assertEqual(annealed_encoder_lr(3, .1, 3), 0.)
