"""Small synthetic checks; no pretrained embeddings or puzzle data required."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyt_connections_solver.partition import (
    PartitionTransformer, decode_partition, evaluate_partitions, partition_loss,
)


class PartitionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        self.targets = torch.arange(4).repeat_interleave(4).unsqueeze(0)

    def test_loss_ignores_gold_group_names(self):
        logits = torch.randn(2, 16, 4, requires_grad=True)
        targets = self.targets.expand(2, -1)
        relabeled = torch.tensor([2, 0, 3, 1])[targets]
        torch.testing.assert_close(
            partition_loss(logits, targets), partition_loss(logits, relabeled),
        )
        partition_loss(logits, targets).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_capacity_and_known_optimum(self):
        # All words prefer the same slot; independent argmax would be invalid.
        logits = torch.randn(3, 16, 4)
        logits[:, :, 0] += 100
        predicted = decode_partition(logits)
        for row in predicted:
            self.assertEqual(torch.bincount(row, minlength=4).tolist(), [4] * 4)
        expected = torch.tensor([2, 0, 3, 1])[self.targets]
        clear_logits = torch.full((1, 16, 4), -10.0)
        clear_logits.scatter_(-1, expected.unsqueeze(-1), 10.0)
        torch.testing.assert_close(decode_partition(clear_logits), expected)
        self.assertLess(partition_loss(clear_logits, self.targets).item(), 1e-6)

    def test_model_gradients_and_word_permutation(self):
        model = PartitionTransformer(8, width=16, heads=4, layers=1, scorer_width=16, dropout=0)
        x = torch.randn(2, 16, 8)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 16, 4))
        loss = partition_loss(logits, self.targets.expand(2, -1))
        loss.backward()
        grad = model.decoder.seed_vectors.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        model.eval()
        order = torch.randperm(16)
        torch.testing.assert_close(model(x[:, order]), model(x)[:, order], atol=1e-6, rtol=1e-5)

    def test_partition_metrics_ignore_slot_names(self):
        predicted = torch.tensor([2, 0, 3, 1])[self.targets].expand(2, -1).clone()
        # One puzzle perfect; the other swaps two words, breaking two groups.
        predicted[1, 0], predicted[1, 4] = predicted[1, 4].item(), predicted[1, 0].item()
        logits = torch.full((2, 16, 4), -10.0)
        logits.scatter_(-1, predicted.unsqueeze(-1), 10.0)
        loader = [
            {"x": logits[:1], "targets": self.targets},
            {"x": logits[1:], "targets": self.targets},
        ]
        metrics = evaluate_partitions(torch.nn.Identity(), loader, torch.device("cpu"))
        self.assertEqual(metrics["solve_rate"], 0.5)
        self.assertEqual(metrics["mean_correct_groups"], 3)
        self.assertEqual(metrics["matched_word_accuracy"], 30 / 32)


if __name__ == "__main__":
    unittest.main()
