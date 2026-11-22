"""Check color attribution on partial states with nonconsecutive source groups."""
import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "feature_experiment", Path(__file__).resolve().parents[1] / "scripts/run_graph_feature_experiment.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ColorMetricsTests(unittest.TestCase):
    def test_partial_state_uses_original_source_color_and_exact_group_rank(self):
        artifacts = runner.make_artifacts()
        positives = artifacts[2]["y"].nonzero().flatten()
        logits = torch.zeros(1, len(artifacts[2]["idx"]), 1)
        logits[0, positives[0], 0] = 10
        logits[0, positives[1], 0] = 9

        class FixedModel(torch.nn.Module):
            def forward(self, x, **kwargs):
                return logits

        batch = {"x": torch.zeros(1, 8, 1), "edge_features": torch.zeros(1, 8, 8, 1),
                 "source_group_idx": torch.tensor([[1, 3]])}
        metrics = runner.evaluate(FixedModel(), {2: [batch]}, artifacts,
                                  torch.device("cpu"), by_color=True)[2]
        self.assertEqual(metrics["hit1"], 1)
        self.assertEqual(set(metrics["by_color"]), {"blue", "yellow"})
        self.assertEqual(metrics["by_color"]["blue"]["top1"], 1)
        self.assertEqual(metrics["by_color"]["yellow"]["top1"], 0)
        self.assertEqual(metrics["by_color"]["yellow"]["mean_rank"], 2)
        self.assertEqual(metrics["by_color"]["yellow"]["mrr"], .5)
        self.assertEqual(metrics["by_color"]["yellow"]["top4"], 1)


if __name__ == "__main__":
    unittest.main()
