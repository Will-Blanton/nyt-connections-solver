import unittest

import torch

from nyt_connections_solver.models import AttentionModel, Baseline, SetTransformer


class WordNetMixerTests(unittest.TestCase):
    def test_all_ltr_models_mix_semantic_and_wordnet_features(self):
        group_idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        features = torch.randn(2, 8, 24)
        common = {
            "in_dim": 24,
            "semantic_dim": 16,
            "wordnet_dim": 8,
            "group_idx": group_idx,
        }
        models = (
            Baseline(**common),
            AttentionModel(**common),
            SetTransformer(**common, lin_width=32, attn_width=16, attn_heads=4),
        )

        for model in models:
            with self.subTest(model=type(model).__name__):
                scores = model(features)
                self.assertEqual(tuple(scores.shape), (2, 2, 1))

    def test_mixer_requires_dimensions_to_match_input(self):
        with self.assertRaises(ValueError):
            Baseline(
                in_dim=24,
                semantic_dim=16,
                wordnet_dim=7,
                group_idx=torch.tensor([[0, 1, 2, 3]]),
            )


if __name__ == "__main__":
    unittest.main()
