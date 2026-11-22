"""Exercise local rarity and puzzle context without downloading an encoder or corpus."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyt_connections_solver.wordnet import WordNetFeatureEncoder


class Sense:
    def __init__(self, name, depth=1, parents=()):
        self.label, self.depth, self.parents = name, depth, list(parents)

    def name(self): return self.label
    def max_depth(self): return self.depth
    def hypernyms(self): return self.parents
    def instance_hypernyms(self): return []
    def lemma_names(self): return [self.label]
    def definition(self): return self.label


class Embedder:
    def get_sentence_embedding_dimension(self): return 2

    def encode(self, texts, **kwargs):
        return np.array([[len(t), sum(map(ord, t))] for t in texts], dtype=np.float32)


def puzzle(*words):
    return list(words) + [f"unknown{i}" for i in range(16 - len(words))]


class ConditionedWordNetTests(unittest.TestCase):
    def encoder(self, mapping, max_senses=1):
        with patch.object(WordNetFeatureEncoder, "_load_wordnet", return_value=None):
            encoder = WordNetFeatureEncoder(
                Embedder(), max_senses=max_senses, puzzle_conditioned=True,
            )
        encoder._senses = lambda word: tuple(mapping.get(word, ()))
        return encoder

    def test_rare_shared_concept_beats_common_and_unshared_senses(self):
        common, rare, unique = Sense("common", 20), Sense("rare", 3), Sense("unique", 100)
        target = Sense("target", 4, [rare])
        encoder = self.encoder({"a": [unique, common, target], "b": [rare], "c": [common], "d": [common]})
        self.assertEqual(encoder.select_senses(puzzle("a", "b", "c", "d"))["a"], (target,))

    def test_support_counts_words_once_and_breaks_ties_by_depth(self):
        shallow, deep = Sense("shallow", 2), Sense("deep", 5)
        encoder = self.encoder({
            "a": [shallow, deep], "b": [shallow],
            "c": [Sense("c1", 6, [deep]), Sense("c2", 6, [deep])],
        })
        self.assertEqual(encoder.select_senses(puzzle("a", "b", "c"))["a"], (deep,))

    def test_fallback_and_legacy_selection(self):
        first, second = Sense("first"), Sense("second", 9)
        encoder = self.encoder({"a": [first, second], "b": [second]})
        self.assertEqual(encoder.select_senses(puzzle("a"))["a"], (first,))
        encoder.puzzle_conditioned = False
        self.assertEqual(encoder.select_senses(["a", "b"])["a"], (first,))

    def test_cache_is_context_sensitive_and_order_invariant(self):
        first, second = Sense("first"), Sense("second")
        encoder = self.encoder({"a": [first, second], "b": [first], "c": [second]})
        words = puzzle("a", "b")
        original = encoder.encode(words)
        changed = encoder.encode(puzzle("a", "c"))
        self.assertFalse(np.array_equal(original[0], changed[0]))
        np.testing.assert_array_equal(encoder.encode(words[::-1]), original[::-1])
        np.testing.assert_array_equal(original[2:], np.zeros((14, 2)))
        with self.assertRaises(ValueError):
            encoder.encode(words[:12])

    def test_notebook_partial_states_use_original_puzzle_features(self):
        from nyt_connections_solver.data_loader import ConnectionsDataset
        words = [f"w{i}" for i in range(16)]
        calls = []

        class Features:
            def encode(self, received, **kwargs):
                calls.append(received)
                return torch.arange(16, dtype=torch.float32).unsqueeze(1)

        class Vectorizer:
            def transform(self, received):
                return type("Matrix", (), {"toarray": lambda self: np.zeros((len(received), 1))})()

        frame = pd.DataFrame({"date": ["2026-01-01"] * 4, "connections": [words[i:i+4] for i in range(0, 16, 4)]})
        for size in (2, 3, 4):
            data = ConnectionsDataset(frame, None, Vectorizer(), num_sets_remaining=size, feature_builder=Features())
            for state in data.puzzle_list:
                expected = torch.tensor([words.index(w) for w in state["word_list"]], dtype=torch.float32)
                torch.testing.assert_close(state["x"].squeeze(1), expected)
        self.assertEqual(calls, [words] * 3)


if __name__ == "__main__":
    unittest.main()
