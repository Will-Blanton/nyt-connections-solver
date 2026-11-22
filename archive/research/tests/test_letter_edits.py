from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import sys
from unittest.mock import patch
from math import log

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.edge_features import EnglishLexicon, LetterEditSimilarity, EdgeFeatureComposer
from nyt_connections_solver.feature_cache import TensorCache


class LetterEditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def provider(self, vocabulary):
        self.embedded = []
        def encode(words):
            self.embedded.extend(words)
            return torch.ones(len(words), 4)
        lexicon = EnglishLexicon(entries=vocabulary)
        embeddings = SimpleNamespace(identity={'test_embedding': 1}, encode=encode)
        return LetterEditSimilarity(lexicon, embeddings)

    def semantic(self, provider, words, operation, alignment):
        channel = provider.feature_names.index(f'{operation}_{alignment}_semantic')
        return provider.encode(words)[..., channel]

    def test_substitution_position_alignment_and_unattested_source(self):
        p = self.provider(['apple', 'date', 'lemon', 'peach', 'five', 'six', 'four', 'eight'])
        words = ['ample', 'dote', 'lumon', 'poach']
        self.assertIn('lemon', p.readings('lumon')[('substitution', 'internal_from_start', 1)])
        values = self.semantic(p, words, 'substitution', 'internal_from_start')
        self.assertTrue(torch.all(values[torch.triu_indices(4, 4, 1).unbind()] == 1))
        self.assertEqual(self.semantic(p, words, 'substitution', 'first').count_nonzero(), 0)
        self.assertEqual(self.semantic(p, ['hive', 'mix', 'pour', 'wight'], 'substitution', 'first')[0, 1], 1)
        p = self.provider(['apple', 'lemon'])
        self.assertEqual(self.semantic(p, ['apxle', 'lomon'], 'substitution', 'anywhere')[0, 1], 1)
        self.assertEqual(self.semantic(p, ['apxle', 'lomon'], 'substitution', 'internal_from_start')[0, 1], 0)
        self.assertEqual(self.semantic(p, ['apxle', 'lomon'], 'substitution', 'internal_from_end')[0, 1], 0)

    def test_deletion_and_mixed_edit_positions(self):
        p = self.provider(['comet', 'nova', 'star', 'sun', 'order', 'class', 'family', 'genus'])
        self.assertEqual(self.semantic(p, ['cometh', 'novak', 'start', 'sung'], 'deletion', 'last')[0, 1], 1)
        self.assertEqual(self.semantic(p, ['foamily', 'genius'], 'deletion', 'anywhere')[0, 1], 1)
        self.assertEqual(self.semantic(p, ['foamily', 'genius'], 'deletion', 'internal_from_start')[0, 1], 0)
        self.assertEqual(self.semantic(p, ['border', 'classy'], 'deletion', 'anywhere')[0, 1], 1)
        self.assertEqual(self.semantic(p, ['border', 'classy'], 'deletion', 'first')[0, 1], 0)

    def test_no_cross_operation_or_identical_target_shortcuts(self):
        p = self.provider(['apple', 'lemon'])
        self.assertEqual(p.encode(['bpple', 'lemonx']).count_nonzero(), 0)
        p = self.provider(['win'])
        self.assertEqual(p.encode(['wind', 'wine', 'wing', 'wink']).count_nonzero(), 0)

    def test_specificity_counts_distinct_reading_pairs(self):
        p = self.provider(['cat', 'bat', 'hat', 'dog', 'log', 'fog'])
        edges = p.encode(['mat', 'bog'])
        channel = p.feature_names.index('substitution_first_reading_specificity')
        self.assertAlmostEqual(edges[0, 1, channel].item(), 1 / (1 + log(9)))
        self.assertEqual(edges[0, 1, channel-1], 1)

    def test_bounds_permutation_and_family_cache(self):
        p = self.provider(['apple', 'date', 'lemon', 'peach'])
        words = ['ample', 'dote', 'lumon', 'poach', 'two words', '!!!']
        with tempfile.TemporaryDirectory() as directory:
            composer = EdgeFeatureComposer([p], cache=TensorCache(Path(directory)))
            edges = composer.encode(words)
            calls = len(self.embedded)
            torch.testing.assert_close(composer.encode(words), edges)
            self.assertEqual(len(self.embedded), calls)
        self.assertEqual(edges.shape, (6, 6, 20))
        self.assertTrue(torch.isfinite(edges).all())
        self.assertGreaterEqual(edges.min(), 0)
        self.assertLessEqual(edges.max(), 1)
        self.assertEqual(edges[4:].count_nonzero(), 0)
        self.assertEqual(edges[torch.arange(6), torch.arange(6)].count_nonzero(), 0)
        torch.testing.assert_close(edges, edges.transpose(0, 1))
        order = [3, 1, 0, 2, 5, 4]
        torch.testing.assert_close(p.encode([words[i] for i in order]), edges[order][:, order])
        changed = LetterEditSimilarity(p.lexicon, p.embeddings, min_length=2)
        self.assertNotEqual(changed.cache_identity, p.cache_identity)
        self.assertEqual(p.readings('a' * 33), {})
        self.assertEqual(p.readings('café'), {})

    def test_feature_builder_preserves_old_metadata_and_reconstructs_new_order(self):
        from nyt_connections_solver.graph_features import build_graph_features
        class Encoder(torch.nn.Module):
            def get_embedding_dimension(self):
                return 4
            def encode(self, words, **kwargs):
                return torch.ones(len(words), 4)
        encoder = Encoder()
        lexicon = EnglishLexicon(entries=['apple', 'date'])
        options = dict(revision='test-revision', include_wordnet=False, use_spelling=False)
        with patch('nyt_connections_solver.graph_features.EnglishLexicon', return_value=lexicon):
            _, old_edges, old = build_graph_features(encoder, 'test-model', **options)
            _, disabled, unchanged = build_graph_features(encoder, 'test-model',
                                                          include_letter_edits=False, **options)
            self.assertEqual(old, unchanged)
            self.assertEqual(old_edges.feature_names, disabled.feature_names)
            _, enabled, metadata = build_graph_features(encoder, 'test-model',
                                                        include_letter_edits=True, **options)
            self.assertEqual(enabled.feature_names[:old_edges.dim], old_edges.feature_names)
            self.assertEqual(enabled.feature_names[old_edges.dim:], LetterEditSimilarity.feature_names)
            _, rebuilt, restored = build_graph_features(encoder, 'test-model', **metadata['options'])
            self.assertEqual(metadata, restored)
            self.assertEqual(enabled.feature_names, rebuilt.feature_names)
            torch.testing.assert_close(enabled.encode(['ample', 'dote']), rebuilt.encode(['ample', 'dote']))


if __name__ == '__main__':
    unittest.main()
