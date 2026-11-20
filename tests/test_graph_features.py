"""Baseline graph features, persistent caching, and subset alignment."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.feature_cache import TensorCache
from nyt_connections_solver.edge_features import (
    CharacterOverlap, Anagrams, SpellingStructure, PhraseCompletions,
    SubstringSimilarity, EnglishLexicon, EmbeddingStore, EdgeFeatureComposer,
    default_edge_providers,
)
from nyt_connections_solver.node_features import TextSpellingFeatures
from nyt_connections_solver.data_loader import make_state_loaders, GraphConnectionsDataset
from nyt_connections_solver.graph_features import build_graph_features


class FakeEmbedder:
    def __init__(self): self.calls = []
    def get_sentence_embedding_dimension(self): return 8
    def eval(self): return self
    def requires_grad_(self, value): return self
    def to(self, device): return self
    def encode(self, words, **kwargs):
        self.calls.extend(words)
        return torch.tensor([list(hashlib.sha256(w.encode()).digest()[:8]) for w in words], dtype=torch.float32) / 255


def frame(words):
    return pd.DataFrame({'date': ['2026-01-01'] * 4, 'connections': [words[i:i+4] for i in range(0, 16, 4)]})




class GraphFeatureTests(unittest.TestCase):
    def test_historical_spelling_examples(self):
        self.assertAlmostEqual(CharacterOverlap().encode(['bough', 'cough'])[0, 1, 0].item(), .8)
        self.assertEqual(CharacterOverlap().encode(['a', 'cat'])[0, 1, 0], 0)
        self.assertEqual(Anagrams().encode(['least', 'tesla'])[0, 1, 0], 1)
        self.assertEqual(Anagrams().encode(['beer hat', 'breathe'])[0, 1, 0], 1)
        self.assertEqual(Anagrams().encode(['!!!', '???'])[0, 1, 0], 0)
        for words, channel in [(['kayak', 'race car'], 0), (['aardvark', 'eerie'], 1), (['balloon', 'coffee'], 2)]:
            self.assertEqual(SpellingStructure().encode(words)[0, 1, channel], 1)
        self.assertEqual(SpellingStructure().encode(['dog', 'cat']).sum(), 0)

    def test_phrases_and_fragments_without_categories(self):
        lexicon = EnglishLexicon(['prime_number', 'prime_time', 'alarm_clock', 'biological_clock',
                                 'ball', 'base', 'bat', 'cap', 'bus', 'car', 'train', 'ship'])
        phrase = PhraseCompletions(lexicon)
        self.assertEqual(phrase.encode(['number', 'time'])[0, 1, 0], 1)
        self.assertEqual(phrase.encode(['alarm', 'biological'])[0, 1, 1], 1)
        self.assertEqual(phrase.encode(['unknown', 'absent']).sum(), 0)
        model = FakeEmbedder()
        store = EmbeddingStore(model, 'fake', revision='1')
        provider = SubstringSimilarity(lexicon, store)
        words = ['ballroom', 'basement', 'batmobile', 'capstone', 'incubus', 'oscar', 'quatrain', 'situationship']
        values = provider.encode(words)
        self.assertGreater(values[0, 1, 0], 0)
        self.assertGreater(values[4, 5, 1], 0)
        self.assertEqual(set(model.calls), {'ball', 'base', 'bat', 'cap', 'bus', 'car', 'train', 'ship'})
        provider.encode(words)
        self.assertEqual(len(model.calls), 8)

    def test_disk_cache_reuse_versions_resources_and_provider_order(self):
        class Counting(CharacterOverlap):
            calls = 0
            def encode(self, words):
                self.calls += 1
                return super().encode(words)
        with tempfile.TemporaryDirectory() as directory:
            a = Counting()
            result = EdgeFeatureComposer([a], cache=TensorCache(directory)).encode(['bough', 'cough'])
            b = Counting()
            reused = EdgeFeatureComposer([b, Anagrams()], cache=TensorCache(directory)).encode(['bough', 'cough'])
            self.assertEqual(b.calls, 0)
            torch.testing.assert_close(reused[..., :3], result)
            b.cache_identity = dict(b.cache_identity, version=2)
            EdgeFeatureComposer([b], cache=TensorCache(directory)).encode(['bough', 'cough'])
            self.assertEqual(b.calls, 1)
            EdgeFeatureComposer([b], cache=TensorCache(directory, refresh=True)).encode(['bough', 'cough'])
            self.assertEqual(b.calls, 2)
            lex1, lex2 = EnglishLexicon(['prime_time']), EnglishLexicon(['prime_time', 'prime_number'])
            first = EdgeFeatureComposer([PhraseCompletions(lex1)], cache=TensorCache(directory)).encode(['time', 'number'])
            second = EdgeFeatureComposer([PhraseCompletions(lex2)], cache=TensorCache(directory)).encode(['time', 'number'])
            self.assertEqual(first[0, 1, 0], 0)
            self.assertEqual(second[0, 1, 0], 1)
            model = FakeEmbedder()
            EmbeddingStore(model, 'fake', revision='1', cache=TensorCache(directory)).encode(['word'])
            other = FakeEmbedder()
            EmbeddingStore(other, 'fake', revision='1', cache=TensorCache(directory)).encode(['word'])
            self.assertEqual(other.calls, [])
            EmbeddingStore(other, 'fake', revision='2', cache=TensorCache(directory)).encode(['word'])
            self.assertEqual(other.calls, ['word'])

    def test_real_wordnet_bounds_permutation_and_partial_states(self):
        words = ['ballroom', 'basement', 'batmobile', 'capstone', 'least', 'slate', 'stale', 'tesla',
                 'number', 'time', 'suspect', 'minister', 'kayak', 'level', 'mom', 'race car']
        model = FakeEmbedder()
        nodes, edges, _ = build_graph_features(model, 'fake', revision='1', char_dim=16)
        full_x, full_e = nodes.encode(words), edges.encode(words)
        perm = torch.randperm(16)
        torch.testing.assert_close(nodes.encode([words[i] for i in perm]), full_x[perm])
        torch.testing.assert_close(edges.encode([words[i] for i in perm]), full_e[perm][:, perm])
        self.assertEqual(full_e.shape, (16, 16, 15))
        datasets, loaders = make_state_loaders(frame(words), model, None, 2,
                                               feature_builder=nodes, edge_providers=edges)
        self.assertEqual([len(datasets[s]) for s in (2, 3, 4)], [6, 4, 1])
        for size, dataset in datasets.items():
            self.assertIsInstance(dataset, GraphConnectionsDataset)
            for sample in dataset:
                positions = [words.index(w) for w in sample['word_list']]
                torch.testing.assert_close(sample['x'], full_x[positions])
                torch.testing.assert_close(sample['edge_features'], full_e[positions][:, positions])
            self.assertEqual(next(iter(loaders[size]))['x'].shape[1], 4 * size)



if __name__ == '__main__':
    unittest.main()
