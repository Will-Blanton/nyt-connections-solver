import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.edge_features import PronunciationLexicon, PronunciationSimilarity, EdgeFeatureComposer
from nyt_connections_solver.edge_features.pronunciation import phone_similarity
from nyt_connections_solver.edge_features.resources import EnglishLexicon


class PronunciationTests(unittest.TestCase):
    def setUp(self):
        self.sounds = PronunciationLexicon({
            'sea': [['S', 'IY1']], 'see': [['S', 'IY1']], 'tea': [['T', 'IY1']],
            'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']], 'red': [['R', 'EH1', 'D']],
            'base': [['B', 'EY1', 'S']], 'bass': [['B', 'EY1', 'S'], ['B', 'AE1', 'S']],
            'loot': [['L', 'UW1', 'T']], 'lute': [['L', 'UW1', 'T']],
            'outersea': [['AW1', 'T', 'ER0', 'S', 'IY0']],
            'outersee': [['AW1', 'T', 'ER0', 'S', 'IY0']],
        })
        self.lexicon = EnglishLexicon(entries=['sea', 'see', 'tea', 'outer', 'out'])
        self.embeddings = SimpleNamespace(identity={'fake': 1}, encode=lambda words: torch.ones(len(words), 4))
        self.provider = PronunciationSimilarity(self.lexicon, self.embeddings, pronunciations=self.sounds)

    def test_alternate_pronunciations_homophones_rhyme_unknown(self):
        edges = EdgeFeatureComposer([self.provider]).encode(['sea', 'see', 'tea', 'read', 'red', 'absent'])
        self.assertEqual(edges[0, 1, 0], 1)
        self.assertEqual(edges[3, 4, 0], 1)
        self.assertEqual(edges[0, 2, 2], 1)
        self.assertEqual(edges[0, 2, 0], 0)
        self.assertEqual(edges[0, 2, 1], .5)
        self.assertEqual(torch.count_nonzero(edges[-1]), 0)
        self.assertEqual(phone_similarity(('AA', 'B'), ('AA', 'B')), 1)

    def test_fragment_sound_uses_valid_proper_fragments_and_confidence(self):
        edges = self.provider.encode(['outersea', 'outersee'])
        self.assertEqual(self.provider.fragments('outersea', suffix=True), ('sea',))
        self.assertAlmostEqual(edges[0, 1, 5].item(), .75**2)
        self.assertEqual(self.provider.fragments('sea'), ())
        self.assertNotIn('outersea', self.provider.fragments('outersea'))

    def test_homophone_alternative_semantics_are_not_direct_word_similarity(self):
        edges = self.provider.encode(['base', 'loot', 'tea'])
        self.assertEqual(self.sounds.alternatives('base'), ('bass',))
        self.assertEqual(edges[0, 1, 0], 0)
        self.assertEqual(edges[0, 1, 3], 1)
        self.assertEqual(edges[0, 2, 3], 0)

    def test_multiword_composition_and_order_invariance(self):
        self.assertEqual(self.sounds.phones('sea tea'), (('S', 'IY', 'T', 'IY'),))
        self.assertEqual(self.sounds.phones('sea absent'), ())
        words = ['sea', 'see', 'read', 'red']
        original = self.provider.encode(words)
        order = [3, 1, 0, 2]
        torch.testing.assert_close(self.provider.encode([words[i] for i in order]), original[order][:, order])

    def test_resource_change_invalidates_identity(self):
        other = PronunciationLexicon({'sea': [['S', 'IY1']]})
        self.assertNotEqual(other.identity, self.sounds.identity)

    def test_expansion_preserves_scores_and_masks_added_channel_gradients(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from run_phonetic_experiment import expanded_model
        from tune_gine import build_model, BASELINE
        from nyt_connections_solver.graph_models import GraphSetTransformer
        torch.set_num_threads(1)
        idx = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5]])
        config = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
                      attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        reference = dict(model_config=config,
                         model_state_dict=GraphSetTransformer(group_idx=idx, **config).state_dict())
        original, _ = build_model(reference, BASELINE, idx, 10)
        x, old = torch.randn(2, 6, 8), torch.rand(2, 6, 6, 3)
        edges = torch.cat([old, torch.rand(2, 6, 6, 6)], dim=-1)
        expected = original(x, group_idx=idx, edge_features=old)
        for active in (range(3), range(9)):
            model, _ = expanded_model(reference, BASELINE, idx, 10, 9, active)
            scores = model(x, group_idx=idx, edge_features=edges)
            torch.testing.assert_close(scores, expected)
            (scores[:, 0] - scores[:, 1]).sum().backward()
            grad = model.edge_encoder.edge_projections[0].weight.grad[:, 3:]
            if len(active) == 3:
                self.assertEqual(torch.count_nonzero(grad), 0)
            else:
                self.assertGreater(torch.count_nonzero(grad), 0)


if __name__ == '__main__':
    unittest.main()
