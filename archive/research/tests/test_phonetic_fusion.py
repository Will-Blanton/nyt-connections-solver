from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.edge_features import PronunciationLexicon
from nyt_connections_solver.feature_cache import TensorCache
from nyt_connections_solver.phonetic_nodes import PhoneticNodeFeatures
from nyt_connections_solver.phonetic_models import PhoneticFusionGINE
from nyt_connections_solver.graph_models import StandaloneCandidateGINE


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.sounds = PronunciationLexicon({'sea': [['S', 'IY1']], 'see': [['S', 'IY1']],
            'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']], 'red': [['R', 'EH1', 'D']]})
        base = SimpleNamespace(dim=4, cache_identity={'fake': 'base'},
            embeddings=SimpleNamespace(cache=TensorCache(), dim=4),
            encode=lambda words, **kw: torch.tensor([[len(w), sum(map(ord, w)) % 7, 0., 1.] for w in words]).float())
        self.nodes = PhoneticNodeFeatures(base, ngram_dim=16, max_pronunciations=2, max_phones=8,
                                          pronunciations=self.sounds)
        self.config = dict(in_dim=self.nodes.dim, text_dim=4, phonetic_dim=16,
                           max_pronunciations=2, max_phones=8, phone_vocab_size=len(self.nodes.vocabulary)+1,
                           edge_dim=1, edge_bias_scale=0., lin_layers=1, lin_width=8,
                           attn_layers=1, attn_width=8, attn_heads=2, dropout=0.,
                           edge_hidden=8, gine_layers=1, train_text_projection=True, train_input_norm=True)
        self.idx = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5]])

    def test_payload_homophones_unknown_alternatives_order_and_cache(self):
        x = self.nodes.encode(['sea', 'see', 'read', 'absent'])
        torch.testing.assert_close(x[0, 4:], x[1, 4:])
        self.assertEqual(torch.count_nonzero(x[-1, 4:]), 0)
        tokens = x[2, 20:].reshape(2, 8)
        self.assertTrue(tokens.ne(0).any(-1).all())
        self.assertTrue(torch.equal(tokens, tokens.long().float()))
        torch.testing.assert_close(self.nodes.encode(['absent', 'read', 'see', 'sea']), x.flip(0))
        x.fill_(42)
        self.assertNotEqual(self.nodes.encode(['sea', 'see', 'read', 'absent'])[0, 0], 42)

    def test_all_fusions_start_from_semantic_scores_and_learn(self):
        x = self.nodes.encode(['sea', 'see', 'read', 'red', 'absent', 'other'])[None]
        edges = torch.rand(1, 6, 6, 1)
        base_config = {k: v for k, v in self.config.items() if k not in (
            'text_dim', 'phonetic_dim', 'max_pronunciations', 'max_phones', 'phone_vocab_size')}
        base_config['in_dim'] = 4
        torch.manual_seed(10)
        base = StandaloneCandidateGINE(group_idx=self.idx, **base_config)
        expected = base(x[..., :4], edge_features=edges)
        for encoding in ('ngram', 'cnn'):
            for fusion in ('linear', 'gated'):
                with self.subTest(encoding=encoding, fusion=fusion):
                    model = PhoneticFusionGINE(group_idx=self.idx, **self.config,
                                              phonetic_encoding=encoding, phonetic_fusion=fusion)
                    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
                    self.assertFalse(unexpected)
                    self.assertTrue(all(n.startswith(('phone_', 'phonetic_')) for n in missing))
                    torch.testing.assert_close(model(x, edge_features=edges), expected)
                    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
                    for _ in range(4):
                        optimizer.zero_grad()
                        scores = model(x, edge_features=edges)
                        (scores[:, 0] - scores[:, 1]).sum().backward()
                        optimizer.step()
                    self.assertGreater(model.phonetic_projection.weight.grad.abs().sum(), 0)
                    if encoding == 'cnn':
                        self.assertGreater(model.phone_embedding.weight.grad.abs().sum(), 0)
                        self.assertEqual(torch.count_nonzero(model.phone_embedding.weight.grad[0]), 0)
                    if fusion == 'gated':
                        self.assertGreater(model.phonetic_gate.weight.grad.abs().sum(), 0)
                    semantic, projected, gate = model.fusion_parts(x)
                    self.assertEqual(torch.count_nonzero(projected[:, -2:]), 0)
                    self.assertEqual(torch.count_nonzero(gate[:, -2:]), 0)
                    restored = PhoneticFusionGINE(group_idx=self.idx, **self.config,
                                                 phonetic_encoding=encoding, phonetic_fusion=fusion)
                    restored.load_state_dict(model.state_dict())
                    torch.testing.assert_close(model(x, edge_features=edges), restored(x, edge_features=edges))

    def test_disabled_nodes_exactly_ignore_payload(self):
        model = PhoneticFusionGINE(group_idx=self.idx, **self.config, phonetic_enabled=False)
        x = self.nodes.encode(['sea', 'see', 'read', 'red', 'absent', 'other'])[None]
        edges = torch.rand(1, 6, 6, 1)
        changed = x.clone()
        changed[..., 4:] = 99
        torch.testing.assert_close(model(x, edge_features=edges), model(changed, edge_features=edges), rtol=0, atol=0)

    def test_cnn_gated_checkpoint_loads_through_normal_harness(self):
        from nyt_connections_solver.checkpoints import load_graph_harness
        config = {**self.config, 'phonetic_encoding': 'cnn', 'phonetic_fusion': 'gated'}
        model = PhoneticFusionGINE(group_idx=self.idx, **config).eval()
        metadata = {'options': {'revision': 'test'}, 'nodes': self.nodes.cache_identity}
        edges = SimpleNamespace(dim=1, feature_names=['edge'],
                                encode=lambda words, _: torch.zeros(len(words), len(words), 1))
        checkpoint = dict(model_class='PhoneticFusionGINE', model_config=config,
                          model_state_dict=model.state_dict(), feature_metadata=metadata,
                          edge_feature_names=['edge'], embedding_model_name='fake')
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'cnn-gated.pt'
            torch.save(checkpoint, weights)
            with patch('nyt_connections_solver.checkpoints.SentenceTransformer', return_value=torch.nn.Identity()), \
                 patch('nyt_connections_solver.checkpoints.build_graph_features', return_value=(self.nodes, edges, metadata)):
                loaded = load_graph_harness(weights, device='cpu', beam_width=1)
            x = self.nodes.encode(['sea', 'see', 'read', 'red', 'absent', 'other'])[None]
            e = torch.zeros(1, 6, 6, 1)
            torch.testing.assert_close(model(x, group_idx=self.idx, edge_features=e),
                                       loaded.model(x, group_idx=self.idx, edge_features=e))

    def test_sequence_encoding_is_independent_between_boards(self):
        model = PhoneticFusionGINE(group_idx=self.idx, **self.config,
                                  phonetic_encoding='cnn', phonetic_fusion='gated')
        torch.nn.init.normal_(model.phonetic_projection.weight, std=.1)
        first = self.nodes.encode(['sea', 'see', 'read', 'red', 'absent', 'other'])
        second = first.flip(0)
        x = torch.stack([first, second])
        edges = torch.rand(2, 6, 6, 1)
        combined = model(x, edge_features=edges)
        for i in range(2):
            torch.testing.assert_close(combined[i:i+1], model(x[i:i+1], edge_features=edges[i:i+1]))


if __name__ == '__main__':
    unittest.main()
