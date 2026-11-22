"""Exact witnesses survive caching, state slicing, GINE, and harness scoring."""
import itertools
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nyt_connections_solver.candidate_features import slice_candidate_features, candidate_edge_kwargs
from nyt_connections_solver.data_loader import make_state_loaders
from nyt_connections_solver.edge_features import EdgeFeatureComposer, CompletionLexicon, SpecificPhraseCompletions
from nyt_connections_solver.feature_cache import TensorCache
from nyt_connections_solver.graph_models import StandaloneCandidateGINE
from nyt_connections_solver.harness import SearchHarness
from nyt_connections_solver.feedback_search import ModelGroupScorer
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.checkpoints import load_graph_harness


WORDS = ('alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel',
         'india','juliet','kilo','lima','mike','november','oscar','papa')


class Nodes:
    def encode(self, words, **kwargs):
        return torch.tensor([[float(ord(c)%19) for c in (word+'abcdefgh')[:8]] for word in words])


class OtherEdges:
    feature_names = ('other',)
    cache_identity = {'type':'fixture_other','version':1}
    def encode(self, words):
        return (1-torch.eye(len(words)))[:,:,None]*.25


class Embedder(torch.nn.Module):
    def get_embedding_dimension(self):
        return 8
    def encode(self,words,**kwargs):
        return Nodes().encode(words)


class SpecificPhraseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lex = CompletionLexicon([w+'fish' for w in WORDS[:4]]+['unrelated'],source='test')
        self.provider = SpecificPhraseCompletions(self.lex)
        self.edges = EdgeFeatureComposer([OtherEdges(),self.provider],cache=TensorCache(self.temp.name))
        self.config = dict(in_dim=8,edge_dim=3,edge_bias_scale=0.,attn_width=8,
                           attn_heads=2,attn_layers=1,lin_width=16,lin_layers=1,
                           dropout=0.,edge_hidden=8,train_text_projection=True,
                           train_input_norm=True,**self.edges.model_options)

    def test_inconsistent_clique_is_removed_and_soft_weights_keep_closed_compounds(self):
        entries = []
        for i,j in itertools.combinations(range(4),2):
            witness='zz'+chr(97+i)+chr(97+j)
            entries.extend([WORDS[i]+witness, WORDS[j]+witness])
        # Additional titles prevent a universal fixture witness from having zero IDF.
        entries.extend(['unrelated','otherword'])
        provider = SpecificPhraseCompletions(CompletionLexicon(entries,source='test'))
        pair = provider.encode(WORDS[:4])
        self.assertEqual(torch.count_nonzero(pair[:,:,1]),12)
        candidate=provider.encode_candidates(WORDS[:4],torch.tensor([[0,1,2,3]]))
        self.assertEqual(candidate.sum(),0)
        lex=CompletionLexicon(entries+[w+'fish' for w in WORDS[:4]],source='test')
        provider=SpecificPhraseCompletions(lex)
        self.assertGreater(provider.encode_candidates(WORDS[:4],torch.tensor([[0,1,2,3]]))[0,1],0)
        capped=SpecificPhraseCompletions(lex,max_frequency=3)
        self.assertEqual(capped.encode_candidates(WORDS[:4],torch.tensor([[0,1,2,3]]))[0,1],0)
        self.assertNotEqual(provider.cache_identity,capped.cache_identity)

    def test_common_endings_get_lower_weight_and_cap_is_optional(self):
        entries=['alphaer','bravoer','charlieer','deltaer','echoer','alphafish','bravofish','unrelated']
        provider=SpecificPhraseCompletions(CompletionLexicon(entries,source='test'))
        self.assertGreater(provider.weight('fish','following'),provider.weight('er','following'))
        self.assertGreater(provider.weight('er','following'),0)
        self.assertEqual(provider.weight('absent','following'),0)
        for cap in (0,-1,1.5,True):
            with self.assertRaises(ValueError):SpecificPhraseCompletions(self.lex,max_frequency=cap)

    def test_cached_tables_slice_identically_to_recomputation_including_permutations(self):
        table=self.edges.encode_candidates(WORDS)
        self.assertEqual(table.shape,(1820,2))
        self.assertEqual(self.edges.candidate_feature_channels,(1,2))
        with patch.object(self.provider,'encode_candidates',side_effect=AssertionError('cache missed')):
            torch.testing.assert_close(table,self.edges.encode_candidates(WORDS))
            reloaded = EdgeFeatureComposer([OtherEdges(),self.provider],cache=TensorCache(self.temp.name))
            torch.testing.assert_close(table,reloaded.encode_candidates(WORDS))
        positions=[3,0,5,1,2,6,7,8]
        idx=torch.tensor(list(itertools.combinations(range(8),4)))
        actual=slice_candidate_features(table,positions)
        expected=self.provider.encode_candidates([WORDS[i] for i in positions],idx)
        torch.testing.assert_close(actual,expected)

    def test_dataloader_gine_and_harness_agree_on_candidate_features_and_gradients(self):
        frame=pd.DataFrame([dict(date='fixture',connections=list(WORDS[i:i+4])) for i in range(0,16,4)])
        datasets,loaders=make_state_loaders(frame,None,None,2,shuffle=False,
            feature_builder=Nodes(),edge_providers=self.edges)
        model=StandaloneCandidateGINE(group_idx=torch.tensor([[0,1,2,3]]),**self.config).eval()
        captured=[]
        handle=model.edge_encoder.register_forward_pre_hook(lambda _,args:captured.append(args[1].detach().clone()))
        self.addCleanup(handle.remove)
        for size in (2,3,4):
            batch=next(iter(loaders[size]))
            idx=torch.tensor(list(itertools.combinations(range(size*4),4)))
            scores=model(batch['x'],group_idx=idx,edge_features=batch['edge_features'],
                         **candidate_edge_kwargs(batch,'cpu'))
            self.assertEqual(scores.shape,(len(batch['x']),len(idx),1))
            self.assertTrue(torch.isfinite(scores).all())
            values=captured[-1]
            torch.testing.assert_close(values[...,0],batch['edge_features'][:,idx[:,:,None],idx[:,None,:],0].reshape(-1,4,4))
            offdiag=~torch.eye(4,dtype=torch.bool)
            expected=batch['candidate_edge_features'].reshape(-1,2)
            torch.testing.assert_close(values[...,1:][:,offdiag],expected[:,None,:].expand(-1,12,-1))
            self.assertFalse(values[...,1:].diagonal(dim1=1,dim2=2).any())
        scores.sum().backward()
        self.assertIsNotNone(model.proj[0].weight.grad)
        self.assertGreater(model.proj[0].weight.grad.abs().sum(),0)
        harness=SearchHarness.from_model(model,None,input_dim=8,device='cpu',
            node_feature_builder=Nodes(),edge_feature_builder=self.edges.encode)
        scorer=ModelGroupScorer(harness)
        scorer.prepare(WORDS)
        state=tuple(WORDS[i] for i in [3,0,5,1,2,6,7,8])
        scored=scorer.score_states([state])[0]
        idx=torch.tensor(list(itertools.combinations(range(8),4)))
        positions=[WORDS.index(w) for w in state]
        direct=model(Nodes().encode(state)[None],group_idx=idx,
            edge_features=self.edges.encode(state)[None],
            candidate_edge_features=slice_candidate_features(self.edges.encode_candidates(WORDS),positions)[None])
        torch.testing.assert_close(torch.tensor([r.log_probability for r in scored]),direct.squeeze().log_softmax(0))
        # The standard beam harness also routes exact quartet features.
        result=harness.solve(WORDS)
        self.assertEqual(len(result.best_groups),4)

    def test_required_candidate_data_cannot_be_silently_ignored_and_reload_is_exact(self):
        idx=torch.tensor([[0,1,2,3]])
        model=StandaloneCandidateGINE(group_idx=idx,**self.config).eval()
        x=Nodes().encode(WORDS[:4])[None]
        edges=self.edges.encode(WORDS[:4])[None]
        with self.assertRaises(ValueError):model(x,edge_features=edges)
        candidates=self.edges.encode_candidates(WORDS[:4])[None]
        restored=StandaloneCandidateGINE(group_idx=idx,**self.config).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(model(x,edge_features=edges,candidate_edge_features=candidates),
                                   restored(x,edge_features=edges,candidate_edge_features=candidates))

    def test_feature_metadata_reconstructs_through_production_checkpoint_loader(self):
        cache=Path(self.temp.name)/'titles.jsonl.gz'
        with gzip.open(cache,'wt') as stream:
            for title in [w+'fish' for w in WORDS[:4]]+['unrelated']:
                stream.write(json.dumps([title,True])+'\n')
        cache.with_suffix('.metadata.json').write_text(json.dumps({'source_url':'fixture','sha256':'fixture'}))
        embedder=Embedder()
        nodes,edges,metadata=build_graph_features(embedder,'fixture',revision='test',use_spelling=False,
            include_wiktionary=True,wiktionary_cache_path=str(cache),cache_dir=self.temp.name)
        self.assertEqual(edges.dim,17)
        self.assertEqual(edges.candidate_feature_channels,(15,16))
        config={**self.config,'edge_dim':edges.dim,**edges.model_options}
        idx=torch.tensor(list(itertools.combinations(range(16),4)))
        model=StandaloneCandidateGINE(group_idx=idx,**config).eval()
        path=Path(self.temp.name)/'model.pt'
        payload=dict(model_class='StandaloneCandidateGINE',model_config=config,
            model_state_dict=model.state_dict(),feature_metadata=metadata,
            edge_feature_names=list(edges.feature_names),embedding_model_name='fixture')
        torch.save(payload,path)
        with patch('nyt_connections_solver.checkpoints.SentenceTransformer',return_value=embedder):
            loaded=load_graph_harness(path,device='cpu',beam_width=1,cache_dir=self.temp.name)
        original=SearchHarness.from_model(model,None,input_dim=8,device='cpu',beam_width=1,
            node_feature_builder=nodes,edge_feature_builder=edges.encode)
        a,b=ModelGroupScorer(original),ModelGroupScorer(loaded)
        a.prepare(WORDS);b.prepare(WORDS)
        torch.testing.assert_close(torch.tensor([r.log_probability for r in a.score_states([WORDS])[0]]),
                                   torch.tensor([r.log_probability for r in b.score_states([WORDS])[0]]))
        # Missing model configuration must be rejected at load, not silently ignored.
        payload['model_config']={**config,'candidate_edge_channels':[]}
        torch.save(payload,path)
        with patch('nyt_connections_solver.checkpoints.SentenceTransformer',return_value=embedder):
            with self.assertRaisesRegex(ValueError,'candidate edge channels'):
                load_graph_harness(path,device='cpu',cache_dir=self.temp.name)


if __name__=='__main__':unittest.main()
