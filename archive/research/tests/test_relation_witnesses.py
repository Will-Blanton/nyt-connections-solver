"""Shared concept identity and aligned, exact letter-edit targets."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nyt_connections_solver.edge_features.resources import EnglishLexicon
from nyt_connections_solver.edge_features.concept_witnesses import SharedWordNetConcept
from nyt_connections_solver.edge_features.edit_targets import SharedLetterEditTargets


class Sense:
    def __init__(self,name,parent=None):self.label,self.parent=name,parent
    def name(self):return self.label
    def pos(self):return 'n'
    def hypernyms(self):return [self.parent] if self.parent else []
    def instance_hypernyms(self):return []


class Corpus:
    def __init__(self,senses):self.values=senses
    def all_synsets(self):return iter(self.values)


class Lexicon:
    identity={'type':'fixture'}
    def __init__(self,groups,senses):self.groups,self.wordnet=groups,Corpus(senses)
    def senses(self,word):return self.groups[word]
    def ancestors(self,sense):
        return {sense}|(self.ancestors(sense.parent) if sense.parent else set())


class RelationWitnessTests(unittest.TestCase):
    def test_expansion_preserves_initial_scores_and_new_channels_receive_gradients(self):
        from run_gine_relation_witnesses import fresh_expanded_model
        torch.set_num_threads(1)
        config=dict(in_dim=8,edge_dim=3,edge_bias_scale=0.,attn_width=8,attn_heads=2,
                    attn_layers=1,lin_width=16,lin_layers=1,dropout=0.)
        params=dict(hidden_width=8,gine_layers=2,readout_width=8,dropout=0.)
        idx=torch.tensor([[0,1,2,3]])
        control,_=fresh_expanded_model(config,params,idx,10,5,range(3),(3,4))
        enabled,_=fresh_expanded_model(config,params,idx,10,5,range(5),(3,4))
        regular,regular_config=fresh_expanded_model(config,params,idx,10,5,range(5),(3,4),coherence_scale=0.)
        self.assertEqual(regular_config['edge_coherence_scale'],0.)
        for name,value in control.state_dict().items():
            torch.testing.assert_close(value,enabled.state_dict()[name],rtol=0,atol=0)
            torch.testing.assert_close(value,regular.state_dict()[name],rtol=0,atol=0)
        x=torch.randn(2,4,8);edges=torch.rand(2,4,4,5);candidates=torch.rand(2,1,2)
        a=control(x,edge_features=edges,candidate_edge_features=candidates)
        b=enabled(x,edge_features=edges,candidate_edge_features=candidates)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        b.square().sum().backward()
        gradients=[p.grad[:,-2:].abs().sum() for name,p in enabled.edge_encoder.named_parameters()
                   if 'edge_projections' in name and name.endswith('weight')]
        self.assertGreater(sum(gradients),0)

    def test_pairwise_concept_clique_does_not_imply_shared_specific_concept(self):
        root=Sense('entity')
        concepts=[Sense(str(i),root) for i in range(6)]
        words=['a','b','c','d'];groups={w:set() for w in words}
        import itertools
        for concept,(i,j) in zip(concepts,itertools.combinations(range(4),2)):
            groups[words[i]].add(concept);groups[words[j]].add(concept)
        provider=SharedWordNetConcept(Lexicon(groups,[root]+concepts))
        pair=provider.encode(words)
        self.assertEqual(torch.count_nonzero(pair),12)
        self.assertEqual(provider.encode_candidates(words,torch.tensor([[0,1,2,3]])).item(),0)
        for group in groups.values():group.add(concepts[0])
        provider.concepts.cache_clear()
        self.assertGreater(provider.encode_candidates(words,torch.tensor([[0,1,2,3]])).item(),0)
        self.assertEqual(provider.weights[root],0)

    def test_win_is_a_real_exact_target_without_semantic_embeddings(self):
        words=['wind','wine','wing','wink']
        provider=SharedLetterEditTargets(EnglishLexicon(['win','wind','wine','wing','wink','unrelated']))
        values=provider.encode_candidates(words,torch.tensor([[0,1,2,3]]))
        channel=provider.feature_names.index('shared_deletion_last_target')
        self.assertGreater(values[0,channel],0)
        for word in words:self.assertIn((0,'win'),provider.witnesses(word,'deletion','last'))
        self.assertTrue(torch.isfinite(values).all())
        pair=provider.encode(words)
        torch.testing.assert_close(pair,pair.transpose(0,1))
        self.assertFalse(pair.diagonal(dim1=0,dim2=1).any())

    def test_alignment_is_preserved_and_unknown_sources_are_allowed(self):
        words=['xcart','cxart','caxrt','carxt']
        provider=SharedLetterEditTargets(EnglishLexicon(['cart','unrelated','other']))
        values=provider.encode_candidates(words,torch.tensor([[0,1,2,3]]))[0]
        channel=provider.feature_names.index('shared_deletion_anywhere_target')
        self.assertGreater(values[channel],0)
        for alignment in ('first','last','internal_from_start','internal_from_end'):
            self.assertEqual(values[provider.feature_names.index('shared_deletion_'+alignment+'_target')],0)


if __name__=='__main__':unittest.main()
