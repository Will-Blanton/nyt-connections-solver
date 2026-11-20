from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from group_mechanisms import classify,validate_review
from audit_group_mechanisms import aggregate_edges,tie_percentile,phrase_consistency,standalone_top_choice


class GroupMechanismAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_specific_mechanisms_precede_generic_topics_and_unknown_is_marked(self):
        cases={'homophones of movie titles':'sound', 'anagrams of instruments':'anagram',
               'fruits with second letters changed':'letter_edit', '___ fish':'phrase_completion',
               'silent "g"':'sound','palindromes':'orthographic_pattern',
               'synonyms for tiny':'synonym','what shell may mean':'polysemy',
               'NBA teams':'named_entity_or_cultural'}
        for label,expected in cases.items():
            with self.subTest(label=label):
                result=classify(label)
                self.assertEqual(result['primary_mechanism'],expected)
                self.assertTrue(result['matched_text'])
        unknown=classify('things you might find on a desk')
        self.assertEqual(unknown['label_confidence'],'low')
        self.assertEqual(unknown['label_source'],'fallback')

    def test_aggregation_distinguishes_three_word_clique_from_four_word_support(self):
        edges=torch.zeros(4,4,2)
        edges[:3,:3,0]=1
        edges[:,:,1]=.75
        edges.diagonal(dim1=0,dim2=1).fill_(999)
        idx=torch.tensor([[0,1,2,3]])
        values,aggregations=aggregate_edges(edges,idx)
        torch.testing.assert_close(aggregations['mean'],torch.tensor([[.5,.75]]))
        torch.testing.assert_close(aggregations['minimum'],torch.tensor([[0.,.75]]))
        torch.testing.assert_close(aggregations['weakest_word'],torch.tensor([[0.,.75]]))
        self.assertEqual(values.shape,(1,6,2))
        self.assertEqual(edges[0,0,0],999)

    def test_percentiles_give_half_credit_for_ties_and_use_only_supplied_competitors(self):
        positives=torch.tensor([[0.,1.,.5]])
        negatives=torch.tensor([[0.,0.,.25],[0.,1.,.75]])
        torch.testing.assert_close(tie_percentile(positives,negatives),torch.tensor([[.5,.75,.5]]))

    def test_all_pair_witnesses_do_not_imply_one_shared_completion(self):
        words=['a','b','c','d']
        idx=torch.tensor([[0,1,2,3]])
        index={word:set() for word in words}
        for i in range(4):
            for j in range(i+1,4):
                witness=str(i)+str(j)
                index[words[i]].add(witness)
                index[words[j]].add(witness)
        pairs,shared,_=phrase_consistency(words,index,idx)
        self.assertTrue(pairs[0])
        self.assertFalse(shared[0])
        for word in words:
            index[word].add('common')
        self.assertTrue(phrase_consistency(words,index,idx)[1][0])

    def test_standalone_ties_do_not_turn_constant_features_into_success(self):
        scores=torch.tensor([[1.,0.],[1.,0.],[0.,0.]])
        gold=torch.tensor([True,False,True])
        expected,parts,counts,_=standalone_top_choice(scores,gold)
        torch.testing.assert_close(expected,torch.tensor([.5,2/3]))
        torch.testing.assert_close(parts.sum(0),expected)
        torch.testing.assert_close(counts,torch.tensor([2,3]))
        gold=torch.zeros(1820,dtype=torch.bool)
        gold[:4]=True
        expected=standalone_top_choice(torch.zeros(1820,2),gold)[0]
        torch.testing.assert_close(expected,torch.full((2,),4/1820))

    def test_closed_normalization_and_review_validation(self):
        words=['race car','go cart','tea pot','sun set']
        index={word.replace(' ',''):{'common'} for word in words}
        idx=torch.tensor([[0,1,2,3]])
        self.assertTrue(phrase_consistency(words,index,idx,closed=True)[1][0])
        self.assertFalse(phrase_consistency(words,index,idx)[1][0])
        review=dict(primary_mechanism='sound',secondary_mechanisms='["semantic_category"]',confidence='medium')
        validate_review(review)
        with self.assertRaises(ValueError):
            validate_review({**review,'primary_mechanism':'made_up'})


if __name__=='__main__':
    unittest.main()
