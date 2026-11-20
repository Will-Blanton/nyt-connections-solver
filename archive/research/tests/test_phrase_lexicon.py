import sys
from pathlib import Path
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from nyt_connections_solver.edge_features.phrase_lexicon import CompletionLexicon, AttestedPhraseCompletions
from audit_wiktionary_phrases import headwords, matching_quartets


class PhraseLexiconTests(unittest.TestCase):
    def test_mixed_compounds_share_completion_and_fragments_need_not_be_words(self):
        lex = CompletionLexicon(['eye candy', 'eye-contact', 'eyeshadow', 'eyewitness', 'crayfish'], source='fixture')
        for word in ['candy', 'contact', 'shadow', 'witness']:
            self.assertIn('eye', lex.completions(word, 'preceding'))
        self.assertEqual(lex.completions('cray', 'following'), {'fish'})
        self.assertNotIn('', lex.completions('crayfish', 'following'))
        self.assertEqual(lex.completions('x', 'following'), set())

    def test_non_english_and_inflection_records(self):
        self.assertIsNone(headwords({'lang_code': 'fr', 'word': 'chat'}))
        self.assertEqual(headwords({'lang_code': 'en', 'word': 'cats', 'senses': [{'form_of': [{'word':'cat'}]}]}), ('cats', False))
        self.assertEqual(headwords({'lang_code': 'en', 'word': 'cats', 'senses': [{'form_of': []}, {'glosses': ['a term']}]}), ('cats', True))

    def test_spaced_mode_avoids_arbitrary_substrings_and_keeps_multiword_templates(self):
        lex = CompletionLexicon(['happy camper', 'happy hour', 'happyness', 'head of state'], source='fixture', spaced_only=True)
        self.assertEqual(lex.completions('happy', 'following'), {'camper','hour'})
        self.assertEqual(lex.completions('state', 'preceding'), {'headof'})
        self.assertEqual(lex.completions('head of', 'following'), {'state'})
        self.assertEqual(lex.completions('ness', 'preceding'), set())

    def test_pairwise_and_common_witness_evidence_are_distinct(self):
        words = ['able', 'baker', 'charlie', 'delta']
        entries = []
        # Every pair gets its own attested completion; no single completion spans four.
        for i in range(4):
            for j in range(i+1,4):
                witness = 'z'+chr(97+i)+chr(97+j)
                entries.extend([words[i]+witness, words[j]+witness])
        lex = CompletionLexicon(entries, source='fixture')
        provider = AttestedPhraseCompletions(lex)
        edge = provider.encode(words)
        self.assertEqual(edge[:,:,1].sum(), 12)
        self.assertEqual(matching_quartets(words, lex.completions, 'following'), set())
        torch.testing.assert_close(edge, edge.transpose(0,1))
        torch.testing.assert_close(edge.diagonal(dim1=0,dim2=1), torch.zeros(2,4))
        permutation = [2,0,3,1]
        torch.testing.assert_close(provider.encode([words[i] for i in permutation]), edge[permutation][:,permutation])


if __name__ == '__main__':
    unittest.main()
