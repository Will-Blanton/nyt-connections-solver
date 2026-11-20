from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wiktionary_filters import dictionary_frequency,quartet_support
from nyt_connections_solver.edge_features.phrase_lexicon import CompletionLexicon


class WiktionaryFilterTests(unittest.TestCase):
    def test_frequency_counts_titles_at_correct_end_and_excludes_empty_source(self):
        lex=CompletionLexicon(['fish','catfish','gold fish','fish cake','cat-fish'],source='test')
        self.assertEqual(dictionary_frequency(lex,'fish','following'),2)
        self.assertEqual(dictionary_frequency(lex,'fish','preceding'),1)
        self.assertEqual(dictionary_frequency(lex,'absent','following'),0)

    def test_shared_witness_removes_inconsistent_pair_clique(self):
        choices=[set() for _ in range(4)]
        for i in range(4):
            for j in range(i):
                choices[i].add(str((i,j)));choices[j].add(str((i,j)))
        quartets=np.array([[0,1,2,3]])
        clique,common=quartet_support(choices,quartets)
        self.assertTrue(clique[0]);self.assertFalse(common[0])
        for c in choices:c.add('shared')
        self.assertTrue(quartet_support(choices,quartets)[1][0])


if __name__=='__main__':unittest.main()
