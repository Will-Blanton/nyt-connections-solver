"""Coverage must distinguish absent/tied evidence and inconsistent witnesses."""
import itertools
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_graph_edge_coverage import candidate_evidence, phrase_witnesses, closed_compound_witnesses


class CoverageTests(unittest.TestCase):
    def test_closed_compounds_require_all_attested_forms(self):
        words = ["cray", "jelly", "silver", "star"]
        vocab = {"fish", "crayfish", "jellyfish", "silverfish", "starfish"}
        self.assertEqual(closed_compound_witnesses(words, vocab)["following"], ["fish"])
        vocab.remove("silverfish")
        self.assertEqual(closed_compound_witnesses(words, vocab)["following"], [])

    def test_zero_evidence_has_half_credit_and_true_groups_are_excluded(self):
        idx = torch.tensor(list(itertools.combinations(range(8), 4)))
        positive = torch.tensor([tuple(row) in {(0, 1, 2, 3), (4, 5, 6, 7)} for row in idx.tolist()])
        edges = torch.zeros(8, 8, 1)
        _, _, percentiles = candidate_evidence(edges, idx, positive)
        torch.testing.assert_close(percentiles, torch.full((2, 1), .5))
        edges[:4, :4] = 1
        edges[4:, 4:] = 1
        values, _, percentiles = candidate_evidence(edges, idx, positive)
        torch.testing.assert_close(percentiles, torch.ones(2, 1))
        self.assertTrue((values[positive] == 1).all())

    def test_pairwise_completion_clique_does_not_imply_one_shared_completion(self):
        words = ["a", "b", "c", "d"]
        index = {word: {"".join(sorted((word, other))) for other in words if other != word}
                 for word in words}
        result = phrase_witnesses(words, index)
        self.assertTrue(result["all_six_pairs_supported"])
        self.assertEqual(result["common_to_all_four"], [])
        for values in index.values():
            values.add("shared")
        self.assertEqual(phrase_witnesses(words, index)["common_to_all_four"], ["shared"])


if __name__ == "__main__":
    unittest.main()
