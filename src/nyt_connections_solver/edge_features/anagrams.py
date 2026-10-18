"""Exact anagrams and multiset character overlap, ignoring spaces/punctuation."""
from collections import Counter
import torch
from .resources import letters


class Anagrams:
    feature_names = ('exact_anagram', 'character_multiset_overlap')
    cache_identity = {'type': 'anagrams', 'version': 1}

    def encode(self, words):
        counts = [Counter(letters(w)) for w in words]
        result = torch.zeros(len(words), len(words), 2)
        for i, a in enumerate(counts):
            for j in range(i + 1, len(counts)):
                b = counts[j]
                result[i, j] = result[j, i] = torch.tensor([
                    float(bool(a) and a == b),
                    2 * sum((a & b).values()) / max(1, sum(a.values()) + sum(b.values()))
                ])
        return result
