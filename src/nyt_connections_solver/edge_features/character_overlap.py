"""Literal contiguous character overlap; unrelated single letters score zero."""
from os.path import commonprefix
import torch
from .resources import letters


class CharacterOverlap:
    feature_names = ('longest_common_substring', 'common_prefix', 'common_suffix')
    cache_identity = {'type': 'character_overlap', 'version': 1, 'min_match': 2}

    def encode(self, words):
        values = [letters(w) for w in words]
        result = torch.zeros(len(words), len(words), 3)
        for i, a in enumerate(values):
            for j in range(i + 1, len(values)):
                b = values[j]
                previous, longest = [0] * (len(b) + 1), 0
                for ca in a:
                    row = [0] * (len(b) + 1)
                    for k, cb in enumerate(b, 1):
                        if ca == cb:
                            row[k] = previous[k - 1] + 1
                            longest = max(longest, row[k])
                    previous = row
                lengths = [longest, len(commonprefix([a, b])), len(commonprefix([a[::-1], b[::-1]]))]
                result[i, j] = result[j, i] = torch.tensor([
                    2 * length / max(1, len(a) + len(b)) if length >= 2 else 0 for length in lengths
                ])
        return result
