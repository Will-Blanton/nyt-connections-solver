"""Separate positive spelling properties; mutual absence never creates an edge."""
import torch
from .resources import letters


def spelling_properties(word):
    value = letters(word)
    return [
        float(len(value) >= 2 and value == value[::-1]),
        float(len(value) >= 2 and value[0] == value[1]),
        float(any(value[k] == value[k+1] and value[k+2] == value[k+3]
                  for k in range(len(value) - 3))),
    ]


class SpellingStructure:
    feature_names = ('both_palindromes', 'both_initial_double', 'both_consecutive_double_pairs')
    cache_identity = {'type': 'spelling_structure', 'version': 1}

    def encode(self, words):
        properties = torch.tensor([spelling_properties(w) for w in words], dtype=torch.float32)
        result = properties[:, None, :] * properties[None, :, :]
        result[torch.arange(len(words)), torch.arange(len(words))] = 0
        return result
