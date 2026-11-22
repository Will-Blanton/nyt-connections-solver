"""Shared phrase completions from lexical entries, without category lookup."""
import torch
from .resources import phrase_form


class PhraseCompletions:
    feature_names = ('shared_preceding_word', 'shared_following_word')

    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.cache_identity = {'type': 'phrase_completions', 'version': 1, 'lexicon': lexicon.identity}

    def encode(self, words):
        result = torch.zeros(len(words), len(words), 2)
        # Exact existence evidence: no invented corpus-frequency confidence.
        for channel, index in enumerate([self.lexicon.preceding, self.lexicon.following]):
            completions = [index.get(phrase_form(w), set()) for w in words]
            for i, a in enumerate(completions):
                for j in range(i + 1, len(words)):
                    result[i, j, channel] = result[j, i, channel] = float(bool(a & completions[j]))
        return result


class ClosedCompoundCompletions:
    """Shared, attested closed-compound completions in either direction."""
    feature_names = ('shared_closed_compound_prefix', 'shared_closed_compound_suffix')

    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.cache_identity = {'type': 'closed_compound_completions', 'version': 1,
                               'lexicon': lexicon.identity}

    def encode(self, words):
        result = torch.zeros(len(words), len(words), 2)
        for channel, index in enumerate((self.lexicon.closed_preceding, self.lexicon.closed_following)):
            completions = [index.get(phrase_form(w).replace(' ', ''), set()) for w in words]
            for i, left in enumerate(completions):
                for j in range(i + 1, len(words)):
                    result[i, j, channel] = result[j, i, channel] = float(bool(left & completions[j]))
        return result
