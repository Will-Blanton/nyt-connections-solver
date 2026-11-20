"""Semantic evidence after one dictionary-attested substitution or deletion."""
from functools import lru_cache
from math import log
from string import ascii_lowercase

import torch
import torch.nn.functional as F


OPERATIONS = ('substitution', 'deletion')
ALIGNMENTS = ('first', 'last', 'internal_from_start', 'internal_from_end', 'anywhere')


class LetterEditSimilarity:
    feature_names = tuple(f'{operation}_{alignment}_{metric}'
                          for operation in OPERATIONS for alignment in ALIGNMENTS
                          for metric in ('semantic', 'reading_specificity'))

    def __init__(self, lexicon, embeddings, *, min_length=3, max_length=32):
        if not 2 <= min_length <= max_length:
            raise ValueError('Require 2 <= min_length <= max_length')
        self.lexicon, self.embeddings = lexicon, embeddings
        self.min_length, self.max_length = min_length, max_length
        self.cache_identity = dict(
            type='letter_edit_semantics', version=1, lexicon=lexicon.identity,
            embedding=embeddings.identity, min_length=min_length, max_length=max_length,
            operations=list(OPERATIONS), alignments=list(ALIGNMENTS),
            distinct_target_words=True, normalization='casefold single ascii alphabetic word',
            specificity='1 / (1 + ln(number of distinct aligned ordered reading pairs))')

    @lru_cache(maxsize=65536)
    def readings(self, word):
        """Map (operation, alignment, position) to sorted, exact one-edit readings.

        Nonwords are allowed as inputs. Only outputs must be in the lexicon.
        Multiword entries are not concatenated into invented single-word forms.
        """
        word = word.strip().casefold()
        if not word.isascii() or not word.isalpha() or not self.min_length <= len(word) <= self.max_length:
            return {}
        buckets = {}
        for position in range(len(word)):
            keys = [('anywhere', 0)]
            if position == 0:
                keys.append(('first', 0))
            elif position == len(word) - 1:
                keys.append(('last', 0))
            else:
                keys.extend([('internal_from_start', position),
                             ('internal_from_end', len(word) - 1 - position)])
            candidates = {
                'substitution': [word[:position] + c + word[position+1:]
                                 for c in ascii_lowercase if c != word[position]],
                'deletion': [word[:position] + word[position+1:]],
            }
            for operation, values in candidates.items():
                valid = {v for v in values if len(v) >= self.min_length and v in self.lexicon.words}
                if valid:
                    for alignment, offset in keys:
                        buckets.setdefault((operation, alignment, offset), set()).update(valid)
        return {key: tuple(sorted(values)) for key, values in buckets.items()}

    def vocabulary(self, words):
        return sorted({value for word in words for values in self.readings(word).values() for value in values})

    @staticmethod
    def aligned_pairs(left, right, operation, alignment):
        """Keep positional witnesses aligned and count unique lexical pairs once."""
        return tuple(sorted({(a, b) for key in left.keys() & right.keys()
                             if key[:2] == (operation, alignment)
                             for a in left[key] for b in right[key] if a != b}))

    def encode(self, words):
        readings = [self.readings(w) for w in words]
        vocabulary = self.vocabulary(words)
        result = torch.zeros(len(words), len(words), len(self.feature_names))
        if not vocabulary:
            return result
        vectors = F.normalize(self.embeddings.encode(vocabulary), dim=-1)
        cosine = (vectors @ vectors.T).clamp(0, 1)
        lookup = {word: i for i, word in enumerate(vocabulary)}
        for i, left in enumerate(readings):
            for j in range(i + 1, len(words)):
                for family, operation in enumerate(OPERATIONS):
                    for slot, alignment in enumerate(ALIGNMENTS):
                        pairs = self.aligned_pairs(left, readings[j], operation, alignment)
                        if not pairs:
                            continue
                        a, b = zip(*pairs)
                        channel = 2 * (len(ALIGNMENTS) * family + slot)
                        values = torch.tensor([
                            cosine[[lookup[w] for w in a], [lookup[w] for w in b]].max().item(),
                            1 / (1 + log(len(pairs))),
                        ])
                        result[i, j, channel:channel+2] = values
                        result[j, i, channel:channel+2] = values
        return result
