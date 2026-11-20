"""Exact common lexical targets after one aligned substitution or deletion."""
from functools import lru_cache
from math import log
from string import ascii_lowercase

import torch

from .letter_edits import LetterEditSimilarity, OPERATIONS, ALIGNMENTS


class SharedLetterEditTargets(LetterEditSimilarity):
    """Reuse the audited reading generator; no transformed-word embeddings needed."""
    feature_names = tuple(f'shared_{operation}_{alignment}_target'
                          for operation in OPERATIONS for alignment in ALIGNMENTS)

    def __init__(self, lexicon, *, min_length=3, max_length=32):
        if not 2 <= min_length <= max_length:
            raise ValueError('Require 2 <= min_length <= max_length')
        self.lexicon, self.min_length, self.max_length = lexicon,min_length,max_length
        self.cache_identity = dict(type='shared_letter_edit_targets',version=1,
            lexicon=lexicon.identity,min_length=min_length,max_length=max_length,
            operations=list(OPERATIONS),alignments=list(ALIGNMENTS),
            normalization='casefold single ascii alphabetic word',
            candidate_rule='one common (position,target) witness across every word',
            specificity='log((vocabulary+1)/(max(1,attested_inverse_neighbours)+1))/log(vocabulary+1)')

    @lru_cache(maxsize=65536)
    def specificity(self, target, operation):
        sources = set()
        if operation == 'deletion':
            for position in range(len(target)+1):
                sources.update(target[:position]+c+target[position:] for c in ascii_lowercase)
        elif operation == 'substitution':
            for position in range(len(target)):
                sources.update(target[:position]+c+target[position+1:] for c in ascii_lowercase if c != target[position])
        else:
            raise ValueError(operation)
        count = len(sources & self.lexicon.words)
        total = len(self.lexicon.words)
        if total <= 1:
            return 0.
        return max(0.,log((total+1)/(max(1,count)+1))/log(total+1))

    def witnesses(self, word, operation, alignment):
        return frozenset((key[2],target) for key,targets in self.readings(word).items()
                         if key[:2] == (operation,alignment) for target in targets)

    def score(self, shared, operation):
        return max((self.specificity(target,operation) for _,target in shared),default=0.)

    def encode(self, words):
        result = torch.zeros(len(words),len(words),len(self.feature_names))
        for channel,(operation,alignment) in enumerate((op,align) for op in OPERATIONS for align in ALIGNMENTS):
            groups = [self.witnesses(w,operation,alignment) for w in words]
            for i in range(len(words)):
                for j in range(i):
                    result[i,j,channel] = result[j,i,channel] = self.score(groups[i]&groups[j],operation)
        return result

    def encode_candidates(self, words, group_idx):
        result = torch.zeros(len(group_idx),len(self.feature_names))
        for channel,(operation,alignment) in enumerate((op,align) for op in OPERATIONS for align in ALIGNMENTS):
            groups = [self.witnesses(w,operation,alignment) for w in words]
            for row,positions in enumerate(group_idx.tolist()):
                shared = frozenset.intersection(*(groups[i] for i in positions))
                result[row,channel] = self.score(shared,operation)
        return result
