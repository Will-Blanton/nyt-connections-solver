"""Specific WordNet ancestor witnesses shared by pairs or entire candidates."""
from collections import Counter
from functools import lru_cache
from math import log

import torch

from ..feature_cache import fingerprint


@lru_cache(maxsize=2)
def concept_statistics(wordnet):
    """Count distinct descendant synsets, including self, without external IC data."""
    totals, descendants = Counter(), Counter()
    for synset in wordnet.all_synsets():
        totals[synset.pos()] += 1
        found, pending = set(), [synset]
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(current.hypernyms())
            pending.extend(current.instance_hypernyms())
        descendants.update(found)
    weights = {sense: max(0., log((totals[sense.pos()] + 1)/(count + 1))/log(totals[sense.pos()] + 1))
               for sense, count in descendants.items()}
    identity = fingerprint(sorted((sense.name(), value) for sense, value in weights.items()))
    return weights, identity


class SharedWordNetConcept:
    feature_names = ('shared_wordnet_concept_specificity',)

    def __init__(self, lexicon):
        if lexicon.wordnet is None:
            raise ValueError('Shared concepts require WordNet')
        self.lexicon = lexicon
        self.weights, statistics_identity = concept_statistics(lexicon.wordnet)
        self.cache_identity = dict(type='shared_wordnet_concept', version=1,
            lexicon=lexicon.identity, statistics=statistics_identity,
            specificity='per-POS log((synsets+1)/(descendant_synsets+1))/log(synsets+1)',
            candidate_rule='maximum specificity of one ancestor shared by all words')

    @lru_cache(maxsize=32768)
    def concepts(self, word):
        return frozenset().union(*(self.lexicon.ancestors(s) for s in self.lexicon.senses(word)))

    def score(self, concepts):
        return max((self.weights.get(c,0.) for c in concepts), default=0.)

    def encode(self, words):
        groups = [self.concepts(w) for w in words]
        result = torch.zeros(len(words),len(words),1)
        for i in range(len(words)):
            for j in range(i):
                result[i,j,0] = result[j,i,0] = self.score(groups[i] & groups[j])
        return result

    def encode_candidates(self, words, group_idx):
        groups = [self.concepts(w) for w in words]
        return torch.tensor([[self.score(frozenset.intersection(*(groups[i] for i in positions)))]
                             for positions in group_idx.tolist()], dtype=torch.float32)
