"""Semantic cosine and the original WordNet overlap channels."""
from collections import Counter
from math import log
import torch
import torch.nn.functional as F


class SemanticSimilarity:
    feature_names = ('semantic_cosine',)

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.cache_identity = {'type': 'semantic_cosine', 'version': 1, 'embedding': embeddings.identity}

    def encode(self, words):
        vectors = self.embeddings.encode(words)
        unit = F.normalize(vectors, dim=-1)
        result = ((unit @ unit.T).clamp(-1, 1) + 1) / 2
        valid = vectors.norm(dim=-1) > 0
        result *= valid[:, None] & valid[None, :]
        result.fill_diagonal_(0)
        return result.unsqueeze(-1)


class WordNetSimilarity:
    feature_names = ('sense_jaccard', 'rare_concept_jaccard')

    def __init__(self, lexicon):
        if lexicon.wordnet is None:
            raise ValueError('WordNetSimilarity requires the English WordNet corpus')
        self.lexicon = lexicon
        self.cache_identity = {'type': 'wordnet_similarity', 'version': 1, 'lexicon': lexicon.identity}

    def encode(self, words):
        senses = [self.lexicon.senses(w) for w in words]
        concepts = [set().union(*(self.lexicon.ancestors(s) for s in group)) for group in senses]
        support = Counter(c for group in concepts for c in group)
        weights = {c: log(len(words) / count) for c, count in support.items()}
        result = torch.zeros(len(words), len(words), 2)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                union = senses[i] | senses[j]
                denominator = sum(weights[c] for c in concepts[i] | concepts[j])
                values = [len(senses[i] & senses[j]) / len(union) if union else 0,
                          sum(weights[c] for c in concepts[i] & concepts[j]) / denominator if denominator else 0]
                result[i, j] = result[j, i] = torch.tensor(values)
        return result
