"""Maximum semantic similarity of valid proper prefixes/suffixes (no categories)."""
import torch
import torch.nn.functional as F
from .resources import letters


class SubstringSimilarity:
    feature_names = ('prefix_semantic_similarity', 'suffix_semantic_similarity')

    def __init__(self, lexicon, embeddings, *, min_length=3):
        if min_length < 2:
            raise ValueError('min_length must be at least 2')
        self.lexicon, self.embeddings, self.min_length = lexicon, embeddings, min_length
        self.cache_identity = {'type': 'substring_similarity', 'version': 1,
                               'lexicon': lexicon.identity, 'embedding': embeddings.identity,
                               'min_length': min_length}

    def encode(self, words):
        normalized = [letters(w) for w in words]
        fragments = []
        for suffix in (False, True):
            fragments.append([
                [s for k in range(self.min_length, len(w))
                 if (s := (w[-k:] if suffix else w[:k])) in self.lexicon.words]
                for w in normalized
            ])
        vocabulary = sorted({s for channel in fragments for group in channel for s in group})
        result = torch.zeros(len(words), len(words), 2)
        if not vocabulary:
            return result
        vectors = F.normalize(self.embeddings.encode(vocabulary), dim=-1)
        lookup = {w: k for k, w in enumerate(vocabulary)}
        for channel, groups in enumerate(fragments):
            for i, a in enumerate(groups):
                for j in range(i + 1, len(words)):
                    b = groups[j]
                    if not a or not b:
                        continue
                    similarity = (vectors[[lookup[s] for s in a]] @ vectors[[lookup[s] for s in b]].T).clamp(0, 1)
                    # Confidence penalizes short fragments; aligned extraction is
                    # enforced by separate prefix and suffix channels.
                    confidence_a = torch.tensor([min(1., len(s) / 4) for s in a])
                    confidence_b = torch.tensor([min(1., len(s) / 4) for s in b])
                    result[i, j, channel] = result[j, i, channel] = (similarity * confidence_a[:, None] * confidence_b[None, :]).max()
        return result
