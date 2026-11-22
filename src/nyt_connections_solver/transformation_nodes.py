"""Cached, position-aware alternatives; lexical extraction never uses labels."""
import re
import torch


class TransformationNodeFeatures:
    # Each slot: frozen vector, [type, left_removed/32, right_removed/32,
    # start/length, end/length, retained_fraction], valid. Fixed-scale offsets
    # preserve exact character-position matches across different word lengths.
    POSITION_DIM = 6

    def __init__(self, base, lexicon, *, max_transformations=12, min_length=3):
        if max_transformations < 2 or min_length < 2:
            raise ValueError('Require >=2 slots and minimum length >=2')
        self.base, self.embeddings, self.lexicon = base, base.embeddings, lexicon
        self.max_transformations, self.min_length = max_transformations, min_length
        self.slot_dim = self.embeddings.dim + self.POSITION_DIM + 1
        self.dim = base.dim + max_transformations * self.slot_dim
        self.cache_identity = dict(type='word_transformations', version=2,
            base=base.cache_identity, lexicon=lexicon.identity, max_transformations=max_transformations,
            min_length=min_length, kinds=['original','prefix','suffix','internal'],
            selection='original_then_round_robin_kind_longest_first',
            position='type_fixed32_left_right_normalized_span_retained', single_ascii_word=True)

    def alternatives(self, word, *, truncate=True):
        word = word.casefold()
        rows = [(word, 0, 0, len(word))]
        if re.fullmatch('[a-z]+', word) and len(word) <= 32:
            groups = {1: [], 2: [], 3: []}
            for start in range(len(word)):
                for end in range(start + self.min_length, len(word)+1):
                    if start == 0 and end == len(word):
                        continue
                    part = word[start:end]
                    if part in self.lexicon.words:
                        kind = 1 if start == 0 else 2 if end == len(word) else 3
                        groups[kind].append((part, kind, start, end))
            for group in groups.values():
                group.sort(key=lambda r: (-(r[3]-r[2]), r[2], r[0]))
            while any(groups.values()):
                for kind in (1, 2, 3):
                    if groups[kind]:
                        rows.append(groups[kind].pop(0))
        return rows[:self.max_transformations] if truncate else rows

    def encode(self, words, convert_to_tensor=True):
        words = list(words)
        result = self.embeddings.cache.get(self.cache_identity, words)
        if result is None:
            alternatives = [self.alternatives(w) for w in words]
            vocabulary = sorted({r[0] for rows in alternatives for r in rows})
            vectors = self.embeddings.encode(vocabulary)
            lookup = dict(zip(vocabulary, vectors))
            payload = torch.zeros(len(words), self.max_transformations, self.slot_dim)
            for i, (word, rows) in enumerate(zip(words, alternatives)):
                n = max(1, len(word))
                for j, (part, kind, start, end) in enumerate(rows):
                    payload[i,j,:self.embeddings.dim] = lookup[part]
                    payload[i,j,self.embeddings.dim:-1] = torch.tensor(
                        [kind, start/32, (n-end)/32, start/n, end/n, (end-start)/n])
                    payload[i,j,-1] = 1
            result = torch.cat([self.base.encode(words), payload.flatten(1)], -1)
            self.embeddings.cache.put(self.cache_identity, words, result)
        return result if convert_to_tensor else result.numpy()
