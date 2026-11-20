"""Cached pronunciation payloads for fixed and learned phonetic node encoders."""
import hashlib

import torch
import torch.nn.functional as F

from .edge_features.pronunciation import shared_pronunciations


class PhoneticNodeFeatures:
    """Append phoneme n-grams and padded phone IDs to the unchanged text features.

    One payload supports every experimental encoder, preserving data order and
    cache identity between fusion arms. Phone IDs are stored as exact float32
    integers for the shared node-tensor interface and converted back in-model.
    """
    def __init__(self, base, *, ngram_dim=128, max_pronunciations=8, max_phones=32,
                 pronunciations=None):
        if min(ngram_dim, max_pronunciations, max_phones) < 1:
            raise ValueError('Phonetic payload dimensions must be positive')
        self.base, self.embeddings = base, base.embeddings
        self.pronunciations = pronunciations if pronunciations is not None else shared_pronunciations()
        self.ngram_dim, self.max_pronunciations, self.max_phones = ngram_dim, max_pronunciations, max_phones
        self.vocabulary = tuple(sorted({phone for variants in self.pronunciations.entries.values()
                                       for variant in variants
                                       for phone in self.pronunciations.strip_stress(variant)}))
        self.phone_ids = {phone: i + 1 for i, phone in enumerate(self.vocabulary)}  # 0 = padding/unknown
        self.dim = base.dim + ngram_dim + max_pronunciations * max_phones
        self.cache_identity = dict(type='phonetic_nodes', version=1, base=base.cache_identity,
                                   pronunciation=self.pronunciations.identity, vocabulary=list(self.vocabulary),
                                   ngram_dim=ngram_dim, max_pronunciations=max_pronunciations,
                                   max_phones=max_phones, stress='ignored', ngrams=[1, 2, 3],
                                   hash='signed_blake2b_64', variant_pool='mean_then_l2')
        self._word_payloads = {}

    def word_payload(self, word):
        if word in self._word_payloads:
            return self._word_payloads[word]
        variants = self.pronunciations.phones(word)[:self.max_pronunciations]
        tokens = torch.zeros(self.max_pronunciations, self.max_phones)
        rows = []
        for i, phones in enumerate(variants):
            phones = phones[:self.max_phones]
            tokens[i, :len(phones)] = torch.tensor([self.phone_ids[p] for p in phones])
            padded = ('^', *phones, '$')
            row = torch.zeros(self.ngram_dim)
            for n in (1, 2, 3):
                for j in range(len(padded) - n + 1):
                    digest = int.from_bytes(hashlib.blake2b(
                        ' '.join(padded[j:j+n]).encode(), digest_size=8).digest(), 'little')
                    row[digest % self.ngram_dim] += 1. if (digest >> 32) & 1 else -1.
            rows.append(F.normalize(row, dim=0))
        ngrams = F.normalize(torch.stack(rows).mean(0), dim=0) if rows else torch.zeros(self.ngram_dim)
        result = torch.cat([ngrams, tokens.flatten()])
        self._word_payloads[word] = result
        return result

    def encode(self, words, convert_to_tensor=True):
        words = list(words)
        result = self.embeddings.cache.get(self.cache_identity, words)
        if result is None:
            base = self.base.encode(words, convert_to_tensor=True)
            payload = (torch.stack([self.word_payload(word) for word in words]) if words
                       else torch.empty(0, self.dim - self.base.dim))
            result = torch.cat([base, payload], dim=-1)
            self.embeddings.cache.put(self.cache_identity, words, result)
        return result if convert_to_tensor else result.numpy()
