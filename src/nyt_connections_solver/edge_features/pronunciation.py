"""Dictionary-attested sound relationships; no puzzle/category supervision."""
from collections import defaultdict
from functools import lru_cache
from itertools import product
import re

import torch
import torch.nn.functional as F

from ..feature_cache import fingerprint
from .resources import letters, phrase_form


class PronunciationLexicon:
    """CMUdict alternatives, cached stress-free phones, and homophone lookup.

    Missing words stay unknown. Multiword entries use attested token
    pronunciations, capped deterministically at 32 combinations.
    """
    def __init__(self, entries=None):
        if entries is None:
            from nltk.corpus import cmudict
            try:
                entries = cmudict.dict()
            except LookupError as exc:
                raise RuntimeError('Install pronunciation data: python -m nltk.downloader cmudict') from exc
        normalized = defaultdict(set)
        for word, variants in entries.items():
            for phones in variants:
                if phones:
                    normalized[phrase_form(word)].add(tuple(phones))
        self.entries = {word: tuple(sorted(variants)) for word, variants in normalized.items()}
        self.identity = dict(type='cmudict_pronunciations', version=1,
                             entries=fingerprint(sorted(self.entries.items())), max_phrase_variants=32)
        self.homophones = defaultdict(set)
        for word, variants in self.entries.items():
            for phones in variants:
                self.homophones[self.strip_stress(phones)].add(word)

    @staticmethod
    def strip_stress(phones):
        return tuple(re.sub(r'[012]$', '', phone) for phone in phones)

    @lru_cache(maxsize=65536)
    def pronunciations(self, word):
        word = phrase_form(word)
        if word in self.entries:
            return self.entries[word]
        tokens = word.split()
        if len(tokens) < 2 or any(token not in self.entries for token in tokens):
            return ()
        combinations = product(*(self.entries[token] for token in tokens))
        result = []
        for variants in combinations:
            result.append(tuple(phone for variant in variants for phone in variant))
            if len(result) == 32:
                break
        return tuple(result)

    @lru_cache(maxsize=65536)
    def phones(self, word):
        return tuple(sorted({self.strip_stress(p) for p in self.pronunciations(word)}))

    @lru_cache(maxsize=65536)
    def alternatives(self, word):
        normalized = phrase_form(word)
        # Only alphabetic words or phrases; exclude spelling/punctuation variants
        # of the original so this channel genuinely changes the lexical reading.
        return tuple(sorted({candidate for p in self.phones(word) for candidate in self.homophones[p]
                             if all(token.isalpha() for token in candidate.split())
                             and letters(candidate) != letters(normalized)}))


@lru_cache(maxsize=1)
def shared_pronunciations():
    """Reuse the immutable corpus/index across model reloads in one process."""
    return PronunciationLexicon()


@lru_cache(maxsize=262144)
def phone_similarity(left, right):
    """Normalized Levenshtein similarity over phoneme tokens, not characters."""
    if not left or not right:
        return 0.
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return 1. - previous[-1] / max(len(left), len(right))


class PronunciationSimilarity:
    feature_names = ('homophones', 'phoneme_edit_similarity', 'shared_rhyme',
                     'homophone_alternative_semantic_similarity',
                     'prefix_phonetic_similarity', 'suffix_phonetic_similarity')

    def __init__(self, lexicon, embeddings, *, pronunciations=None, min_length=3):
        if min_length < 2:
            raise ValueError('min_length must be at least 2')
        self.lexicon, self.embeddings = lexicon, embeddings
        self.pronunciations = pronunciations if pronunciations is not None else shared_pronunciations()
        self.min_length = min_length
        self.cache_identity = dict(type='pronunciation_similarity', version=1,
                                   pronunciation=self.pronunciations.identity, lexicon=lexicon.identity,
                                   embedding=embeddings.identity, min_length=min_length)

    @staticmethod
    def rhyme(phones):
        # Last primary/secondary stressed vowel through the end. Retain stress
        # while locating it; ignore stress after extracting the rhyme.
        for i in range(len(phones) - 1, -1, -1):
            if phones[i].endswith(('1', '2')):
                return PronunciationLexicon.strip_stress(phones[i:])
        return ()

    def fragments(self, word, suffix=False):
        word = letters(word)
        return tuple(fragment for k in range(self.min_length, len(word))
                     if (fragment := (word[-k:] if suffix else word[:k])) in self.lexicon.words
                     and self.pronunciations.phones(fragment))

    def encode(self, words):
        phones = [self.pronunciations.phones(w) for w in words]
        rhymes = [{r for p in self.pronunciations.pronunciations(w) if (r := self.rhyme(p))} for w in words]
        alternatives = [self.pronunciations.alternatives(w) for w in words]
        vocabulary = sorted({a for group in alternatives for a in group})
        vectors = F.normalize(self.embeddings.encode(vocabulary), dim=-1) if vocabulary else None
        lookup = {word: i for i, word in enumerate(vocabulary)}
        fragments = [[self.fragments(w, suffix) for w in words] for suffix in (False, True)]
        result = torch.zeros(len(words), len(words), len(self.feature_names))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                result[i, j, 0] = bool(set(phones[i]) & set(phones[j]))
                result[i, j, 1] = max((phone_similarity(a, b) for a in phones[i] for b in phones[j]), default=0.)
                result[i, j, 2] = bool(rhymes[i] & rhymes[j])
                if alternatives[i] and alternatives[j]:
                    # A genuinely different attested reading on BOTH sides.
                    result[i, j, 3] = (vectors[[lookup[w] for w in alternatives[i]]] @
                                       vectors[[lookup[w] for w in alternatives[j]]].T).clamp(0, 1).max()
                for channel, groups in enumerate(fragments, 4):
                    result[i, j, channel] = max((
                        phone_similarity(pa, pb) * min(1., len(a)/4) * min(1., len(b)/4)
                        for a in groups[i] for b in groups[j]
                        for pa in self.pronunciations.phones(a) for pb in self.pronunciations.phones(b)
                    ), default=0.)
                result[j, i] = result[i, j]
        return result
