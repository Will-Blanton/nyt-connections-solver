"""Shared English lexical indexes and frozen, cached word embeddings."""
from collections import defaultdict
from functools import lru_cache

import torch

from ..feature_cache import TensorCache, fingerprint


def letters(word):
    return ''.join(c for c in word.casefold() if c.isalpha())


def phrase_form(word):
    return ' '.join(word.casefold().replace('_', ' ').replace('-', ' ').split())


class EnglishLexicon:
    """WordNet English lemmas, with phrase indexes at every token boundary.

    No category labels or puzzle solutions are used. Compound spelling variants
    are not invented: only attested WordNet entries are indexed.
    """
    def __init__(self, entries=None, *, wordnet=None):
        self.wordnet = wordnet
        if entries is None:
            if self.wordnet is None:
                from nltk.corpus import wordnet as wn
                try:
                    wn.ensure_loaded()
                except LookupError as exc:
                    raise RuntimeError('Install the English corpus first: python -m nltk.downloader wordnet') from exc
                self.wordnet = wn
            entries = self.wordnet.all_lemma_names(lang='eng')
        self.entries = frozenset(phrase_form(w) for w in entries)
        self.words = frozenset(w for w in self.entries if w.isalpha())
        self.preceding = defaultdict(set)
        self.following = defaultdict(set)
        self.closed_preceding = defaultdict(set)
        self.closed_following = defaultdict(set)
        for entry in self.entries:
            tokens = entry.split()
            for split in range(1, len(tokens)):
                left, right = ' '.join(tokens[:split]), ' '.join(tokens[split:])
                self.preceding[right].add(left)
                self.following[left].add(right)
        for entry in self.words:
            for split in range(2, len(entry) - 1):
                left, right = entry[:split], entry[split:]
                if left in self.words and right in self.words:
                    self.closed_following[left].add(right)
                    self.closed_preceding[right].add(left)
        self.identity = {'type': 'english_lexicon', 'version': 1,
                         'entries': fingerprint(sorted(self.entries)),
                         'wordnet_version': self.wordnet.get_version() if self.wordnet is not None else None}

    @lru_cache(maxsize=32768)
    def senses(self, word):
        if self.wordnet is None:
            return frozenset()
        forms = {word.casefold(), phrase_form(word).replace(' ', '_')}
        return frozenset(s for form in forms for s in self.wordnet.synsets(form, lang='eng'))

    @lru_cache(maxsize=32768)
    def ancestors(self, sense):
        found, pending = set(), [sense]
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(current.hypernyms())
                pending.extend(current.instance_hypernyms())
        return frozenset(found)


class EmbeddingStore:
    """Batch unseen strings once; cache individual vectors across feature families."""
    def __init__(self, model, model_id, *, revision=None, cache=None):
        self.model = model
        dimension = getattr(model, 'get_embedding_dimension', None)
        self.dim = int(dimension() if dimension else model.get_sentence_embedding_dimension())
        if revision is None:
            try:
                revision = model[0].auto_model.config._commit_hash
            except (AttributeError, TypeError, KeyError):
                pass
        if not revision:
            raise ValueError('Supply a stable revision for a local/custom embedding model')
        self.identity = {'type': 'embedding', 'version': 1, 'model': model_id,
                         'revision': revision, 'dim': self.dim, 'prompt': ''}
        self.cache = cache if cache is not None else TensorCache()
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, words):
        words = list(words)
        found = {w: self.cache.get(self.identity, [w]) for w in dict.fromkeys(words)}
        missing = [w for w, vector in found.items() if vector is None]
        if missing:
            with torch.inference_mode():
                vectors = torch.as_tensor(self.model.encode(
                    missing, convert_to_tensor=True, show_progress_bar=False, prompt=''
                )).detach().float().cpu()
            if vectors.shape != (len(missing), self.dim) or not torch.isfinite(vectors).all():
                raise ValueError('Invalid embedding output')
            for word, vector in zip(missing, vectors):
                self.cache.put(self.identity, [word], vector)
                found[word] = vector
        return torch.stack([found[w] for w in words]) if words else torch.empty(0, self.dim)
