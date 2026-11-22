"""WordNet-derived text representations for puzzle words."""

from __future__ import annotations

from typing import Sequence
from collections import Counter
from math import log

import numpy as np


class WordNetFeatureEncoder:
    """Embed selected WordNet glosses, optionally conditioned on a full puzzle.

    NLTK's Open Multilingual WordNet is queried across its available languages;
    synset glosses remain English, while the supplied sentence embedder is
    multilingual. Words with no WordNet entry receive a zero vector.
    """

    def __init__(self, embedder, *, max_senses: int = 8, puzzle_conditioned: bool = False) -> None:
        if max_senses < 1:
            raise ValueError("max_senses must be positive")
        self.embedder = embedder
        self.max_senses = max_senses
        self.puzzle_conditioned = puzzle_conditioned
        self.dim = int(embedder.get_sentence_embedding_dimension())
        self._cache: dict[tuple, np.ndarray] = {}
        self._sense_cache = {}
        self._ancestor_cache = {}
        self._wordnet = self._load_wordnet()

    @staticmethod
    def _load_wordnet():
        import nltk

        try:
            from nltk.corpus import wordnet

            wordnet.ensure_loaded()
        except LookupError:
            if not nltk.download("wordnet", quiet=True):
                raise RuntimeError("Could not download the NLTK WordNet corpus")
            from nltk.corpus import wordnet

            wordnet.ensure_loaded()

        # OMW is optional: English WordNet still works if its download is unavailable.
        try:
            wordnet.langs()
        except LookupError:
            nltk.download("omw-1.4", quiet=True)
        return wordnet

    def _senses(self, word: str):
        normalized = str(word).strip().lower()
        if normalized in self._sense_cache:
            return self._sense_cache[normalized]
        synsets = {}
        query_forms = list(dict.fromkeys((normalized, normalized.replace(" ", "_"), normalized.replace("-", "_"))))
        for query in query_forms:
            for synset in self._wordnet.synsets(query, lang="eng"):
                synsets[synset.name()] = synset
        try:
            languages = self._wordnet.langs()
        except LookupError:
            languages = ["eng"]
        # The English puzzle corpus takes the fast path; use OMW languages for
        # words absent from English WordNet.
        for language in languages if not synsets else ():
            try:
                for query in query_forms:
                    for synset in self._wordnet.synsets(query, lang=language):
                        synsets[synset.name()] = synset
            except (LookupError, ValueError):
                continue
        self._sense_cache[normalized] = tuple(synsets.values())
        return self._sense_cache[normalized]

    def _ancestors(self, sense):
        # Include the sense itself so exact synonym overlap also counts.
        if sense not in self._ancestor_cache:
            concepts, pending = set(), [sense]
            while pending:
                concept = pending.pop()
                if concept not in concepts:
                    concepts.add(concept)
                    pending.extend(concept.hypernyms())
                    pending.extend(concept.instance_hypernyms())
            self._ancestor_cache[sense] = concepts
        return self._ancestor_cache[sense]

    def select_senses(self, words: Sequence[str]):
        """Rank senses by rare shared ancestors; ties prefer deeper concepts.

        Count each concept at most once per word, across all of its senses.
        Unsupported senses retain their original order as a fallback.
        """
        words = list(dict.fromkeys(str(word) for word in words))
        senses = {word: self._senses(word) for word in words}
        if not self.puzzle_conditioned:
            return {word: values[:self.max_senses] for word, values in senses.items()}
        if len(words) != 16:
            raise ValueError("Conditioned WordNet features require the original 16 distinct words")

        support = Counter()
        for values in senses.values():
            support.update(set().union(*(self._ancestors(sense) for sense in values)))
        scores = {
            concept: (log(len(words) / count), concept.max_depth())
            for concept, count in support.items()
            if 2 <= count < len(words)
        }

        def score(sense):
            return max((scores.get(c, (0.0, -1)) for c in self._ancestors(sense)))

        return {
            word: tuple(sorted(values, key=score, reverse=True)[:self.max_senses])
            for word, values in senses.items()
        }

    def _context(self, word: str, senses) -> str | None:
        if not senses:
            return None

        descriptions = []
        for synset in senses:
            lemmas = sorted(set(synset.lemma_names()))
            hypernyms = sorted({
                lemma
                for parent in synset.hypernyms()[:3]
                for lemma in parent.lemma_names()
            })
            descriptions.append(
                f"{', '.join(lemmas)}: {synset.definition()}. "
                f"Related broader concepts: {', '.join(hypernyms)}."
            )
        return "WordNet senses for " + str(word).strip().lower() + ": " + " ".join(descriptions)

    def encode(self, words: Sequence[str]) -> np.ndarray:
        words = [str(word) for word in words]
        if not words:
            return np.empty((0, self.dim), dtype=np.float32)
        selected = self.select_senses(words)
        # Different puzzles may select different senses for the same word.
        keys = {word: (word, tuple(s.name() for s in selected[word])) for word in selected}
        missing = [word for word in selected if keys[word] not in self._cache]
        contexts = []
        context_words = []
        for word in missing:
            context = self._context(word, selected[word])
            if context is None:
                self._cache[keys[word]] = np.zeros(self.dim, dtype=np.float32)
            else:
                context_words.append(word)
                contexts.append(context)

        if contexts:
            vectors = self.embedder.encode(
                contexts, convert_to_numpy=True, show_progress_bar=False
            )
            for word, vector in zip(context_words, vectors):
                self._cache[keys[word]] = np.asarray(vector, dtype=np.float32)

        return np.stack([self._cache[keys[word]] for word in words]).astype(np.float32)
