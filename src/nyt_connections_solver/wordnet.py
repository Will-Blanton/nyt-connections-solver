"""WordNet-derived text representations for puzzle words."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class WordNetFeatureEncoder:
    """Embed WordNet gloss/synonym context, caching one vector per word.

    NLTK's Open Multilingual WordNet is queried across its available languages;
    synset glosses remain English, while the supplied sentence embedder is
    multilingual. Words with no WordNet entry receive a zero vector.
    """

    def __init__(self, embedder, *, max_senses: int = 8) -> None:
        if max_senses < 1:
            raise ValueError("max_senses must be positive")
        self.embedder = embedder
        self.max_senses = max_senses
        self.dim = int(embedder.get_sentence_embedding_dimension())
        self._cache: dict[str, np.ndarray] = {}
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

    def _context(self, word: str) -> str | None:
        normalized = str(word).strip().lower()
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
        if not synsets:
            return None

        descriptions = []
        for synset in list(synsets.values())[: self.max_senses]:
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
        return "WordNet senses for " + normalized + ": " + " ".join(descriptions)

    def encode(self, words: Sequence[str]) -> np.ndarray:
        words = [str(word) for word in words]
        missing = list(dict.fromkeys(word for word in words if word not in self._cache))
        contexts = []
        context_words = []
        for word in missing:
            context = self._context(word)
            if context is None:
                self._cache[word] = np.zeros(self.dim, dtype=np.float32)
            else:
                context_words.append(word)
                contexts.append(context)

        if contexts:
            vectors = self.embedder.encode(
                contexts, convert_to_numpy=True, show_progress_bar=False
            )
            for word, vector in zip(context_words, vectors):
                self._cache[word] = np.asarray(vector, dtype=np.float32)

        return np.stack([self._cache[word] for word in words]).astype(np.float32)
