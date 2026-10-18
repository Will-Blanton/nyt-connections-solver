"""Fold-fitted auxiliary word features for global partition experiments."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


class PartitionFeatureEncoder:
    """Combine frozen semantic vectors with fold-fitted lexical features."""

    def __init__(self, semantic_vectors, char_dim=64):
        self.semantic_vectors = semantic_vectors
        self.char_dim = char_dim
        self.vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
        self.svd = None
        self.struct_mean = None
        self.struct_std = None

    @staticmethod
    def structural_features(words):
        suffixes = ("s", "ed", "ing", "er", "est", "ly", "tion", "ness", "able", "less")
        rows = []
        for word in words:
            value = str(word).lower()
            letters = [char for char in value if char.isalpha()]
            vowels = sum(char in "aeiou" for char in letters)
            row = [
                len(value), len(letters), vowels,
                sum(char.isalpha() and char not in "aeiou" for char in letters),
                vowels / max(1, len(letters)), int(" " in value),
                int("-" in value), int("'" in value), int(value == value[::-1]),
                sum(char.isdigit() for char in value),
            ]
            row.extend(int(value.endswith(suffix)) for suffix in suffixes)
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def soundex_features(words):
        codes = {
            **dict.fromkeys("BFPV", "1"), **dict.fromkeys("CGJKQSXZ", "2"),
            **dict.fromkeys("DT", "3"), "L": "4",
            **dict.fromkeys("MN", "5"), "R": "6",
        }
        rows = []
        for word in words:
            value = "".join(char for char in str(word).upper() if char.isalpha())
            if not value:
                value = "Z"
            first = value[0]
            encoded = []
            previous = codes.get(first, "")
            for char in value[1:]:
                code = codes.get(char, "")
                if code and code != previous:
                    encoded.append(code)
                previous = code
            code = (first + "".join(encoded) + "000")[:4]
            first_letter = np.zeros(26, dtype=np.float32)
            if "A" <= code[0] <= "Z":
                first_letter[ord(code[0]) - ord("A")] = 1.0
            digits = np.zeros(18, dtype=np.float32)
            for position, digit in enumerate(code[1:]):
                if digit.isdigit() and 1 <= int(digit) <= 6:
                    digits[position * 6 + int(digit) - 1] = 1.0
            rows.append(np.concatenate((first_letter, digits)))
        return np.asarray(rows, dtype=np.float32)

    def fit(self, training_words):
        words = list(dict.fromkeys(training_words))
        structural = self.structural_features(words)
        self.struct_mean = structural.mean(axis=0, keepdims=True)
        self.struct_std = structural.std(axis=0, keepdims=True)
        self.struct_std[self.struct_std == 0] = 1.0
        char_features = self.vectorizer.fit_transform(words)
        components = min(self.char_dim, char_features.shape[1] - 1, len(words) - 1)
        if components < 1:
            raise ValueError("Not enough training words to fit character SVD")
        self.svd = TruncatedSVD(n_components=components, random_state=0)
        self.svd.fit(char_features)
        self.feature_dims = {
            "semantic": int(next(iter(self.semantic_vectors.values())).shape[0]),
            "structural": int(structural.shape[1]),
            "phonetic": 44,
            "char_svd": int(components),
        }
        return self

    def encode(self, words):
        words = list(words)
        if self.svd is None:
            raise ValueError("Call fit() before encode()")
        structural = (self.structural_features(words) - self.struct_mean) / self.struct_std
        char_features = self.svd.transform(self.vectorizer.transform(words))
        return {
            "semantic": np.stack([self.semantic_vectors[word] for word in words]).astype(np.float32),
            "structural": structural.astype(np.float32),
            "phonetic": self.soundex_features(words),
            "char_svd": char_features.astype(np.float32),
        }
