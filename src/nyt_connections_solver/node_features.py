"""Frozen text plus deterministic spelling features, with no sense embeddings."""
import hashlib
import math
import torch
import torch.nn.functional as F
from .edge_features.resources import letters
from .edge_features.spelling_structure import spelling_properties


class TextSpellingFeatures:
    structural_names = ('log_length', 'log_letter_count', 'vowel_fraction',
                        'log_token_count', 'has_space', 'has_hyphen', 'has_apostrophe',
                        'digit_fraction', 'double_fraction', 'initial_double',
                        'terminal_double', 'palindrome', 'consecutive_double_pairs')

    def __init__(self, embeddings, *, char_dim=128, use_spelling=True):
        if char_dim < 1:
            raise ValueError('char_dim must be positive')
        self.embeddings = embeddings
        self.char_dim, self.use_spelling = char_dim, use_spelling
        self.dim = embeddings.dim + (char_dim + len(self.structural_names) if use_spelling else 0)
        self.cache_identity = {'type': 'text_spelling', 'version': 1, 'embedding': embeddings.identity,
                               'char_dim': char_dim, 'use_spelling': use_spelling}

    def encode(self, words, convert_to_tensor=True):
        words = list(words)
        result = self.embeddings.cache.get(self.cache_identity, words)
        if result is None:
            semantic = self.embeddings.encode(words)
            result = torch.cat([semantic, self.spelling(words)], dim=-1) if self.use_spelling else semantic
            self.embeddings.cache.put(self.cache_identity, words, result)
        return result if convert_to_tensor else result.numpy()

    def spelling(self, words):
        rows = []
        for word in words:
            w, alpha = word.casefold(), letters(word)
            padded = '^' + w + '$'
            hashed = torch.zeros(self.char_dim)
            for n in (1, 2, 3):
                for i in range(len(padded) - n + 1):
                    digest = hashlib.blake2b(padded[i:i+n].encode(), digest_size=8).digest()
                    hashed[int.from_bytes(digest, 'little') % self.char_dim] += 1
            hashed = F.normalize(hashed, dim=0)
            doubles = sum(a == b for a, b in zip(alpha, alpha[1:]))
            palindrome, initial, consecutive = spelling_properties(word)
            structural = torch.tensor([
                math.log1p(len(w)), math.log1p(len(alpha)),
                sum(c in 'aeiou' for c in alpha) / max(1, len(alpha)),
                math.log1p(len(w.split())), float(' ' in w), float('-' in w), float("'" in w),
                sum(c.isdigit() for c in w) / max(1, len(w)), doubles / max(1, len(alpha)-1),
                initial, float(len(alpha) >= 2 and alpha[-1] == alpha[-2]), palindrome, consecutive,
            ])
            rows.append(torch.cat([hashed, structural]))
        return torch.stack(rows) if rows else torch.empty(0, self.char_dim + len(self.structural_names))
