"""Offline, attested completions, unifying spaced and closed compounds.

The caller supplies a versioned dictionary snapshot. No puzzle labels, network
lookups, or requirement that a puzzle fragment itself be a dictionary word.
"""
from bisect import bisect_left
from bisect import bisect_right
from functools import lru_cache
import math
import gzip
import json
from pathlib import Path

from .resources import phrase_form
from ..feature_cache import fingerprint


def compact_form(text):
    return phrase_form(text).replace(' ', '')


class CompletionLexicon:
    """Sorted prefix/suffix indexes; retain multiple possible witnesses per word.

    Only alphabetic entries participate. Removing spaces/hyphens intentionally
    allows mixed spellings (e.g. eye shadow / eyewitness), but also introduces
    incidental substring matches. This is evidence, not a grouping constraint.
    """
    def __init__(self, entries, *, source, spaced_only=False):
        self.spaced_only = spaced_only
        normalize = phrase_form if spaced_only else compact_form
        forms = {normalize(entry) for entry in entries}
        self.entries = tuple(sorted(form for form in forms if form.replace(' ', '').isalpha()))
        self.reversed_entries = tuple(sorted(form[::-1] for form in self.entries))
        self.identity = {'type': 'completion_lexicon', 'version': 1,
                         'source': source, 'entries': fingerprint(self.entries),
                         'minimum_completion_length': 2, 'spaced_only': spaced_only}

    @classmethod
    def from_headword_cache(cls, path, *, include_inflections=False, spaced_only=False):
        """Load the compact cache emitted by audit_wiktionary_phrases.py."""
        path = Path(path)
        metadata = json.loads(path.with_suffix('.metadata.json').read_text())
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            entries = [json.loads(line) for line in stream]
        source = {'source_url': metadata['source_url'], 'sha256': metadata['sha256'],
                  'inflections': include_inflections}
        return cls((word for word, lexical in entries if lexical or include_inflections),
                   source=source, spaced_only=spaced_only)

    @lru_cache(maxsize=32768)
    def completions(self, word, direction):
        if direction not in ('preceding', 'following'):
            raise ValueError('direction must be preceding or following')
        word = phrase_form(word) if self.spaced_only else compact_form(word)
        if not word or not word.replace(' ', '').isalpha():
            return frozenset()
        reverse = direction == 'preceding'
        query = word[::-1] if reverse else word
        if self.spaced_only:
            query += ' '
        entries = self.reversed_entries if reverse else self.entries
        found = set()
        start = bisect_left(entries, query)
        for index in range(start, len(entries)):
            entry = entries[index]
            if not entry.startswith(query):
                break
            remainder = entry[len(query):]
            remainder = compact_form(remainder[::-1] if reverse else remainder)
            if len(remainder) >= 2:
                found.add(remainder)
        return frozenset(found)

    @lru_cache(maxsize=65536)
    def completion_frequency(self, witness, direction):
        """Distinct compact titles with this end completion and a nonempty source."""
        if self.spaced_only:
            raise ValueError('Specificity requires the unified compact dictionary')
        if direction not in ('preceding', 'following'):
            raise ValueError('direction must be preceding or following')
        entries = self.entries if direction == 'preceding' else self.reversed_entries
        prefix = witness if direction == 'preceding' else witness[::-1]
        start = bisect_left(entries, prefix)
        stop = bisect_right(entries, prefix + '\U0010ffff')
        return stop - start - int(start < len(entries) and entries[start] == prefix)


class AttestedPhraseCompletions:
    """Optional pairwise channels; defaults and existing checkpoints are unchanged."""
    feature_names = ('attested_shared_preceding', 'attested_shared_following')

    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.cache_identity = {'type': 'attested_phrase_completions', 'version': 1,
                               'lexicon': lexicon.identity}

    def encode(self, words):
        import torch
        result = torch.zeros(len(words), len(words), 2)
        for channel, direction in enumerate(('preceding', 'following')):
            completions = [self.lexicon.completions(w, direction) for w in words]
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    result[i, j, channel] = result[j, i, channel] = bool(completions[i] & completions[j])
        return result


class SpecificPhraseCompletions:
    """IDF-weighted pair evidence plus exact shared-witness candidate overrides.

    Unrestricted attested closed compounds are retained. A candidate channel is
    positive only when *one identical witness* supports every candidate word.
    No dictionary/category annotations from the validation audit are used.
    """
    feature_names = ('specific_shared_preceding', 'specific_shared_following')

    def __init__(self, lexicon, *, max_frequency=None):
        if lexicon.spaced_only:
            raise ValueError('Use a unified compact dictionary, including closed compounds')
        if max_frequency is not None and (isinstance(max_frequency, bool) or
                not isinstance(max_frequency, int) or max_frequency < 1):
            raise ValueError('max_frequency must be a positive integer or None')
        self.lexicon = lexicon
        self.max_frequency = max_frequency
        self.cache_identity = {'type': 'specific_phrase_completions', 'version': 1,
                               'lexicon': lexicon.identity, 'max_frequency': max_frequency,
                               'weight': 'log((titles+1)/(frequency+1))/log(titles+1)',
                               'candidate_rule': 'max_weight_of_identical_witness_shared_by_all_words'}

    @lru_cache(maxsize=65536)
    def weight(self, witness, direction):
        frequency = self.lexicon.completion_frequency(witness, direction)
        total = len(self.lexicon.entries)
        if not total or not frequency or (self.max_frequency is not None and frequency > self.max_frequency):
            return 0.
        return max(0., min(1., math.log((total + 1)/(frequency + 1))/math.log(total + 1)))

    def _score(self, choices, direction):
        return max((self.weight(w, direction) for w in choices), default=0.)

    def encode(self, words):
        import torch
        result = torch.zeros(len(words), len(words), 2)
        for channel, direction in enumerate(('preceding', 'following')):
            choices = [self.lexicon.completions(w, direction) for w in words]
            for i in range(len(words)):
                for j in range(i):
                    result[i,j,channel] = result[j,i,channel] = self._score(choices[i] & choices[j], direction)
        return result

    def encode_candidates(self, words, group_idx):
        import torch
        result = torch.zeros(len(group_idx), 2)
        for channel, direction in enumerate(('preceding', 'following')):
            choices = [self.lexicon.completions(w, direction) for w in words]
            for row, positions in enumerate(group_idx.tolist()):
                shared = set.intersection(*(set(choices[i]) for i in positions))
                result[row,channel] = self._score(shared, direction)
        return result
