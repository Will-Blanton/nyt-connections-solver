"""Versioned, optional SQLite tensor cache shared by nodes and edge providers."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sqlite3

import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TensorCache:
    """Each operation opens its own connection; no live handles enter DataLoader workers.

    refresh=True ignores prior disk entries once per key in this instance. The
    caller owns versioning: change a provider's cache_identity when its algorithm
    or resources change. Only tensors are serialized, using weights-only loading.
    """
    def __init__(self, directory=None, *, refresh=False):
        self.path = None
        self.refresh = refresh
        self._memory = {}
        self.hits = self.misses = 0
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / 'features.sqlite3'
            with sqlite3.connect(self.path, timeout=60) as connection:
                connection.execute('CREATE TABLE IF NOT EXISTS tensors (key TEXT PRIMARY KEY, value BLOB NOT NULL)')

    def get(self, identity, words):
        key = fingerprint([identity, list(words)])
        value = self._memory.get(key)
        if value is None and self.path is not None and not self.refresh:
            with sqlite3.connect(self.path, timeout=60) as connection:
                row = connection.execute('SELECT value FROM tensors WHERE key=?', (key,)).fetchone()
            if row is not None:
                value = torch.load(io.BytesIO(row[0]), map_location='cpu', weights_only=True)
                self._memory[key] = value
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value.clone()

    def put(self, identity, words, value):
        key = fingerprint([identity, list(words)])
        value = value.detach().float().cpu().contiguous().clone()
        self._memory[key] = value
        if self.path is not None:
            buffer = io.BytesIO()
            torch.save(value, buffer)
            with sqlite3.connect(self.path, timeout=60) as connection:
                connection.execute('INSERT OR REPLACE INTO tensors VALUES (?, ?)', (key, buffer.getvalue()))
