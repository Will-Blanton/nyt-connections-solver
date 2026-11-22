"""Validate and concatenate provider channels, caching each family independently."""
import torch
from ..feature_cache import TensorCache


class EdgeFeatureComposer:
    def __init__(self, providers, *, cache=None):
        self.providers = tuple(providers)
        self.feature_names = tuple(name for provider in self.providers for name in provider.feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('Edge channel names must be nonempty and globally unique')
        self.dim = len(self.feature_names)
        self.cache = cache if cache is not None else TensorCache()
        self.configuration = [dict(identity=p.cache_identity, features=list(p.feature_names)) for p in self.providers]
        offset = 0
        candidate_channels = []
        for provider in self.providers:
            if callable(getattr(provider, 'encode_candidates', None)):
                candidate_channels.extend(range(offset, offset + len(provider.feature_names)))
            offset += len(provider.feature_names)
        self.candidate_feature_channels = tuple(candidate_channels)

    @property
    def model_options(self):
        """Include with StandaloneCandidateGINE's saved model_config."""
        return ({'candidate_edge_channels': list(self.candidate_feature_channels)}
                if self.candidate_feature_channels else {})

    def encode_candidates(self, words):
        """Cache compact quartet overrides once on each original board."""
        if not self.candidate_feature_channels:
            return None
        words = tuple(words)
        if len(words) < 4 or len(set(words)) != len(words):
            raise ValueError('Expected at least four distinct words')
        import itertools
        idx = torch.tensor(list(itertools.combinations(range(len(words)), 4)), dtype=torch.long)
        parts = []
        for provider, identity in zip(self.providers, self.configuration):
            if not callable(getattr(provider, 'encode_candidates', None)):
                continue
            key = {'type': 'candidate_edge_overrides', 'version': 1, 'provider': identity}
            value = self.cache.get(key, words)
            if value is None:
                value = torch.as_tensor(provider.encode_candidates(words, idx)).detach().float().cpu()
                self._validate_candidates(value, len(idx), len(provider.feature_names))
                self.cache.put(key, words, value)
            else:
                self._validate_candidates(value, len(idx), len(provider.feature_names))
            parts.append(value)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _validate_candidates(value, count, channels):
        if value.shape != (count, channels) or not torch.isfinite(value).all():
            raise ValueError('Invalid candidate edge feature shape or nonfinite values')
        if value.min() < 0 or value.max() > 1:
            raise ValueError('Candidate edge features must be in [0, 1]')

    def encode(self, words, _node_features=None):
        words = tuple(words)
        if not words or len(set(words)) != len(words):
            raise ValueError('Expected nonempty distinct puzzle words')
        parts = []
        for provider, identity in zip(self.providers, self.configuration):
            value = self.cache.get(identity, words)
            if value is None:
                value = torch.as_tensor(provider.encode(words)).detach().float().cpu()
                self._validate(value, len(words), len(provider.feature_names))
                self.cache.put(identity, words, value)
            else:
                self._validate(value, len(words), len(provider.feature_names))
            parts.append(value)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _validate(value, n, channels):
        if value.shape != (n, n, channels) or not torch.isfinite(value).all():
            raise ValueError('Edge provider returned an invalid shape or nonfinite values')
        if value.min() < 0 or value.max() > 1:
            raise ValueError('Edge values must be in [0, 1]')
        if not torch.allclose(value, value.transpose(0, 1), atol=1e-6):
            raise ValueError('Edges must be symmetric')
        if torch.count_nonzero(value[torch.arange(n), torch.arange(n)]):
            raise ValueError('Edge diagonals must be zero')
