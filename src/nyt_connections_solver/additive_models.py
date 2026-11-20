"""Additive candidate logits with a frozen first-stage GINE."""
import math

import torch
from torch import nn

from .graph_models import StandaloneCandidateGINE


class AdditiveGINE(nn.Module):
    """A frozen scorer plus a trainable correction, or a frozen logit ensemble.

    Both component configurations are saved in checkpoints; loading does not
    depend on the original component checkpoint paths.
    """

    def __init__(self, *, group_idx, base_config, correction_config,
                 base_scale=1., correction_scale=.25, train_correction=True,
                 in_dim=None, edge_dim=None):
        super().__init__()
        for name, dimension in (('in_dim', in_dim), ('edge_dim', edge_dim)):
            expected = base_config[name] if dimension is None else dimension
            if base_config[name] != expected or correction_config[name] != expected:
                raise ValueError(f'Components must share checkpoint {name}')
        if not all(math.isfinite(x) and x > 0 for x in (base_scale, correction_scale)):
            raise ValueError('Component scales must be finite and positive')
        self.base = StandaloneCandidateGINE(group_idx=group_idx, **base_config)
        self.correction = StandaloneCandidateGINE(group_idx=group_idx, **correction_config)
        self.base.requires_grad_(False)
        if not train_correction:
            self.correction.requires_grad_(False)
        self.train_correction = train_correction
        self.base_scale = base_scale
        self.correction_scale = correction_scale
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        if not self.train_correction:
            self.correction.eval()
        return self

    def forward(self, x, group_idx=None, edge_features=None):
        with torch.no_grad():
            base = self.base(x, group_idx=group_idx, edge_features=edge_features)
        correction = self.correction(x, group_idx=group_idx, edge_features=edge_features)
        return self.base_scale * base + self.correction_scale * correction
