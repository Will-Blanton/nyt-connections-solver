"""Experimental objectives for supervising multiple correct candidate groups."""
import math
import torch
from torch.nn import functional as F


def per_group_contrastive_loss(scores, relevance):
    """Average positive-vs-all-negatives log-softmax, per state then per batch.

    Other correct groups are excluded from each positive's denominator. Thus
    recognizing one positive cannot satisfy the objective for another positive.
    """
    logits = scores.squeeze(-1).float() if scores.ndim == 3 else scores.float()
    positive = relevance.unsqueeze(0).expand_as(logits) if relevance.ndim == 1 else relevance
    if logits.ndim != 2 or positive.shape != logits.shape or positive.dtype != torch.bool:
        raise ValueError('Expected boolean candidate labels matching batch logits')
    counts = positive.sum(dim=1)
    if not (counts > 0).all():
        raise ValueError('Every training state needs at least one positive candidate')
    has_negative = (~positive).any(dim=1)
    negative_logits = logits.masked_fill(positive, -torch.inf)
    # Give all-positive states a finite dummy denominator; their loss is zero.
    negative_logits = torch.where(has_negative[:, None], negative_logits, torch.zeros_like(logits))
    negative_mass = torch.logsumexp(negative_logits, dim=1, keepdim=True)
    terms = F.softplus(negative_mass - logits)
    return (torch.where(positive & has_negative[:, None], terms, 0.).sum(dim=1) / counts).mean()


def probability_mass_with_floor(scores, relevance, *, weight=.05, beta=.1):
    """Mass objective plus an averaged log-probability shortfall below beta/m.

    Only correct candidates receive a floor, which stops pushing once satisfied.
    beta is the total minimum mass if all m correct candidates meet their floor.
    """
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(beta) or not 0 < beta < 1:
        raise ValueError('Require finite nonnegative weight and 0 < beta < 1')
    logits = scores.squeeze(-1).float() if scores.ndim == 3 else scores.float()
    positive = relevance.unsqueeze(0).expand_as(logits) if relevance.ndim == 1 else relevance
    if logits.ndim != 2 or positive.shape != logits.shape or positive.dtype != torch.bool:
        raise ValueError('Expected boolean candidate labels matching batch logits')
    counts = positive.sum(dim=1)
    if not (counts > 0).all():
        raise ValueError('Every training state needs at least one positive candidate')
    log_probs = logits.log_softmax(dim=1)
    mass = -torch.logsumexp(log_probs.masked_fill(~positive, -torch.inf), dim=1)
    if weight == 0:
        return mass.mean()
    log_floor = math.log(beta) - counts.float().log()
    shortfall = torch.clamp_min(log_floor[:, None] - log_probs, 0.)
    penalty = torch.where(positive, shortfall, 0.).sum(dim=1) / counts
    return (mass + weight * penalty).mean()


class ProbabilityFloorLoss:
    """Named training callable; save numeric settings, never this object, in weights."""
    __name__ = 'probability_mass_with_floor'

    def __init__(self, *, weight, beta=.1):
        self.weight, self.beta = weight, beta

    def __call__(self, scores, relevance):
        return probability_mass_with_floor(scores, relevance, weight=self.weight, beta=self.beta)
