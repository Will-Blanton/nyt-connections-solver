"""Puzzle-local supervised contrastive loss for pre-GINE word features."""
import torch
import torch.nn.functional as F


def puzzle_contrastive_loss(words, *, temperature=0.1):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if words.ndim != 3 or words.shape[1] not in (8, 12, 16):
        raise ValueError('Expected batched, contiguous true groups of four words')
    # The state loader preserves complete groups as consecutive blocks of four.
    n = words.shape[1]
    z = F.normalize(words.float(), dim=-1)
    logits = z @ z.transpose(-1, -2) / temperature
    diagonal = torch.eye(n, device=words.device, dtype=torch.bool)
    log_probs = torch.log_softmax(logits.masked_fill(diagonal, -torch.inf), dim=-1)
    labels = torch.arange(n, device=words.device) // 4
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    # Three positives per anchor; no cross-puzzle negatives or self-positive.
    return -log_probs.masked_fill(~positive, 0).sum(-1).div(3).mean()
