"""Supervised prediction of a complete, balanced Connections partition."""

from itertools import permutations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .models import SetDecoder, SetEncoder


NUM_GROUPS = 4
GROUP_SIZE = 4
NUM_WORDS = NUM_GROUPS * GROUP_SIZE
DEFAULT_WIDTH = 256
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 8
DEFAULT_SCORER_WIDTH = 128
DEFAULT_DROPOUT = 0.3
SLOT_PERMUTATIONS = tuple(permutations(range(NUM_GROUPS)))


class PartitionTransformer(nn.Module):
    """Encode a puzzle, decode four slots, and score word/slot compatibility."""

    def __init__(
        self, in_dim=None, feature_dims=None, width=DEFAULT_WIDTH,
        layers=DEFAULT_LAYERS, heads=DEFAULT_HEADS,
        scorer_width=DEFAULT_SCORER_WIDTH, dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or layers < 1:
            raise ValueError("width must be divisible by heads; layers must be positive")
        if feature_dims is None:
            if in_dim is None:
                raise ValueError("Provide in_dim or feature_dims")
            feature_dims = {"semantic": in_dim}
        elif in_dim is not None:
            raise ValueError("Provide in_dim or feature_dims, not both")
        if not feature_dims or any(dim <= 0 for dim in feature_dims.values()):
            raise ValueError("feature_dims must contain positive dimensions")
        self.feature_dims = dict(feature_dims)
        self.feature_names = tuple(feature_dims)
        self.feature_norms = nn.ModuleDict({
            name: nn.LayerNorm(dim) for name, dim in feature_dims.items()
        })
        self.feature_projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, width), nn.ReLU())
            for name, dim in feature_dims.items()
        })
        self.feature_gate = nn.Linear(width * len(feature_dims), len(feature_dims))
        self.encoder = SetEncoder(
            width, [width] * layers, layers, [heads] * layers, dropout=dropout,
        )
        self.decoder = SetDecoder(
            in_dim=width, h1=width, h2=width, out_dim=width,
            num_seeds=NUM_GROUPS, mab_attn_heads=heads,
            sab_attn_heads=heads, dropout=dropout,
        )
        # Learned compatibility; no assumption of Euclidean cluster geometry.
        self.scorer = nn.Sequential(
            nn.Linear(3 * width, scorer_width), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(scorer_width, 1),
        )

    def forward(self, x):
        if isinstance(x, dict):
            if set(x) != set(self.feature_names):
                raise ValueError("Input feature names must match model feature_dims")
            for name, values in x.items():
                if values.ndim != 3 or values.shape[1:] != (NUM_WORDS, self.feature_dims[name]):
                    raise ValueError(
                        f"Expected {name} inputs shaped (batch, 16, {self.feature_dims[name]})"
                    )
            projected = [
                self.feature_projections[name](self.feature_norms[name](x[name]))
                for name in self.feature_names
            ]
            stacked = torch.stack(projected, dim=2)  # (B, 16, F, H)
            joined = torch.cat(projected, dim=-1)  # (B, 16, F*H)
            gates = self.feature_gate(joined).softmax(dim=-1)  # (B, 16, F)
            mixed = (stacked * gates.unsqueeze(-1)).sum(dim=2)  # (B, 16, H)
        else:
            if x.ndim != 3 or x.shape[1] != NUM_WORDS:
                raise ValueError("Expected inputs shaped (batch, 16, feature_dim)")
            if len(self.feature_names) != 1:
                raise ValueError("Multi-feature models require a feature dictionary")
            name = self.feature_names[0]
            mixed = self.feature_projections[name](self.feature_norms[name](x))
        words = self.encoder(mixed)  # (B, 16, H)
        slots = self.decoder(words)  # (B, 4, H)
        word_pairs = words.unsqueeze(2).expand(-1, -1, NUM_GROUPS, -1)  # (B, 16, 4, H)
        slot_pairs = slots.unsqueeze(1).expand(-1, NUM_WORDS, -1, -1)  # (B, 16, 4, H)
        pairs = torch.cat(
            (word_pairs, slot_pairs, word_pairs * slot_pairs), dim=-1,
        )  # (B, 16, 4, 3H)
        return self.scorer(pairs).squeeze(-1)  # (B, 16, 4)


def _permuted_targets(targets):
    """Every possible alignment of gold group IDs to predicted slot IDs."""
    perms = torch.tensor(SLOT_PERMUTATIONS, device=targets.device)
    return perms[:, targets].permute(1, 0, 2)  # (B, 24, 16)


def partition_loss(logits, targets, reduction="mean"):
    """Minimum word cross-entropy over all 24 gold-to-slot alignments.

    Equivalent to a 4x4 assignment using summed negative log probabilities as
    matching costs. Exhaustive matching keeps this tiny operation on the GPU.
    Gradients flow through the selected alignment's cross-entropy.
    """
    if logits.ndim != 3 or logits.shape[1:] != (NUM_WORDS, NUM_GROUPS):
        raise ValueError("Expected logits shaped (batch, 16, 4)")
    if targets.shape != logits.shape[:2]:
        raise ValueError("Expected targets shaped (batch, 16)")
    mapped = _permuted_targets(targets)  # (B, 24, 16)
    log_probs = logits.log_softmax(-1)  # (B, 16, 4)
    aligned = log_probs.unsqueeze(1).expand(-1, len(SLOT_PERMUTATIONS), -1, -1)
    losses = -aligned.gather(-1, mapped.unsqueeze(-1)).squeeze(-1).mean(-1)
    losses = losses.min(dim=1).values  # (B,)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'mean' or 'none'")


@torch.no_grad()
def decode_partition(logits):
    """Maximize total compatibility with exactly four words in every slot.

    Four copies of each slot turn the capacity constraint into a 16x16 linear
    assignment. Only these small detached score matrices move to CPU.
    """
    if logits.ndim != 3 or logits.shape[1:] != (NUM_WORDS, NUM_GROUPS):
        raise ValueError("Expected logits shaped (batch, 16, 4)")
    assignments = []
    for scores in logits.detach().float().cpu().numpy():
        costs = -np.repeat(scores, GROUP_SIZE, axis=1)  # (16, 16)
        rows, columns = linear_sum_assignment(costs)
        assignment = np.empty(NUM_WORDS, dtype=np.int64)
        assignment[rows] = columns // GROUP_SIZE
        assignments.append(assignment)
    return torch.as_tensor(np.stack(assignments), device=logits.device)  # (B, 16)


@torch.inference_mode()
def evaluate_partitions(model, loader, device):
    """Exact solves, exact groups, and word accuracy after best slot matching."""
    model.to(device).eval()
    count = solved = correct_groups = matched_words = 0
    total_loss = 0.0
    for batch in loader:
        raw_x = batch["x"]
        x = (
            {name: value.to(device) for name, value in raw_x.items()}
            if isinstance(raw_x, dict) else raw_x.to(device)
        )
        targets = batch["targets"].to(device)
        logits = model(x)  # (B, 16, 4)
        predictions = decode_partition(logits)  # (B, 16)
        batch_size = len(targets)
        total_loss += partition_loss(logits, targets, reduction="none").sum().item()
        mapped = _permuted_targets(targets)  # (B, 24, 16)
        best_matches = (predictions[:, None, :] == mapped).sum(-1).max(-1).values
        matched_words += best_matches.sum().item()
        solved += (best_matches == NUM_WORDS).sum().item()
        group_ids = torch.arange(NUM_GROUPS, device=device)
        pred_members = predictions[:, None, :] == group_ids[None, :, None]  # (B, 4, 16)
        true_members = targets[:, None, :] == group_ids[None, :, None]  # (B, 4, 16)
        exact_matches = (
            pred_members[:, :, None, :] == true_members[:, None, :, :]
        ).all(-1)  # (B, 4, 4)
        correct_groups += exact_matches.any(-1).sum().item()
        count += batch_size
    if not count:
        raise ValueError("No puzzles selected for evaluation")
    return {
        "puzzles": count, "solved": solved, "solve_rate": solved / count,
        "mean_correct_groups": correct_groups / count,
        "matched_word_accuracy": matched_words / (NUM_WORDS * count),
        "loss": total_loss / count,
    }
