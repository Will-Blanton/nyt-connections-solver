"""Training primitives for the selected candidate-local GINE baseline.

Only pretrained Qwen vectors are frozen. There is no curriculum, transferred
adapter, auxiliary contrastive task, scheduler, or checkpoint selection on test.
"""
import itertools
import math
import random
from collections import Counter
from time import perf_counter

import numpy as np
import torch

from .graph_models import StandaloneCandidateGINE


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def validate_splits(splits):
    """Require explicit, nonoverlapping puzzle-date splits; never infer from CSV."""
    if set(splits) != {'train', 'val', 'test'} or not splits['train']:
        raise ValueError('Require train/val/test date lists and a nonempty train split')
    seen = set()
    for dates in splits.values():
        if not isinstance(dates, list) or not all(isinstance(d, str) for d in dates):
            raise ValueError('Split dates must be lists of strings')
        if len(set(dates)) != len(dates) or seen.intersection(dates):
            raise ValueError('Split dates repeat or overlap')
        seen.update(dates)


def candidate_artifacts():
    """Enumerate quartets and labels for canonical 8/12/16-word states."""
    artifacts = {}
    for size in (2, 3, 4):
        candidates = list(itertools.combinations(range(size * 4), 4))
        positives = {tuple(range(group * 4, group * 4 + 4)) for group in range(size)}
        artifacts[size] = {
            'idx': torch.tensor(candidates, dtype=torch.long),
            'y': torch.tensor([c in positives for c in candidates], dtype=torch.bool),
        }
    return artifacts


def mixed_state_schedule(seed, updates):
    """Shuffle each block of 2/3/4-group states; equal exposure from the start."""
    if updates < 3 or updates % 3:
        raise ValueError('Updates must be positive and divisible by three')
    rng = random.Random(seed)
    schedule = []
    for _ in range(updates // 3):
        block = [2, 3, 4]
        rng.shuffle(block)
        schedule.extend(block)
    return schedule


def probability_mass_with_floor(scores, relevance, *, weight=.1, beta=.1):
    """Negative log total correct mass plus the selected small positive floor.

    Each correct candidate gets a log-probability shortfall penalty only below
    beta / number_of_correct_candidates. This is the final baseline loss.
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


def build_model(config, group_idx, seed):
    seed_everything(seed)
    model = StandaloneCandidateGINE(group_idx=group_idx, **config)
    for module in (model.input_norm, model.proj, model.edge_encoder, model.group_readout):
        if not all(p.requires_grad for p in module.parameters()):
            raise ValueError('Baseline requires all active learned layers to be trainable')
    return model


def train_fixed(model, loaders, artifacts, device, *, seed, updates, lr,
                weight_decay, floor_weight=.1, floor_beta=.1, log_every=500,
                callback=None):
    """Run a fixed budget and return final weights; no validation/test access."""
    if set(loaders) != {2, 3, 4} or any(len(loader) == 0 for loader in loaders.values()):
        raise ValueError('Require nonempty loaders for all three state sizes')
    if log_every < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Invalid optimizer or logging settings')
    schedule = mixed_state_schedule(seed, updates)
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=weight_decay)
    # Matches the selected refit: gradient scaling, but no autocast.
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    labels = {s: {k: v.to(device) for k, v in a.items()} for s, a in artifacts.items()}
    iterators = {s: iter(loader) for s, loader in loaders.items()}
    history = []
    running = count = 0
    started = perf_counter()
    for step, size in enumerate(schedule, 1):
        try:
            batch = next(iterators[size])
        except StopIteration:
            iterators[size] = iter(loaders[size])
            batch = next(iterators[size])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(batch['x'].to(device), group_idx=labels[size]['idx'],
                       edge_features=batch['edge_features'].to(device))
        loss = probability_mass_with_floor(scores, labels[size]['y'],
                                           weight=floor_weight, beta=floor_beta)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite training loss at update {step}')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        count += 1
        if step % log_every == 0 or step == updates:
            row = dict(update=step, train_loss=running/count, lr=lr,
                       seconds=perf_counter()-started)
            history.append(row)
            print(f"Update {step}/{updates} | train loss {row['train_loss']:.4f}", flush=True)
            if callback is not None:
                callback(row)
            running = count = 0
            started = perf_counter()
    return model.cpu().eval(), dict(
        updates_ran=updates, history=history, state_updates=dict(Counter(schedule)),
        checkpoint_rule='final fixed update; no validation or test selection',
        loss='probability_mass_with_floor', floor_weight=floor_weight, floor_beta=floor_beta)
