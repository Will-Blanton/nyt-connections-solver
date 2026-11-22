#!/usr/bin/env python3
"""Run one controlled graph-feature ablation outside the notebooks.

The runner reuses the date split and base hyperparameters from a reference graph
checkpoint. It trains one variant at a time, chooses the checkpoint by validation
full-board (s4) Hit@1 with s4 MRR as a tie-breaker, and never reads test examples.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nyt_connections_solver.data_loader import (  # noqa: E402
    collate_keep_wordlists,
    make_state_loaders,
)
from nyt_connections_solver.edge_features import EdgeFeatureComposer, ClosedCompoundCompletions  # noqa: E402
from nyt_connections_solver.graph_features import build_graph_features  # noqa: E402
from nyt_connections_solver.graph_models import GraphSetTransformer, GatedEdgeResidual, GatedGINEResidual, StandaloneCandidateGINE  # noqa: E402
from nyt_connections_solver.node_features import TextSpellingFeatures  # noqa: E402


VARIANTS = {
    "standalone-gine-adapter-single": {
        "description": "One learned 1024-to-1024 Linear/ReLU residual adapter, trainable norm and GINE.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-adapter-0": {
        "description": "Adapter control: trainable input norm and direct GINE, no adapter contribution.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-adapter-16": {
        "description": "Residual embedding adapter with bottleneck 16, trainable norm and GINE; no transferred layers.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-adapter-64": {
        "description": "Residual embedding adapter with bottleneck 64, trainable norm and GINE; no transferred layers.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-scratch-512": {
        "description": "Fresh trainable 1024-to-512 projection and input norm; no transferred layers.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-scratch-64": {
        "description": "Fresh trainable 1024-to-64 projection and input norm; no transferred layers.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-scratch-128": {
        "description": "Fresh trainable 1024-to-128 projection and input norm; no transferred layers.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-scratch-projected-frozen-norm": {
        "description": "Fresh trainable 1024-to-256 projection with its fresh input norm frozen.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-random-frozen": {
        "description": "GINE with a randomly initialized, frozen 1024-to-256 projection and input norm.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-raw": {
        "description": "GINE learns a direct 1024-to-32 projection of frozen text embeddings; no transferred layers.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-scratch-projected": {
        "description": "GINE learns a fresh 1024-to-256 projection and input norm; no transferred layers.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-trainable-projection": {
        "description": "Standalone four-word GINE scorer with trainable text projection and all original edges.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "standalone-gine-projected": {
        "description": "Standalone four-word GINE scorer with frozen text projection and all original edges.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-projected-margin": {
        "description": "Projected GINE with relative margin against three-word-overlap incorrect sets.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-projected-compounds": {
        "description": "Projected GINE with the original edges and shared closed-compound completions.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-projected-compounds-control": {
        "description": "Matched projected GINE with closed-compound channels masked.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-projected": {
        "description": "GINE branch receives projected text before global or candidate attention.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-projected-control": {
        "description": "Projected-text GINE branch with zero edge inputs.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-global": {
        "description": "GINE branch receives globally encoded words before candidate attention.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-global-control": {
        "description": "Global-text GINE branch with zero edge inputs.",
        "use_spelling": False, "edge_families": "all", "use_edges": False,
    },
    "gated-gine-mlp-edge-control": {
        "description": "Nonlinear-edge GINE with all raw edge inputs zeroed.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "gated-gine-mlp-edges": {
        "description": "GINE with a two-layer nonlinear edge encoder per message-passing layer.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "gated-gine-zero-edge-control": {
        "description": "Frozen text scorer plus identical candidate-local GINE with all edges zeroed.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "gated-gine-residual": {
        "description": "Frozen text scorer plus two candidate-local GINE layers and gated correction.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "gated-zero-edge-control": {
        "description": "Frozen text scorer plus identical gated branch with all edge inputs zeroed.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "gated-linguistic-residual": {
        "description": "Frozen text scorer plus gated correction using twelve spelling/phrase/fragment channels.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "gated-semantic-residual": {
        "description": "Frozen text scorer plus gated correction using semantic/WordNet channels only.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "gated-edge-residual": {
        "description": "Frozen text scorer plus pooled six-pair edge encoder and gated correction.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "text-only": {
        "description": "Frozen text nodes; edge inputs are present but disabled.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": False,
    },
    "text-spelling": {
        "description": "Frozen text plus spelling nodes; edge inputs are disabled.",
        "use_spelling": True,
        "edge_families": "all",
        "use_edges": False,
    },
    "text-semantic-edges": {
        "description": "Frozen text nodes with semantic and WordNet edge channels.",
        "use_spelling": False,
        "edge_families": "semantic",
        "use_edges": True,
    },
    "text-all-edges": {
        "description": "Frozen text nodes with all 15 edge channels.",
        "use_spelling": False,
        "edge_families": "all",
        "use_edges": True,
    },
    "full": {
        "description": "Frozen text plus spelling nodes and all 15 edge channels.",
        "use_spelling": True,
        "edge_families": "all",
        "use_edges": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--evaluate-only", type=Path,
                        help="Evaluate an existing experiment checkpoint without training.")
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path("weights/graph_set_transformer_optuna_text_spelling_binary_20261018_213815.pt"),
        help="Checkpoint supplying the frozen date split, embedding revision, and base hyperparameters.",
    )
    parser.add_argument("--dataset", type=Path, help="Override the dataset saved in the checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/graph-features"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/feature_cache/graph"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--min-updates-before-stop", type=int, default=700)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, help="Override the reference learning rate.")
    parser.add_argument("--projection-lr", type=float,
                        help="Separate learning rate for the trainable standalone text projection.")
    parser.add_argument("--weight-decay", type=float, help="Override the reference weight decay.")
    parser.add_argument("--near-miss-weight", type=float, default=0.25)
    parser.add_argument("--near-miss-margin", type=float, default=0.5)
    parser.add_argument("--diagnose-gradients", action="store_true")
    parser.add_argument("--aux-weight", type=float, default=0.)
    parser.add_argument("--cosine-lr", action="store_true",
                        help="Decay the learning rate of all trainable layers to zero over the update budget.")
    parser.add_argument("--aux-anneal-steps", type=int, default=0,
                        help="Fade the auxiliary loss weight to zero at this update.")
    parser.add_argument("--aux-temperature", type=float, default=0.1)
    parser.add_argument("--encoder-anneal-steps", type=int, default=0,
                        help="Linearly decay input LayerNorm/projection LR to zero at this update.")
    parser.add_argument("--train-input-norm", action="store_true",
                        help="Also train input LayerNorm in the unfrozen pretrained-projection diagnostic.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_connections(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    corrections = {
        1298: "['line', 'plane', 'point', 'solid']",
        1892: "['abyss', 'fly', 'matrix', 'thing']",
    }
    for row, value in corrections.items():
        if row in frame.index:
            frame.loc[row, "connections"] = value
    frame = frame[~frame["date"].astype(str).str.contains("04-01")].copy()
    frame["date"] = frame["date"].astype(str)
    frame["connections"] = frame["connections"].apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else value
    )
    return frame.reset_index(drop=True)


def make_artifacts() -> dict[int, dict[str, torch.Tensor]]:
    artifacts = {}
    for size in (2, 3, 4):
        candidates = list(itertools.combinations(range(size * 4), 4))
        true_groups = {tuple(range(group * 4, group * 4 + 4)) for group in range(size)}
        relevance = [int(candidate in true_groups) for candidate in candidates]
        artifacts[size] = {
            "idx": torch.tensor(candidates, dtype=torch.long),
            "y": torch.tensor(relevance, dtype=torch.bool),
        }
        true_groups_tensor = artifacts[size]["idx"][artifacts[size]["y"]]
        overlaps = (artifacts[size]["idx"][:, None, :, None] ==
                    true_groups_tensor[None, :, None, :]).any(dim=-1).sum(dim=-1)
        artifacts[size]["near_miss"] = (overlaps == 3).any(dim=1) & ~artifacts[size]["y"]
    return artifacts


def state_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    logits = scores.squeeze(-1).float()
    positive = relevance.unsqueeze(0).expand(len(logits), -1) if relevance.ndim == 1 else relevance
    if positive.shape != logits.shape or positive.dtype != torch.bool:
        raise ValueError('Relevance must be boolean candidate labels matching scores')
    if not positive.any(dim=1).all():
        raise ValueError('Every training state needs at least one positive candidate')
    log_probs = torch.log_softmax(logits, dim=1)
    masked = torch.where(positive, log_probs, torch.full_like(log_probs, float("-inf")))
    return -torch.logsumexp(masked, dim=1).mean()


def relative_near_miss_loss(scores, relevance, near_miss, *, margin):
    logits = scores.squeeze(-1).float()
    positives = logits[:, relevance].max(dim=1).values
    negatives = logits[:, near_miss].max(dim=1).values
    return F.softplus(negatives - positives + margin).mean()


def ranking_metrics(scores: torch.Tensor, relevance: torch.Tensor) -> dict[str, torch.Tensor]:
    logits = scores.squeeze(-1).float()
    positive = relevance.unsqueeze(0).expand(len(logits), -1)
    order = logits.argsort(dim=1, descending=True)
    ranked = positive.gather(1, order)
    first_rank = ranked.float().argmax(dim=1) + 1
    ranks = torch.arange(1, logits.shape[1] + 1, device=logits.device, dtype=torch.float)
    precision = ranked.float().cumsum(dim=1) / ranks
    average_precision = (precision * ranked).sum(dim=1) / positive.sum(dim=1)
    cutoff = min(4, logits.shape[1])
    discounts = torch.log2(torch.arange(2, cutoff + 2, device=logits.device, dtype=torch.float))
    dcg = (ranked[:, :cutoff].float() / discounts).sum(dim=1)
    ideal_count = torch.minimum(positive.sum(dim=1), torch.tensor(cutoff, device=logits.device))
    ideal = torch.stack([
        (torch.ones(int(count.item()), device=logits.device) / discounts[: int(count.item())]).sum()
        for count in ideal_count
    ])
    return {
        "hit1": ranked[:, 0].float(),
        "mrr": first_rank.float().reciprocal(),
        "hit4": ranked[:, :cutoff].any(dim=1).float(),
        "map": average_precision,
        "ndcg4": dcg / ideal,
    }


def score_batch(
    model: GraphSetTransformer,
    batch: dict,
    group_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    from nyt_connections_solver.candidate_features import candidate_edge_kwargs
    return model(
        batch["x"].to(device),
        group_idx=group_idx,
        edge_features=batch["edge_features"].to(device),
        **candidate_edge_kwargs(batch, device),
    )


def evaluate(
    model: GraphSetTransformer,
    loaders: dict[int, DataLoader],
    artifacts: dict[int, dict[str, torch.Tensor]],
    device: torch.device,
    *, by_color: bool = False,
) -> dict[int, dict[str, float]]:
    model.eval()
    results = {}
    with torch.inference_mode():
        for size, loader in loaders.items():
            group_idx = artifacts[size]["idx"].to(device)
            relevance = artifacts[size]["y"].to(device)
            totals = {key: 0.0 for key in ("hit1", "mrr", "hit4", "map", "ndcg4")}
            count = 0
            color_rows = {color: [] for color in ("purple", "blue", "green", "yellow")}
            for batch in loader:
                scores = score_batch(model, batch, group_idx, device)
                metrics = ranking_metrics(scores, relevance)
                for key, values in metrics.items():
                    totals[key] += values.sum().item()
                if by_color:
                    # True groups occupy consecutive blocks; source_group_idx
                    # identifies their original CSV rows, including partial states.
                    logits = scores.squeeze(-1)
                    order = logits.argsort(dim=1, descending=True)
                    ranks = order.argsort(dim=1) + 1
                    targets = relevance.nonzero().flatten()
                    target_ranks = ranks[:, targets].cpu()
                    for source_groups, row in zip(batch["source_group_idx"], target_ranks):
                        for source, rank in zip(source_groups.tolist(), row.tolist()):
                            color_rows[("purple", "blue", "green", "yellow")[source]].append(rank)
                count += len(scores)
            if count == 0:
                raise ValueError(f"Validation loader s{size} is empty")
            results[size] = {key: value / count for key, value in totals.items()}
            if by_color:
                results[size]["by_color"] = {
                    color: {
                        "groups": len(rows),
                        "mean_rank": float(np.mean(rows)),
                        "mrr": float(np.mean([1 / rank for rank in rows])),
                        "top1": float(np.mean([rank == 1 for rank in rows])),
                        "top4": float(np.mean([rank <= 4 for rank in rows])),
                    }
                    for color, rows in color_rows.items() if rows
                }
    return results


def next_batch(iterators: dict, loaders: dict, size: int) -> dict:
    try:
        return next(iterators[size])
    except StopIteration:
        iterators[size] = iter(loaders[size])
        return next(iterators[size])


def train(
    model: GraphSetTransformer,
    train_loaders: dict[int, DataLoader],
    val_loaders: dict[int, DataLoader],
    artifacts: dict[int, dict[str, torch.Tensor]],
    device: torch.device,
    *,
    lr: float,
    projection_lr: float | None,
    weight_decay: float,
    updates: int,
    eval_every: int,
    min_updates_before_stop: int,
    patience: int,
    min_delta: float,
    near_miss_weight: float = 0.0,
    near_miss_margin: float = 0.5,
    diagnose_gradients: bool = False,
    aux_weight: float = 0.,
    aux_temperature: float = 0.1,
    encoder_anneal_steps: int = 0,
    aux_anneal_steps: int = 0,
    cosine_lr: bool = False,
    evaluation_callback=None,
    state_schedule=None,
    candidate_schedule=None,
    training_artifacts=None,
    ranking_loss=None,
) -> tuple[GraphSetTransformer, dict]:
    model = model.to(device)
    if diagnose_gradients:
        from graph_training_probe import TrainingProbe
        probe = TrainingProbe(model)
    else:
        probe = None
    if encoder_anneal_steps:
        from graph_training_schedule import annealed_encoder_lr
        encoder_ids = {id(p) for m in (model.input_norm, model.proj) for p in m.parameters()}
        parameter_groups = [
            {'params': [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]},
            {'params': [p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids],
             'lr': lr},
        ]
    elif projection_lr is not None:
        if not isinstance(model, StandaloneCandidateGINE) or not model.train_text_projection:
            raise ValueError("--projection-lr requires the trainable standalone projection variant")
        projection_ids = {id(p) for p in model.proj.parameters()}
        parameter_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in projection_ids]},
            {"params": [p for p in model.proj.parameters() if p.requires_grad], "lr": projection_lr},
        ]
    else:
        parameter_groups = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    sizes = tuple(sorted(train_loaders))
    if state_schedule is not None:
        state_schedule = tuple(state_schedule)
        if len(state_schedule) != updates or not set(state_schedule).issubset(sizes):
            raise ValueError('State schedule must supply one available state size per update')
    if candidate_schedule is not None:
        if near_miss_weight or aux_weight:
            raise ValueError('Candidate-size curriculum currently supports the ranking loss alone')
        candidate_schedule = tuple(candidate_schedule)
        if len(candidate_schedule) != updates or training_artifacts is None:
            raise ValueError('Candidate schedule requires one size per update and training artifacts')
        if state_schedule is None or any((state, candidate) not in training_artifacts
                                        for state, candidate in zip(state_schedule, candidate_schedule)):
            raise ValueError('Every scheduled board/candidate size needs training artifacts')
    iterators = {size: iter(loader) for size, loader in train_loaders.items()}
    artifacts_on_device = {
        size: {key: value.to(device) for key, value in values.items()}
        for size, values in artifacts.items()
    }
    training_artifacts_on_device = {
        key: {name: value.to(device) for name, value in values.items()}
        for key, values in (training_artifacts or {}).items()
    }
    pending_sizes: list[int] = []
    best_key = (-float("inf"), -float("inf"))
    best_state = None
    best_update = 0
    baseline_metrics = None
    if isinstance(model, GatedEdgeResidual):
        baseline_metrics = evaluate(model, val_loaders, artifacts, device, by_color=True)
        best_key = (baseline_metrics[4]["hit1"], baseline_metrics[4]["mrr"])
        best_state = {name: value.detach().cpu().clone()
                      for name, value in model.state_dict().items()}
        print(f"Update 0 | baseline s4 Hit@1 {best_key[0]:.4f}, MRR {best_key[1]:.4f}", flush=True)
    patience_reference = -float("inf")
    stale_evals = 0
    history = []
    running_loss = 0.0
    running_rank_loss = running_aux_loss = 0.0
    interval_updates = 0
    interval_started = perf_counter()
    frozen_encoder_snapshot = None

    for step in range(1, updates + 1):
        if cosine_lr:
            from graph_training_schedule import cosine_to_zero
            for group in optimizer.param_groups:
                group['lr'] = cosine_to_zero(step, lr, updates)
        if encoder_anneal_steps:
            optimizer.param_groups[1]['lr'] = annealed_encoder_lr(step, lr, encoder_anneal_steps)
            if step == encoder_anneal_steps:
                frozen_encoder_snapshot = [p.detach().clone() for m in (model.input_norm, model.proj)
                                           for p in m.parameters()]
        if state_schedule is not None:
            size = state_schedule[step - 1]
        else:
            if not pending_sizes:
                pending_sizes = list(sizes)
                random.shuffle(pending_sizes)
            size = pending_sizes.pop()
        batch = next_batch(iterators, train_loaders, size)
        candidate_size = candidate_schedule[step - 1] if candidate_schedule is not None else 4
        artifact = (training_artifacts_on_device[(size, candidate_size)]
                    if candidate_schedule is not None else artifacts_on_device[size])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sampled = probe is not None and (step in (1, 10) or step % eval_every == 0)
        if sampled:
            probe.begin(step, size)
        scores = score_batch(model, batch, artifact["idx"], device)
        custom_relevance = isinstance(batch, dict) and 'relevance' in batch
        relevance = batch['relevance'].to(device) if custom_relevance else artifact['y']
        if custom_relevance:
            if near_miss_weight or aux_weight:
                raise ValueError('Custom relevance requires sample-specific auxiliary/near-miss labels')
        loss = (ranking_loss or state_loss)(scores, relevance)
        rank_loss = loss
        effective_aux_weight = aux_weight
        if aux_anneal_steps:
            from graph_training_schedule import linear_to_zero
            effective_aux_weight = linear_to_zero(step, aux_weight, aux_anneal_steps)
        if effective_aux_weight:
            from graph_auxiliary_loss import puzzle_contrastive_loss
            word_features = model.proj(model.input_norm(batch['x'].to(device)))
            aux_loss = puzzle_contrastive_loss(word_features, temperature=aux_temperature)
            loss = loss + effective_aux_weight * aux_loss
            running_aux_loss += aux_loss.item()
        running_rank_loss += rank_loss.item()
        if near_miss_weight:
            loss = loss + near_miss_weight * relative_near_miss_loss(
                scores, artifact["y"], artifact["near_miss"], margin=near_miss_margin)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at update {step}")
        scaler.scale(loss).backward()
        if sampled:
            scaler.unscale_(optimizer)
            probe.before_step()
        scaler.step(optimizer)
        scaler.update()
        if sampled:
            probe.after_step(loss)
        running_loss += loss.item()
        interval_updates += 1

        if step % eval_every != 0 and step != updates:
            continue

        val_started = perf_counter()
        metrics = evaluate(model, val_loaders, artifacts, device)
        val_seconds = perf_counter() - val_started
        key = (metrics[4]["hit1"], metrics[4]["mrr"])
        if key > best_key:
            best_key = key
            best_update = step
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        if metrics[4]["hit1"] > patience_reference + min_delta:
            patience_reference = metrics[4]["hit1"]
            stale_evals = 0
        else:
            stale_evals += 1
        row = {
            "update": step,
            "state_size": size,
            "candidate_size": candidate_size,
            "train_loss": running_loss / interval_updates,
            "rank_loss": running_rank_loss / interval_updates,
            "aux_loss": running_aux_loss / interval_updates,
            "aux_weight": effective_aux_weight,
            "encoder_lr": optimizer.param_groups[1]['lr'] if encoder_anneal_steps else None,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "train_seconds": perf_counter() - interval_started - val_seconds,
            "validation_seconds": val_seconds,
            "best_update": best_update,
            **{
                f"val_s{size}_{metric}": value
                for size, state_metrics in metrics.items()
                for metric, value in state_metrics.items()
            },
        }
        history.append(row)
        print(
            f"Update {step}/{updates} | loss {row['train_loss']:.4f} | "
            f"s4 Hit@1 {metrics[4]['hit1']:.4f}, MRR {metrics[4]['mrr']:.4f} | "
            f"s3 Hit@1 {metrics[3]['hit1']:.4f} | s2 Hit@1 {metrics[2]['hit1']:.4f} | "
            f"best s4 {best_key[0]:.4f} @ {best_update}",
            flush=True,
        )
        if evaluation_callback is not None:
            evaluation_callback(step, metrics, best_key, row)
        running_loss = 0.0
        running_rank_loss = running_aux_loss = 0.0
        interval_updates = 0
        interval_started = perf_counter()
        if step >= min_updates_before_stop and stale_evals >= patience:
            print(f"Early stopping at update {step}; restoring update {best_update}.", flush=True)
            break

    if probe is not None:
        probe.close()
    if frozen_encoder_snapshot is not None:
        encoder_parameters = [p for m in (model.input_norm, model.proj) for p in m.parameters()]
        if not all(torch.equal(p.detach(), saved) for p, saved in zip(encoder_parameters, frozen_encoder_snapshot)):
            raise RuntimeError('Encoder changed after its learning rate reached zero')
    if best_state is None:
        raise ValueError("No validation checkpoint was recorded")
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loaders, artifacts, device, by_color=True)
    summary = {
        "ranking_objective": getattr(ranking_loss or state_loss, '__name__', type(ranking_loss or state_loss).__name__),
        "selection_metric": "validation_s4_hit1",
        "tie_break_metric": "validation_s4_mrr",
        "best_update": best_update,
        "updates_ran": history[-1]["update"],
        "validation": {f"s{size}": values for size, values in final_metrics.items()},
        "validation_macro": {
            metric: float(np.mean([final_metrics[size][metric] for size in final_metrics]))
            for metric in ("hit1", "mrr", "hit4", "map", "ndcg4")
        },
        "history": history,
        "baseline_validation": baseline_metrics,
        "gradient_diagnostics": probe.records if probe is not None else None,
        "encoder_frozen_verified": frozen_encoder_snapshot is not None,
    }
    return model.to("cpu").eval(), summary


def main() -> None:
    args = parse_args()
    if args.cosine_lr and (args.updates < 2 or args.encoder_anneal_steps or args.projection_lr is not None):
        raise ValueError('Global cosine schedule requires >=2 updates and no separate encoder/projection schedule')
    if args.aux_weight < 0 or args.aux_temperature <= 0:
        raise ValueError('Auxiliary weight must be nonnegative and temperature positive')
    if args.aux_weight and args.variant != 'standalone-gine-scratch-projected':
        raise ValueError('Auxiliary experiment requires the fully trainable scratch projection variant')
    if args.aux_anneal_steps and (
        not args.aux_weight or args.aux_anneal_steps < 2 or args.aux_anneal_steps > args.updates
    ):
        raise ValueError('Auxiliary annealing requires positive aux weight and 2 <= stop <= updates')
    if args.encoder_anneal_steps and (
        args.encoder_anneal_steps < 2 or args.encoder_anneal_steps > args.updates
        or args.variant != 'standalone-gine-scratch-projected'
        or args.projection_lr is not None or args.aux_weight
    ):
        raise ValueError('Encoder annealing requires scratch-projected, 2 <= stop <= updates, no auxiliary or separate projection LR')
    if args.updates < 1 or args.eval_every < 1 or args.patience < 1:
        raise ValueError("updates, eval-every, and patience must be positive")
    started = perf_counter()
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    reference_path = (args.evaluate_only or args.reference_checkpoint).resolve()
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if reference.get("model_class") not in {"GraphSetTransformer", "GatedEdgeResidual", "GatedGINEResidual"}:
        raise ValueError("Reference must be a GraphSetTransformer checkpoint")
    split_dates = reference.get("split_dates")
    if not split_dates or not split_dates.get("train") or not split_dates.get("val"):
        raise ValueError("Reference checkpoint must contain train and validation dates")
    dataset_path = (args.dataset or Path(reference["dataset_path"])).resolve()
    frame = load_connections(dataset_path)
    train_frame = frame[frame["date"].isin(split_dates["train"])]
    val_frame = frame[frame["date"].isin(split_dates["val"])]
    if train_frame["date"].nunique() != len(split_dates["train"]):
        raise ValueError("Dataset does not reproduce the reference training split")
    if val_frame["date"].nunique() != len(split_dates["val"]):
        raise ValueError("Dataset does not reproduce the reference validation split")

    variant = VARIANTS[args.variant]
    if args.evaluate_only and reference["training_config"].get("variant") != args.variant:
        raise ValueError("--variant must match the evaluated checkpoint")
    options = reference["feature_metadata"]["options"]
    embedder = SentenceTransformer(
        reference["embedding_model_name"], revision=options["revision"], device=str(device)
    )
    _, full_edges, _ = build_graph_features(
        embedder,
        reference["embedding_model_name"],
        cache_dir=args.cache_dir,
        revision=options["revision"],
        char_dim=options["char_dim"],
        use_spelling=True,
        include_wordnet=options["include_wordnet"],
        min_fragment_length=options["min_fragment_length"],
    )
    nodes = TextSpellingFeatures(
        full_edges.providers[0].embeddings,
        char_dim=options["char_dim"],
        use_spelling=variant["use_spelling"],
    )
    providers = full_edges.providers
    if variant["edge_families"] == "semantic":
        providers = providers[:2]
    if args.variant in {"gated-gine-projected-compounds", "gated-gine-projected-compounds-control"}:
        providers = (*providers, ClosedCompoundCompletions(providers[1].lexicon))
    edges = EdgeFeatureComposer(providers, cache=full_edges.cache)

    print(
        f"Variant: {args.variant} | device: {device} | nodes: {nodes.dim} | "
        f"edges: {edges.dim} | train/val dates: {len(split_dates['train'])}/{len(split_dates['val'])}",
        flush=True,
    )
    datasets = {}
    for name, subset in (("train", train_frame), ("val", val_frame)):
        if args.evaluate_only and name == "train":
            continue
        datasets[name], _ = make_state_loaders(
            subset,
            embedder,
            None,
            args.batch_size,
            shuffle=False,
            feature_builder=nodes,
            edge_providers=edges,
            cache=edges.cache,
        )
    train_loaders = {
        size: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_keep_wordlists,
            generator=torch.Generator().manual_seed(args.seed + size),
        )
        for size, dataset in datasets.get("train", {}).items()
    }
    val_loaders = {
        size: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_keep_wordlists,
        )
        for size, dataset in datasets["val"].items()
    }
    artifacts = make_artifacts()
    model_config = deepcopy(reference["model_config"])
    model_config.update(
        in_dim=nodes.dim,
        edge_dim=edges.dim,
        edge_bias_scale=(reference["model_config"]["edge_bias_scale"] if variant["use_edges"] else 0.0),
    )
    lr = args.lr if args.lr is not None else reference["training_config"]["lr"]
    weight_decay = (args.weight_decay if args.weight_decay is not None
                    else reference["training_config"]["weight_decay"])
    seed_everything(args.seed)
    is_residual = args.variant in {"gated-edge-residual", "gated-semantic-residual",
                                   "gated-linguistic-residual", "gated-zero-edge-control",
                                   "gated-gine-residual", "gated-gine-zero-edge-control",
                                   "gated-gine-mlp-edge-control", "gated-gine-mlp-edges",
                                   "gated-gine-projected", "gated-gine-projected-control",
                                   "gated-gine-projected-margin",
                                   "gated-gine-projected-compounds", "gated-gine-projected-compounds-control",
                                   "gated-gine-global", "gated-gine-global-control"}
    standalone_variants = {"standalone-gine-projected", "standalone-gine-trainable-projection",
                           "standalone-gine-adapter-0", "standalone-gine-adapter-16", "standalone-gine-adapter-64",
                           "standalone-gine-adapter-single",
                           "standalone-gine-scratch-projected", "standalone-gine-scratch-projected-frozen-norm",
                           "standalone-gine-scratch-64", "standalone-gine-scratch-128",
                           "standalone-gine-scratch-512",
                           "standalone-gine-random-frozen",
                           "standalone-gine-raw"}
    is_residual = is_residual or args.variant in standalone_variants
    if args.variant in {"gated-zero-edge-control", "gated-gine-zero-edge-control",
                        "gated-gine-mlp-edge-control", "gated-gine-projected-control",
                        "gated-gine-global-control"}:
        model_config["residual_edge_channels"] = []
    if args.variant in {"gated-semantic-residual", "gated-linguistic-residual"}:
        semantic_only = args.variant == "gated-semantic-residual"
        model_config["residual_edge_channels"] = [
            index for index, name in enumerate(edges.feature_names)
            if (name in {"semantic_cosine", "sense_jaccard", "rare_concept_jaccard"}) == semantic_only
        ]
    if args.variant == "gated-gine-projected-compounds-control":
        model_config["residual_edge_channels"] = list(range(edges.dim - 2))
    model_class = GatedEdgeResidual if is_residual else GraphSetTransformer
    if args.variant in {"gated-gine-residual", "gated-gine-zero-edge-control",
                        "gated-gine-mlp-edge-control", "gated-gine-mlp-edges",
                        "gated-gine-projected", "gated-gine-projected-control",
                        "gated-gine-projected-margin",
                        "gated-gine-projected-compounds", "gated-gine-projected-compounds-control",
                        "gated-gine-global", "gated-gine-global-control"}:
        model_class = GatedGINEResidual
        model_config.update(edge_hidden=32, gine_layers=2)
        if args.variant in {"gated-gine-mlp-edge-control", "gated-gine-mlp-edges"}:
            model_config["nonlinear_edges"] = True
        if args.variant in {"gated-gine-projected", "gated-gine-projected-control", "gated-gine-projected-margin",
                            "gated-gine-projected-compounds", "gated-gine-projected-compounds-control"}:
            model_config["gine_input_stage"] = "projected"
        if args.variant in {"gated-gine-global", "gated-gine-global-control"}:
            model_config["gine_input_stage"] = "global"
    if args.variant in standalone_variants:
        model_class = StandaloneCandidateGINE
        model_config.update(edge_hidden=32, gine_layers=2)
        if args.variant == "standalone-gine-adapter-single":
            model_config.update(single_layer_adapter=True, train_input_norm=True)
        elif args.variant.startswith("standalone-gine-adapter-"):
            model_config.update(adapter_hidden=int(args.variant.rsplit("-", 1)[-1]), train_input_norm=True)
        if args.variant in {"standalone-gine-trainable-projection", "standalone-gine-scratch-projected",
                            "standalone-gine-scratch-projected-frozen-norm",
                            "standalone-gine-scratch-64", "standalone-gine-scratch-128",
                            "standalone-gine-scratch-512"}:
            model_config["train_text_projection"] = True
        if args.variant in {"standalone-gine-scratch-projected", "standalone-gine-scratch-64",
                            "standalone-gine-scratch-128", "standalone-gine-scratch-512"}:
            model_config["train_input_norm"] = True
        if args.variant in {"standalone-gine-scratch-64", "standalone-gine-scratch-128",
                            "standalone-gine-scratch-512"}:
            model_config["projection_width"] = int(args.variant.rsplit("-", 1)[-1])
        if args.variant == "standalone-gine-raw":
            model_config["raw_text_embeddings"] = True
    if args.train_input_norm:
        if args.variant != "standalone-gine-trainable-projection":
            raise ValueError("--train-input-norm requires the trainable projection variant")
        model_config['train_input_norm'] = True
    model = model_class(group_idx=artifacts[4]["idx"], **model_config)
    if is_residual and not args.evaluate_only and args.variant not in {
        "standalone-gine-adapter-0", "standalone-gine-adapter-16", "standalone-gine-adapter-64",
        "standalone-gine-adapter-single",
        "standalone-gine-scratch-projected", "standalone-gine-scratch-projected-frozen-norm",
        "standalone-gine-scratch-64", "standalone-gine-scratch-128",
        "standalone-gine-scratch-512",
        "standalone-gine-random-frozen", "standalone-gine-raw"
    }:
        if reference["training_config"].get("variant") != "text-only":
            raise ValueError("Supply the trained text-only experiment as --reference-checkpoint")
        base_state = reference["model_state_dict"]
        current_state = model.state_dict()
        resized = [name for name, value in base_state.items()
                   if name in current_state and value.shape != current_state[name].shape]
        if resized and (not args.variant.startswith("gated-gine-projected-compounds") or
                        any("edge_projection.weight" not in name for name in resized)):
            raise ValueError(f"Unexpected reference tensor dimensions: {resized}")
        base_state = {name: value for name, value in base_state.items() if name not in resized}
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        if unexpected or any(not name.startswith(("edge_encoder.", "edge_gate.", "edge_correction.", "group_readout."))
                             and name not in resized for name in missing):
            raise ValueError("Text-only checkpoint does not match the frozen base")
    if args.evaluate_only:
        model.load_state_dict(reference["model_state_dict"])
        metrics = evaluate(model.to(device), val_loaders, artifacts, device, by_color=True)
        output_path = args.output_dir / f"{reference_path.stem}-validation-by-color.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"checkpoint": str(reference_path),
                                          "split": "val", "metrics": metrics}, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
        print(f"Result: {output_path}", flush=True)
        return
    model, training_summary = train(
        model,
        train_loaders,
        val_loaders,
        artifacts,
        device,
        lr=lr,
        projection_lr=args.projection_lr,
        weight_decay=weight_decay,
        updates=args.updates,
        eval_every=args.eval_every,
        min_updates_before_stop=args.min_updates_before_stop,
        patience=args.patience,
        min_delta=args.min_delta,
        near_miss_weight=args.near_miss_weight if args.variant == "gated-gine-projected-margin" else 0.0,
        near_miss_margin=args.near_miss_margin,
        diagnose_gradients=args.diagnose_gradients,
        aux_weight=args.aux_weight,
        aux_temperature=args.aux_temperature,
        encoder_anneal_steps=args.encoder_anneal_steps,
        aux_anneal_steps=args.aux_anneal_steps,
        cosine_lr=args.cosine_lr,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{timestamp}-{args.variant}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_metadata = {
        "version": 1,
        "options": {
            **options,
            "use_spelling": variant["use_spelling"],
        },
        "nodes": nodes.cache_identity,
        "edges": edges.configuration,
    }
    config = {
        "variant": args.variant,
        "description": variant["description"],
        "seed": args.seed,
        "device": str(device),
        "reference_checkpoint": str(reference_path),
        "dataset": str(dataset_path),
        "model_config": model_config,
        "lr": lr,
        "projection_lr": args.projection_lr,
        "diagnose_gradients": args.diagnose_gradients,
        "aux_weight": args.aux_weight,
        "aux_temperature": args.aux_temperature,
        "encoder_anneal_steps": args.encoder_anneal_steps,
        "aux_anneal_steps": args.aux_anneal_steps,
        "cosine_lr": args.cosine_lr,
        "weight_decay": weight_decay,
        "batch_size": args.batch_size,
        "updates": args.updates,
        "eval_every": args.eval_every,
        "min_updates_before_stop": args.min_updates_before_stop,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "near_miss_weight": args.near_miss_weight if args.variant == "gated-gine-projected-margin" else 0.0,
        "near_miss_margin": args.near_miss_margin,
        "edge_feature_names": list(edges.feature_names),
    }
    result = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "config": config,
        "training": training_summary,
    }
    result_path = output_dir / f"{run_name}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint_dir = REPO_ROOT / "weights" / "experiments"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_class": model_class.__name__,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "embedding_model_name": reference["embedding_model_name"],
            "feature_metadata": feature_metadata,
            "edge_feature_names": list(edges.feature_names),
            "split_dates": split_dates,
            "dataset_path": str(dataset_path),
            "training_config": config,
            "training_summary": training_summary,
        },
        checkpoint_path,
    )
    print(f"Result: {result_path}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(json.dumps(training_summary["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
