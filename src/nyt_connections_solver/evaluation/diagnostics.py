"""Replayable model decisions with gold labels confined to diagnostic overlays."""

from dataclasses import dataclass
import math
import random
from time import perf_counter

import numpy as np
import torch

from ..feedback_search import DecisionTrace, FeedbackConstraints, FeedbackSearchHarness
from ..game import GameState, GuessFeedback, LocalConnectionsEnvironment


def candidate_rankings(state, candidates, gold_groups, chosen):
    """Compare raw model ranking, exact feedback pruning, and ground truth.

    Gold labels are inspected only after public-feedback feasibility is computed.
    Softmax values describe the next-choice distribution, not calibrated truth.
    """
    constraints = FeedbackConstraints(state)
    conditioned = {frozenset(candidate.words): candidate.log_probability
                   for candidate in constraints.condition_candidates(candidates)}
    rows = []
    feasible_rank = 0
    for raw_rank, candidate in enumerate(sorted(candidates, key=lambda c: c.log_probability, reverse=True), 1):
        group = constraints.mask(candidate.words)
        flags = constraints.group_flags(group)
        feasible = frozenset(candidate.words) in conditioned
        if feasible:
            feasible_rank += 1
        selected = frozenset(candidate.words)
        overlap = max(len(selected & set(gold)) for gold in gold_groups)
        rows.append({
            "raw_rank": raw_rank,
            "feasible_rank": feasible_rank if feasible else None,
            "words": " · ".join(candidate.words),
            "group": candidate.words,
            "model_probability": math.exp(candidate.log_probability),
            "log_probability": candidate.log_probability,
            "log_probability_after_pruning": conditioned.get(selected),
            "probability_after_pruning": math.exp(conditioned[selected]) if feasible else 0.0,
            "feasible": feasible,
            "pruning": "allowed" if feasible else (
                "contradicts a miss" if flags is None else "no consistent partition"
            ),
            "true_group": overlap == 4,
            "max_gold_overlap": overlap,
            "chosen": selected == frozenset(chosen),
        })
    return rows


def project_words(values):
    """Deterministic centered PCA, fit once to the original sixteen words."""
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    coordinates = np.pad(coordinates, ((0, 0), (0, max(0, 2 - coordinates.shape[1]))))
    variance = singular[:2] ** 2 / max(float((singular ** 2).sum()), 1e-30)
    return coordinates, float(variance.sum())


@dataclass
class DiagnosticDecision:
    before: GameState
    chosen: tuple[str, ...]
    plan: tuple[tuple[str, ...], ...]
    trace: DecisionTrace
    rankings: list[dict]
    search_seconds: float
    diagnostic_seconds: float
    feedback: GuessFeedback | None = None


@dataclass
class DiagnosticRecord:
    date: str
    words: tuple[str, ...]
    gold_groups: tuple[tuple[str, ...], ...]
    decisions: list[DiagnosticDecision]
    final_state: GameState
    projections: dict
    edges: np.ndarray
    edge_names: tuple[str, ...]
    feature_seconds: float
    gold_group_labels: tuple[str, ...] = ()
    beam_width: int = 20
    max_mistakes: int = 4
    shuffle_seed: int = 0
    score_mode: str = "full_board"


class DiagnosticSession:
    """Interactive session or batch runner around the production play policy."""

    def __init__(self, harness, puzzle, *, beam_width=20, max_mistakes=4, seed=0,
                 score_mode="full_board"):
        words = list(puzzle.words)
        random.Random(f"{seed}:{puzzle.date}").shuffle(words)
        self.environment = LocalConnectionsEnvironment(words, tuple(puzzle.groups), max_mistakes=max_mistakes)
        self.policy = FeedbackSearchHarness.from_harness(
            harness, beam_width=beam_width, capture_trace=True, score_mode=score_mode)
        self.puzzle = puzzle
        self.max_mistakes = max_mistakes
        self.shuffle_seed = seed
        ordered_groups = getattr(puzzle, "ordered_groups", ())
        if not ordered_groups:
            # Legacy loaded puzzles retain CSV row order in their word list.
            ordered_groups = tuple(tuple(puzzle.words[i:i + 4]) for i in range(0, 16, 4))
        if frozenset(map(frozenset, ordered_groups)) == frozenset(map(frozenset, puzzle.groups)):
            self.gold_groups = tuple(tuple(group) for group in ordered_groups)
            self.gold_group_labels = ("Purple", "Blue", "Green", "Yellow")
        else:
            # Custom puzzles without ordered answers cannot establish difficulty.
            self.gold_groups = tuple(sorted(tuple(sorted(group)) for group in puzzle.groups))
            self.gold_group_labels = tuple(f"Group {i + 1}" for i in range(4))
        self.decisions = []
        self.pending = None
        start = perf_counter()
        self.policy.scorer.prepare(tuple(words))
        self.feature_seconds = perf_counter() - start
        nodes = self.policy.scorer.nodes
        edges = self.policy.scorer.edges
        node_builder = getattr(harness, "node_feature_builder", None)
        embeddings = getattr(node_builder, "embeddings", None)
        semantic_dim = getattr(embeddings, "dim", None)
        representations = ({"Text embeddings": nodes[:, :semantic_dim]} if semantic_dim
                           else {"Node features": nodes})
        representation_hook = getattr(harness.model, "diagnostic_representations", None)
        with torch.inference_mode():
            if representation_hook is not None:
                representations.update(representation_hook(nodes, edges))
        for name, tensor in representations.items():
            if tensor.ndim != 2 or tensor.shape[0] != len(words) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid per-word diagnostic representation: {name}")
        self.projections = {
            name: project_words(tensor.detach().cpu().numpy())
            for name, tensor in representations.items()
        }
        self.edges = (edges.detach().cpu().numpy() if edges is not None
                      else np.empty((len(words), len(words), 0)))
        names = harness.checkpoint.get("edge_feature_names", ()) if edges is not None else ()
        self.edge_names = tuple(names) or tuple(f"Edge {i + 1}" for i in range(self.edges.shape[-1]))
        if len(self.edge_names) != self.edges.shape[-1]:
            raise ValueError("Edge feature names do not match diagnostic channels")

    def analyze(self):
        if self.pending is not None:
            return self.pending
        state = self.environment.observe()
        if state.finished:
            return None
        start = perf_counter()
        chosen = self.policy.choose(state)
        search_seconds = perf_counter() - start
        trace = self.policy.last_trace
        start = perf_counter()
        rankings = candidate_rankings(state, trace.candidates, self.puzzle.groups, chosen)
        self.pending = DiagnosticDecision(
            state, chosen, self.policy.last_plan, trace, rankings,
            search_seconds, perf_counter() - start,
        )
        return self.pending

    def advance(self):
        decision = self.analyze()
        if decision is None:
            return None
        decision.feedback = self.environment.submit(decision.chosen)
        self.decisions.append(decision)
        self.pending = None
        return decision

    def finish(self):
        while not self.environment.observe().finished:
            self.advance()
        return self.record()

    def record(self):
        state = self.environment.observe()
        return DiagnosticRecord(
            self.puzzle.date, state.words,
            self.gold_groups,
            self.decisions + ([self.pending] if self.pending else []),
            state, self.projections, self.edges, self.edge_names, self.feature_seconds,
            self.gold_group_labels,
            self.policy.beam_width, self.max_mistakes, self.shuffle_seed, self.policy.score_mode,
        )


def summarize_record(record):
    submitted = [d for d in record.decisions if d.feedback is not None]
    return {
        "date": record.date,
        "beam_width": record.beam_width,
        "score_mode": record.score_mode,
        "allowed_incorrect_guesses": record.max_mistakes,
        "shuffle_seed": record.shuffle_seed,
        "won": record.final_state.won,
        "correct_groups": len(record.final_state.solved_groups),
        "guesses": len(submitted),
        "mistakes": sum(d.feedback.outcome.value != "correct" for d in submitted),
        "feature_seconds": record.feature_seconds,
        "search_seconds": sum(d.search_seconds for d in submitted),
        "diagnostic_seconds": sum(d.diagnostic_seconds for d in submitted),
    }
