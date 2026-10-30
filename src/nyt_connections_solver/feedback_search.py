"""Feedback-constrained partition search, independent of the game backend."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from itertools import combinations
import math
from typing import Protocol, Sequence

from .game import GameState, Group, Outcome


@dataclass(frozen=True)
class ScoredGroup:
    words: Group
    log_probability: float


@dataclass(frozen=True)
class BeamPath:
    groups: tuple[Group, ...]
    log_probability: float


@dataclass(frozen=True)
class DecisionTrace:
    """Public model/search output; never includes gold labels."""

    candidates: tuple[ScoredGroup, ...]
    beams_by_depth: tuple[tuple[BeamPath, ...], ...]


class GroupScorer(Protocol):
    """Scores use the same original-board word order at every residual state."""

    def prepare(self, words: Group) -> None: ...

    def score_states(self, states: Sequence[Group]) -> list[tuple[ScoredGroup, ...]]: ...


class ModelGroupScorer:
    """Adapt a SearchHarness, including its graph features, to the play policy.

    Encode once per board and cache residual-state scores across retries. The
    cache is bounded and reset for a different board. No answers are accessed.
    """

    def __init__(self, harness, *, max_cached_states=1024):
        if max_cached_states < 1:
            raise ValueError("max_cached_states must be positive")
        self.harness = harness
        self.max_cached_states = max_cached_states
        self.words = None
        self._scores = OrderedDict()

    @classmethod
    def from_model(cls, model, word_embedder=None, **harness_options):
        """Accept the same feature providers/options as SearchHarness.from_model."""
        from .harness import SearchHarness

        return cls(SearchHarness.from_model(model, word_embedder, **harness_options))

    def prepare(self, words: Group) -> None:
        if tuple(words) == self.words:
            return
        self.words = tuple(words)
        self._positions = {word: i for i, word in enumerate(words)}
        self._scores.clear()
        self.nodes, self.edges = self.harness.encode_puzzle(words)

    def score_states(self, states: Sequence[Group]) -> list[tuple[ScoredGroup, ...]]:
        import torch

        if self.words is None:
            raise RuntimeError("Call prepare before scoring")
        states = [tuple(state) for state in states]
        missing = list(dict.fromkeys(state for state in states if state not in self._scores))
        # Keep requested results locally even when a batch exceeds the cache bound.
        results = {state: self._scores[state] for state in states if state in self._scores}
        harness = self.harness
        harness.model.eval()
        with torch.inference_mode():
            for count in sorted({len(state) for state in missing}):
                if count not in (8, 12, 16):
                    raise ValueError("Model scoring expects 8, 12, or 16 remaining words")
                rows = [state for state in missing if len(state) == count]
                candidates = harness.group_indices[count]
                candidate_ids = candidates.cpu().tolist()
                for start in range(0, len(rows), harness.state_batch_size):
                    batch = rows[start:start + harness.state_batch_size]
                    positions = torch.tensor(
                        [[self._positions[word] for word in state] for state in batch],
                        device=harness.device, dtype=torch.long,
                    )
                    kwargs = {}
                    if self.edges is not None:
                        kwargs["edge_features"] = self.edges[
                            positions[:, :, None], positions[:, None, :]
                        ]
                    logits = harness.model(
                        self.nodes[positions], group_idx=candidates, **kwargs
                    ).squeeze(-1).float()
                    if logits.shape != (len(batch), len(candidates)) or not torch.isfinite(logits).all():
                        raise ValueError("Model must return finite scores for every candidate")
                    probabilities = logits.log_softmax(dim=1).cpu().tolist()
                    for state, values in zip(batch, probabilities):
                        results[state] = tuple(
                            ScoredGroup(tuple(state[i] for i in indices), score)
                            for indices, score in zip(candidate_ids, values)
                        )
        for state in states:
            self._scores[state] = results[state]
            self._scores.move_to_end(state)
        while len(self._scores) > self.max_cached_states:
            self._scores.popitem(last=False)
        return [results[state] for state in states]


class FeedbackConstraints:
    """Exact feasibility of a partition given only public feedback.

    A one-away guess requires *some* group with intersection exactly three;
    it never identifies the triple. An ordinary miss implies all intersections
    are at most two. Accepted groups are fixed in every surviving partition.
    """

    def __init__(self, state: GameState):
        self.words = state.words
        self.positions = {word: i for i, word in enumerate(state.words)}
        self.one_away = tuple(self.mask(row.guess) for row in state.history
                              if row.outcome == Outcome.ONE_AWAY)
        self.misses = tuple(self.mask(row.guess) for row in state.history
                            if row.outcome == Outcome.INCORRECT)
        self.required = (1 << len(self.one_away)) - 1
        self.remaining = self.mask(state.remaining)
        self.group_flags = cache(self._group_flags)
        self.satisfied = 0
        for group in state.solved_groups:
            flags = self.group_flags(self.mask(group))
            if flags is None:
                raise ValueError("Accepted group contradicts previous feedback")
            self.satisfied |= flags

        # Per-decision caches are discarded after feedback changes.
        self.completion = cache(self._completion)

    def mask(self, words: Sequence[str]) -> int:
        return sum(1 << self.positions[word] for word in words)

    def decode(self, mask: int) -> Group:
        return tuple(word for i, word in enumerate(self.words) if mask & (1 << i))

    def _group_flags(self, group: int) -> int | None:
        if any((group & guess).bit_count() >= 3 for guess in self.misses):
            return None
        flags = 0
        for i, guess in enumerate(self.one_away):
            overlap = (group & guess).bit_count()
            if overlap == 4:
                return None
            if overlap == 3:
                flags |= 1 << i
        return flags

    def _completion(self, remaining: int, satisfied: int) -> tuple[int, ...] | None:
        if not remaining:
            return () if satisfied == self.required else None
        for i, guess in enumerate(self.one_away):
            if not satisfied & (1 << i) and (remaining & guess).bit_count() < 3:
                return None
        # Fix one word in the next group to enumerate unordered partitions once.
        pivot = remaining & -remaining
        rest = [1 << i for i in range(len(self.words)) if remaining & (1 << i) and (1 << i) != pivot]
        for others in combinations(rest, 3):
            group = pivot | sum(others)
            flags = self.group_flags(group)
            if flags is None:
                continue
            tail = self.completion(remaining ^ group, satisfied | flags)
            if tail is not None:
                return (group,) + tail
        return None

    def condition_candidates(self, candidates, *, remaining=None, satisfied=None):
        """Prune impossible choices, then stably renormalize their log scores.

        This is done before beam truncation and at every hypothetical residual
        state. Raw model scores stay cached independently of public feedback.
        """
        remaining = self.remaining if remaining is None else remaining
        satisfied = self.satisfied if satisfied is None else satisfied
        remaining_words = set(self.decode(remaining))
        feasible = []
        for candidate in candidates:
            if not math.isfinite(candidate.log_probability):
                raise ValueError("Candidate scores must be finite")
            if (len(candidate.words) != 4 or len(set(candidate.words)) != 4
                    or not set(candidate.words) <= remaining_words):
                raise ValueError("Scorer returned an invalid candidate group")
            group = self.mask(candidate.words)
            flags = self.group_flags(group)
            if flags is not None and self.completion(remaining ^ group, satisfied | flags) is not None:
                feasible.append(candidate)
        if not feasible:
            return ()
        maximum = max(candidate.log_probability for candidate in feasible)
        log_mass = maximum + math.log(math.fsum(
            math.exp(candidate.log_probability - maximum) for candidate in feasible
        ))
        return tuple(ScoredGroup(candidate.words, candidate.log_probability - log_mass)
                     for candidate in feasible)


class FeedbackSearchHarness:
    """Beam-search complete feasible paths, submit their first group, then replan.

    Every retained partial path has an exact completion consistent with *all*
    feedback. Beam width approximates score optimization, never feasibility.
    Scores sum log probabilities renormalized over feedback-consistent choices
    at every state, before beam truncation.
    """

    def __init__(self, scorer: GroupScorer, *, beam_width=20, capture_trace=False,
                 full_board_weight=0.0, score_mode="path", weakest_group_weight=0.0):
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self.scorer = scorer
        self.beam_width = beam_width
        self.last_plan: tuple[Group, ...] = ()
        self.capture_trace = capture_trace
        if full_board_weight < 0:
            raise ValueError("full_board_weight must be nonnegative")
        self.full_board_weight = float(full_board_weight)
        if score_mode not in ("path", "full_board"):
            raise ValueError("score_mode must be 'path' or 'full_board'")
        self.score_mode = score_mode
        if weakest_group_weight < 0 or (weakest_group_weight and score_mode != "full_board"):
            raise ValueError("weakest_group_weight requires full_board mode and must be nonnegative")
        self.weakest_group_weight = float(weakest_group_weight)
        self.last_trace: DecisionTrace | None = None

    @classmethod
    def from_harness(cls, harness, *, beam_width=None, max_cached_states=1024,
                     capture_trace=False, full_board_weight=0.0, score_mode="path",
                     weakest_group_weight=0.0):
        return cls(ModelGroupScorer(harness, max_cached_states=max_cached_states),
                   beam_width=harness.beam_width if beam_width is None else beam_width,
                   capture_trace=capture_trace, full_board_weight=full_board_weight,
                   score_mode=score_mode, weakest_group_weight=weakest_group_weight)

    @classmethod
    def from_model(cls, model, word_embedder=None, *, capture_trace=False, **harness_options):
        scorer = ModelGroupScorer.from_model(model, word_embedder, **harness_options)
        return cls(scorer, beam_width=scorer.harness.beam_width, capture_trace=capture_trace)

    def choose(self, state: GameState) -> Group:
        self.last_trace = None
        if state.finished:
            raise ValueError("Cannot choose for a finished game")
        constraints = FeedbackConstraints(state)
        if constraints.completion(constraints.remaining, constraints.satisfied) is None:
            raise ValueError("Public feedback admits no valid partition")
        self.scorer.prepare(state.words)
        full_board_scores = (
            {frozenset(row.words): row.log_probability
             for row in self.scorer.score_states([state.words])[0]}
            if self.full_board_weight or self.score_mode == "full_board" else None
        )
        # remaining mask, chosen masks, summed score, satisfied one-away flags
        beam = [(constraints.remaining, (), 0.0, constraints.satisfied)]
        root_candidates = ()
        depth_traces = []
        for _ in range(len(state.remaining) // 4):
            states = [constraints.decode(remaining) for remaining, _, _, _ in beam]
            if len(states[0]) == 4:
                rows = [(ScoredGroup(words, 0.0),) for words in states]
            else:
                rows = self.scorer.score_states(states)
            if len(rows) != len(beam):
                raise ValueError("Scorer returned the wrong number of states")
            if self.capture_trace and not root_candidates:
                root_candidates = tuple(rows[0])
            children = []
            for (remaining, groups, score, satisfied), candidates in zip(beam, rows):
                conditioned = constraints.condition_candidates(
                    candidates, remaining=remaining, satisfied=satisfied,
                )
                for candidate in conditioned:
                    group = constraints.mask(candidate.words)
                    flags = constraints.group_flags(group)
                    next_remaining = remaining ^ group
                    extra = 0.0
                    if full_board_scores is not None and remaining.bit_count() > 4:
                        extra = full_board_scores[frozenset(candidate.words)]
                        if next_remaining.bit_count() == 4:
                            extra += full_board_scores[frozenset(constraints.decode(next_remaining))]
                    if self.score_mode == "full_board":
                        scored_groups = groups + (group,)
                        if next_remaining.bit_count() == 4:
                            scored_groups += (next_remaining,)
                        values = [full_board_scores[frozenset(constraints.decode(mask))]
                                  for mask in scored_groups]
                        next_score = sum(values) + self.weakest_group_weight * min(values)
                    else:
                        next_score = score + candidate.log_probability + self.full_board_weight * extra
                    children.append((remaining ^ group, groups + (group,),
                                     next_score, satisfied | flags))
            children.sort(key=lambda path: path[2], reverse=True)
            beam = []
            seen = set()
            for child in children:
                remaining, groups, score, satisfied = child
                key = (remaining, satisfied)
                if key not in seen:
                    beam.append(child)
                    seen.add(key)
                    if len(beam) == self.beam_width:
                        break
            if not beam:
                raise ValueError("Scorer supplied no feedback-consistent path; supply all candidates")
            if self.capture_trace:
                depth_traces.append(tuple(
                    BeamPath(tuple(constraints.decode(group) for group in groups), score)
                    for _, groups, score, _ in beam
                ))
        self.last_plan = tuple(constraints.decode(group) for group in beam[0][1])
        if self.capture_trace:
            self.last_trace = DecisionTrace(root_candidates, tuple(depth_traces))
        return self.last_plan[0]
