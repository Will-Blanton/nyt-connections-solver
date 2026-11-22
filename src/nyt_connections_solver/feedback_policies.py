"""Model-driven alternatives for choosing a guess from feedback search paths."""

from math import exp, log

from .feedback_search import FeedbackConstraints, FeedbackSearchHarness


class PartitionBeliefPolicy:
    """Vote across distinct feasible partitions, optionally valuing feedback."""

    def __init__(self, harness, *, beam_width=100, information_weight=0.0,
                 temperature=2.0):
        self.search = FeedbackSearchHarness.from_harness(
            harness, beam_width=beam_width, capture_trace=True)
        self.information_weight = information_weight
        self.temperature = temperature
        self.last_trace = None

    def choose(self, state):
        baseline = self.search.choose(state)
        self.last_trace = self.search.last_trace
        constraints = FeedbackConstraints(state)
        # The final depth contains one deduplicated completed state. Use the
        # penultimate depth and its forced remaining group instead.
        paths = self.last_trace.beams_by_depth[-2] if len(self.last_trace.beams_by_depth) > 1 else ()
        partitions = {}
        for path in paths:
            used = set().union(*(set(group) for group in path.groups))
            last = tuple(word for word in state.remaining if word not in used)
            groups = tuple(map(frozenset, (*path.groups, last)))
            key = frozenset(groups)
            if key not in partitions or path.log_probability > partitions[key][0]:
                partitions[key] = (path.log_probability, groups)
        if not partitions:
            return baseline
        scored = list(partitions.values())
        peak = max(score for score, _ in scored)
        weights = [exp((score - peak) / self.temperature) for score, _ in scored]
        total = sum(weights)
        weights = [weight / total for weight in weights]
        feasible = constraints.condition_candidates(self.last_trace.candidates)
        best = None
        for candidate in feasible:
            group = frozenset(candidate.words)
            outcomes = [0.0, 0.0, 0.0]
            for weight, (_, partition) in zip(weights, scored):
                overlap = max(len(group & answer) for answer in partition)
                outcomes[0 if overlap == 4 else 1 if overlap == 3 else 2] += weight
            entropy = -sum(value * log(value) for value in outcomes if value)
            # Information has no value when a wrong answer would end the game.
            info = self.information_weight if state.mistakes_remaining > 1 else 0.0
            utility = outcomes[0] + info * entropy / log(3)
            key = (utility, candidate.log_probability)
            if best is None or key > best[0]:
                best = key, candidate.words
        return best[1] if best else baseline
