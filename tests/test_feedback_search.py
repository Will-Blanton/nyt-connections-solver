"""Feedback correctness, exact pruning, and backend/model interchangeability."""

from itertools import combinations
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyt_connections_solver.game import (
    GameState, GuessFeedback, LocalConnectionsEnvironment, Outcome, play_game,
)
from nyt_connections_solver.feedback_search import (
    FeedbackConstraints, FeedbackSearchHarness, ModelGroupScorer, ScoredGroup,
)
from nyt_connections_solver.evaluation.feedback import evaluate_feedback_games


WORDS = tuple("ABCDEFGHIJKLMNOP")
GROUPS = tuple(tuple(WORDS[i:i + 4]) for i in range(0, 16, 4))


class TableScorer:
    """Rank provided groups first, while still enumerating every candidate."""

    def __init__(self, ranking):
        self.ranking = list(map(frozenset, ranking))
        self.prepared = []

    def prepare(self, words):
        self.prepared.append(words)

    def score_states(self, states):
        def score(group):
            key = frozenset(group)
            return -float(self.ranking.index(key)) if key in self.ranking else -100.0
        return [tuple(ScoredGroup(group, score(group)) for group in combinations(words, 4))
                for words in states]


class FeedbackGameTests(unittest.TestCase):
    def test_feedback_and_mistake_budget(self):
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        self.assertEqual(env.submit(tuple("ABCE")).outcome, Outcome.ONE_AWAY)
        self.assertEqual(env.observe().mistakes_remaining, 3)
        self.assertEqual(env.submit(tuple("AEIM")).outcome, Outcome.INCORRECT)
        self.assertEqual(env.submit(GROUPS[0]).outcome, Outcome.CORRECT)
        self.assertEqual(env.observe().mistakes_remaining, 2)
        self.assertEqual(len(env.observe().remaining), 12)
        self.assertFalse(hasattr(env.observe(), "answers"))
        self.assertFalse(hasattr(env.observe().history[0], "correct_triple"))

    def test_invalid_or_repeated_guesses_do_not_cost_mistakes(self):
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        env.submit(tuple("ABCE"))
        for guess in (tuple("ECBA"), tuple("AABC"), tuple("ABCZ"), tuple("ABC")):
            before = env.observe()
            with self.assertRaises(ValueError):
                env.submit(guess)
            self.assertEqual(env.observe(), before)
        env.submit(GROUPS[0])
        with self.assertRaises(ValueError):
            env.submit(tuple("ABCF"))

    def test_four_mistakes_lose(self):
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        for guess in ("AEIM", "BFJN", "CGKO", "DHLP"):
            env.submit(tuple(guess))
        self.assertTrue(env.observe().finished)
        self.assertFalse(env.observe().won)
        with self.assertRaises(ValueError):
            env.submit(GROUPS[0])

    def test_win_requires_submitting_last_group(self):
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        for group in GROUPS[:3]:
            env.submit(group)
        self.assertFalse(env.observe().finished)
        env.submit(GROUPS[3])
        self.assertTrue(env.observe().won)

    def test_runner_accepts_an_alternative_backend(self):
        local = LocalConnectionsEnvironment(WORDS, GROUPS)
        class Adapter:
            def observe(self): return local.observe()
            def submit(self, guess): return local.submit(guess)
        policy = FeedbackSearchHarness(TableScorer(GROUPS), beam_width=2)
        result = play_game(Adapter(), policy)
        self.assertTrue(result.won)
        self.assertEqual(len(result.final_state.history), 4)


class ConstraintTests(unittest.TestCase):
    def state(self, feedback):
        return GameState(WORDS[:8], WORDS[:8], (), tuple(feedback), 4)

    def test_pruning_matches_exhaustive_partitions(self):
        # Every partition of eight words: a cheap independent exhaustive oracle.
        histories = [(),
                     (GuessFeedback(tuple("ABCE"), Outcome.ONE_AWAY),),
                     (GuessFeedback(tuple("ABCE"), Outcome.INCORRECT),),
                     (GuessFeedback(tuple("ABCE"), Outcome.ONE_AWAY),
                      GuessFeedback(tuple("ABCF"), Outcome.INCORRECT))]
        for history in histories:
            constraints = FeedbackConstraints(self.state(history))
            any_valid = False
            for others in combinations(WORDS[1:8], 3):
                first = frozenset((WORDS[0],) + others)
                second = frozenset(WORDS[:8]) - first
                expected = all(
                    (max(len(first & set(row.guess)), len(second & set(row.guess))) == 3
                     if row.outcome == Outcome.ONE_AWAY else
                     max(len(first & set(row.guess)), len(second & set(row.guess))) < 3)
                    for row in history
                )
                masks = [constraints.mask(group) for group in (first, second)]
                flags = [constraints.group_flags(mask) for mask in masks]
                actual = (all(flag is not None for flag in flags)
                          and (flags[0] | flags[1]) == constraints.required)
                self.assertEqual(actual, expected)
                any_valid |= expected
                first_flags = flags[0]
                feasible_tail = (first_flags is not None and
                                 constraints.completion(masks[1], first_flags) is not None)
                self.assertEqual(feasible_tail, expected)
            self.assertEqual(constraints.completion(constraints.remaining, 0) is not None, any_valid)

    def test_one_away_keeps_every_possible_triple(self):
        constraints = FeedbackConstraints(self.state((GuessFeedback(tuple("ABCE"), Outcome.ONE_AWAY),)))
        for group in ("ABCD", "ABEF", "ACEF", "BCEF"):
            mask = constraints.mask(tuple(group))
            flags = constraints.group_flags(mask)
            self.assertIsNotNone(flags)
            self.assertIsNotNone(constraints.completion(constraints.remaining ^ mask, flags))
        self.assertIsNone(constraints.group_flags(constraints.mask(tuple("ABCE"))))

    def test_solved_group_can_satisfy_earlier_one_away(self):
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        env.submit(tuple("ABCE"))
        env.submit(GROUPS[0])
        constraints = FeedbackConstraints(env.observe())
        self.assertEqual(constraints.satisfied, constraints.required)
        self.assertIsNotNone(constraints.completion(constraints.remaining, constraints.satisfied))

    def test_unsatisfied_triple_cannot_be_split_across_solved_words(self):
        state = GameState(WORDS[:8], tuple("EFGH"), (tuple("ABCD"),),
                          (GuessFeedback(tuple("ABEF"), Outcome.ONE_AWAY),), 3)
        constraints = FeedbackConstraints(state)
        self.assertIsNone(constraints.completion(constraints.remaining, constraints.satisfied))

    def test_retry_prunes_rejected_guess_and_inconsistent_partition(self):
        # First a high-ranked wrong partition, then a true partition. Feedback
        # must forbid the rejected group even though it still has the top score.
        ranking = [tuple("ABCE"), tuple("DFGH"), GROUPS[2], GROUPS[3], *GROUPS]
        policy = FeedbackSearchHarness(TableScorer(ranking), beam_width=2)
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        first = policy.choose(env.observe())
        self.assertEqual(set(first), set("ABCE"))
        env.submit(first)
        second = policy.choose(env.observe())
        self.assertNotEqual(set(first), set(second))
        self.assertTrue(any(len(set(group) & set(first)) == 3 for group in policy.last_plan))
        result = play_game(env, policy)
        self.assertTrue(result.won)
        guesses = [frozenset(row.guess) for row in result.final_state.history]
        self.assertEqual(len(guesses), len(set(guesses)))

    def test_ordinary_miss_rejects_paths_with_a_three_word_overlap(self):
        constraints = FeedbackConstraints(self.state((GuessFeedback(tuple("ABCE"), Outcome.INCORRECT),)))
        self.assertIsNone(constraints.group_flags(constraints.mask(tuple("ABCD"))))
        self.assertIsNotNone(constraints.group_flags(constraints.mask(tuple("ABFG"))))

    def test_evaluation_records_retries_and_reproducible_shuffled_words(self):
        puzzle = SimpleNamespace(date="example", words=WORDS, groups=GROUPS)
        scorers = [TableScorer(GROUPS), TableScorer(GROUPS)]
        for scorer in scorers:
            metrics = evaluate_feedback_games(FeedbackSearchHarness(scorer, beam_width=1), [puzzle], seed=7)
            self.assertEqual(metrics["solve_rate"], 1.0)
            self.assertEqual(metrics["mean_guesses"], 4.0)
        self.assertEqual(scorers[0].prepared[0], scorers[1].prepared[0])
        self.assertNotEqual(scorers[0].prepared[0], WORDS)


class ConditioningTests(unittest.TestCase):
    def test_prune_then_renormalize_is_stable_and_keeps_raw_scores(self):
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        env.submit(tuple("ABCE"))
        constraints = FeedbackConstraints(env.observe())
        raw = (ScoredGroup(tuple("ABCE"), -.1), ScoredGroup(GROUPS[0], -1000.),
               ScoredGroup(GROUPS[1], -1001.))
        conditioned = constraints.condition_candidates(raw)
        self.assertEqual(len(conditioned), 2)
        self.assertAlmostEqual(sum(math.exp(c.log_probability) for c in conditioned), 1.)
        self.assertAlmostEqual(math.exp(conditioned[0].log_probability), 1 / (1 + math.exp(-1)))
        self.assertEqual(raw[1].log_probability, -1000.)

    def test_residual_state_renormalization_changes_the_winning_path(self):
        class BranchScorer:
            def prepare(self, words): pass
            def score_states(self, states):
                rows = []
                for state in states:
                    groups = list(combinations(state, 4))
                    special = {}
                    if len(state) == 16:
                        special = {frozenset("ABCD"): .45, frozenset("EFGH"): .4}
                    elif len(state) == 12 and 'A' not in state:
                        special = {frozenset("EFGH"): .1}
                    elif len(state) == 12 and 'E' not in state:
                        # AIMN is ruled out by the earlier ordinary miss AEIM.
                        special = {frozenset("AIMN"): .98, frozenset("ABCD"): .01}
                    elif len(state) == 8:
                        special = {frozenset(g): .49 for g in ("IJKL", "MNOP") if set(g) <= set(state)}
                    fallback = (1 - sum(special.values())) / (len(groups) - len(special))
                    weights = [special.get(frozenset(g), fallback) for g in groups]
                    total = sum(weights)
                    rows.append(tuple(ScoredGroup(g, math.log(w / total)) for g, w in zip(groups, weights)))
                return rows
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        env.submit(tuple("AEIM"))
        policy = FeedbackSearchHarness(BranchScorer(), beam_width=2, capture_trace=True)
        chosen = policy.choose(env.observe())
        self.assertEqual(set(chosen), set("EFGH"))
        # Root normalization is over every feasible choice, not the top-two beam.
        root = FeedbackConstraints(env.observe()).condition_candidates(policy.last_trace.candidates)
        expected = next(c.log_probability for c in root if set(c.words) == set(chosen))
        self.assertAlmostEqual(policy.last_trace.beams_by_depth[0][1].log_probability, expected)
        self.assertLess(sum(math.exp(path.log_probability) for path in policy.last_trace.beams_by_depth[0]), 1.)


class ModelAdapterTests(unittest.TestCase):
    def test_from_model_with_feature_provider_and_no_separate_embedder(self):
        import torch
        class Model(torch.nn.Module):
            def forward(self, x, group_idx):
                return x[:, group_idx, :].sum(dim=(2, 3)).unsqueeze(-1)
        builder = SimpleNamespace(encode=lambda words, **kwargs: torch.ones(16, 2))
        policy = FeedbackSearchHarness.from_model(
            Model(), input_dim=2, node_feature_builder=builder, device="cpu", beam_width=1,
        )
        guess = policy.choose(LocalConnectionsEnvironment(WORDS, GROUPS).observe())
        self.assertEqual(len(set(guess)), 4)
        # The original API continues to produce complete partitions.
        self.assertEqual(len(policy.scorer.harness.solve(WORDS).best_groups), 4)

    def test_graph_features_are_sliced_and_scores_cached_across_retries(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, x, group_idx, edge_features):
                self.calls.append((x.clone(), edge_features.clone()))
                return x[:, group_idx, :].sum(dim=(2, 3)).unsqueeze(-1)
        model = Model()
        nodes = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        edges = torch.arange(256, dtype=torch.float32).reshape(16, 16, 1)
        encoded = []
        def encode(words):
            encoded.append(words)
            return nodes, edges
        harness = SimpleNamespace(
            model=model, device="cpu", state_batch_size=2, encode_puzzle=encode,
            group_indices={n: torch.tensor(list(combinations(range(n), 4))) for n in (8, 12, 16)},
        )
        scorer = ModelGroupScorer(harness, max_cached_states=2)
        scorer.prepare(WORDS)
        remaining = WORDS[::2]
        result = scorer.score_states([remaining])
        scorer.prepare(WORDS)
        self.assertEqual(scorer.score_states([remaining]), result)
        self.assertEqual(len(encoded), 1)
        self.assertEqual(len(model.calls), 1)
        positions = torch.arange(0, 16, 2)
        torch.testing.assert_close(model.calls[0][0][0], nodes[positions])
        torch.testing.assert_close(model.calls[0][1][0], edges[positions[:, None], positions[None, :]])
        self.assertEqual(len(result[0]), 70)
        scorer.score_states([WORDS[:8], WORDS[8:], WORDS[:12]])
        self.assertEqual(len(scorer._scores), 2)


if __name__ == "__main__":
    unittest.main()
