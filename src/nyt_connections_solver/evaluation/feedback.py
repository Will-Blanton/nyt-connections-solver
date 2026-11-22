"""End-to-end feedback evaluation, kept outside training/Optuna loops."""

import random
from time import perf_counter
from typing import Sequence

from ..game import GuessPolicy, LocalConnectionsEnvironment, Outcome, play_game


def evaluate_feedback_games(policy: GuessPolicy, puzzles: Sequence, *, max_mistakes=4, seed=0) -> dict:
    """Play held-out puzzles with retries; retain public transcripts and timings.

    Shuffle each board reproducibly: historical CSV word order is grouped by
    answer and must not become a shortcut. The policy receives no gold labels.
    """
    if not puzzles:
        raise ValueError("No puzzles selected for evaluation")
    rng = random.Random(seed)
    rows = []
    for puzzle in puzzles:
        words = list(puzzle.words)
        rng.shuffle(words)
        environment = LocalConnectionsEnvironment(words, tuple(puzzle.groups), max_mistakes=max_mistakes)
        start = perf_counter()
        result = play_game(environment, policy)
        seconds = perf_counter() - start
        state = result.final_state
        rows.append({
            "date": puzzle.date, "won": result.won,
            "correct_groups": len(state.solved_groups), "guesses": len(state.history),
            "mistakes": sum(row.outcome != Outcome.CORRECT for row in state.history),
            "one_away": sum(row.outcome == Outcome.ONE_AWAY for row in state.history),
            "first_guess_correct": state.history[0].outcome == Outcome.CORRECT,
            "seconds": seconds, "history": state.history,
        })
    count = len(rows)
    solved = sum(row["won"] for row in rows)
    return {
        "puzzles": count, "solved": solved, "solve_rate": solved / count,
        "mean_correct_groups": sum(row["correct_groups"] for row in rows) / count,
        "first_guess_accuracy": sum(row["first_guess_correct"] for row in rows) / count,
        "mean_guesses": sum(row["guesses"] for row in rows) / count,
        "mean_mistakes": sum(row["mistakes"] for row in rows) / count,
        "one_away_guesses": sum(row["one_away"] for row in rows),
        "correct_groups_histogram": {k: sum(row["correct_groups"] == k for row in rows) for k in range(5)},
        "total_seconds": sum(row["seconds"] for row in rows),
        "mean_seconds": sum(row["seconds"] for row in rows) / count,
        "games": rows,
    }
