"""Answer-backed Connections environment for historical puzzle evaluation."""

from typing import Sequence

from .base import GameState, GuessFeedback, Outcome


class LocalConnectionsEnvironment:
    """NYT-style rules: four mistakes lose, and a 3/4 guess costs a mistake.

    Supply the displayed word order independently of the private answer groups
    so the policy cannot infer answers from their positions.
    """

    def __init__(self, words: Sequence[str], groups: Sequence[Sequence[str]], *, max_mistakes=4):
        words = tuple(words)
        answers = tuple(frozenset(group) for group in groups)
        flattened = [word for group in groups for word in group]
        if (len(words) != 16 or len(set(words)) != 16
                or any(not isinstance(word, str) for word in words)
                or len(answers) != 4 or any(len(group) != 4 for group in answers)
                or len(flattened) != 16 or set(flattened) != set(words)):
            raise ValueError("Expected 16 distinct words partitioned into four groups of four")
        if not isinstance(max_mistakes, int) or max_mistakes < 1:
            raise ValueError("max_mistakes must be a positive integer")
        self._answers = answers
        self._state = GameState(words, words, (), (), max_mistakes)

    def observe(self) -> GameState:
        return self._state

    def submit(self, guess: Sequence[str]) -> GuessFeedback:
        state = self._state
        if state.finished:
            raise ValueError("Game is finished")
        chosen = frozenset(guess)
        if len(guess) != 4 or len(chosen) != 4 or not chosen <= set(state.remaining):
            raise ValueError("Guess must contain four distinct remaining words")
        if any(chosen == frozenset(row.guess) for row in state.history):
            raise ValueError("This group has already been guessed")
        canonical = tuple(word for word in state.words if word in chosen)
        overlap = max(len(chosen & group) for group in self._answers)
        outcome = (Outcome.CORRECT if overlap == 4 else
                   Outcome.ONE_AWAY if overlap == 3 else Outcome.INCORRECT)
        feedback = GuessFeedback(canonical, outcome)
        correct = outcome == Outcome.CORRECT
        self._state = GameState(
            state.words,
            tuple(word for word in state.remaining if not correct or word not in chosen),
            state.solved_groups + ((canonical,) if correct else ()),
            state.history + (feedback,),
            state.mistakes_remaining - (not correct),
        )
        return feedback
