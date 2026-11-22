"""Shared game protocol, public feedback types, and backend-independent play loop.

An eventual browser adapter only needs ``observe`` and ``submit``. Answers and
overlap counts below three never cross this interface.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence


Group = tuple[str, ...]


class Outcome(str, Enum):
    CORRECT = "correct"
    ONE_AWAY = "one_away"
    INCORRECT = "incorrect"


@dataclass(frozen=True)
class GuessFeedback:
    guess: Group
    outcome: Outcome


@dataclass(frozen=True)
class GameState:
    """Everything a player may know, including the full public guess history."""

    words: Group
    remaining: Group
    solved_groups: tuple[Group, ...]
    history: tuple[GuessFeedback, ...]
    mistakes_remaining: int

    @property
    def won(self) -> bool:
        return not self.remaining

    @property
    def finished(self) -> bool:
        return self.won or self.mistakes_remaining == 0


class ConnectionsEnvironment(Protocol):
    def observe(self) -> GameState: ...

    def submit(self, guess: Sequence[str]) -> GuessFeedback: ...


class GuessPolicy(Protocol):
    def choose(self, state: GameState) -> Group: ...


@dataclass(frozen=True)
class PlayResult:
    final_state: GameState

    @property
    def won(self) -> bool:
        return self.final_state.won


def play_game(environment: ConnectionsEnvironment, policy: GuessPolicy) -> PlayResult:
    """Run any policy against a local or browser-backed environment."""
    state = environment.observe()
    while not state.finished:
        guess = tuple(policy.choose(state))
        if (len(guess) != 4 or len(set(guess)) != 4 or not set(guess) <= set(state.remaining)
                or any(set(guess) == set(row.guess) for row in state.history)):
            raise ValueError("Policy proposed an invalid or repeated guess")
        environment.submit(guess)
        updated = environment.observe()
        if updated == state:
            raise RuntimeError("Environment did not update after a submitted guess")
        state = updated
    return PlayResult(state)
