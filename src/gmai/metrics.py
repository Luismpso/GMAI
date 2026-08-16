"""Episode instrumentation.

The metric that hid four bugs for 2000 episodes was ``score = (W + 0.5D)/n``.
Because ~87% of random full-chess games end in a draw, that score starts at
0.499 and saturates there — a learning agent and a coin flip look identical.

Everything here reports **wins, draws and losses separately**, plus episode
length and *why* the game ended. ``win_rate`` is the pure fraction of wins,
whose random baseline is ~0.065 rather than ~0.5, so there is real dynamic
range to measure progress in.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import chess

# Draws that mean "the agent failed to convert", i.e. real errors in a
# forced-mate endgame, as opposed to simply running out of moves.
CONVERSION_FAILURES = frozenset(
    {"STALEMATE", "INSUFFICIENT_MATERIAL", "FIFTY_MOVES", "THREEFOLD_REPETITION"}
)


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single episode, from the agent's point of view."""

    win: bool
    draw: bool
    loss: bool
    plies: int
    termination: str  # e.g. CHECKMATE, STALEMATE, TRUNCATED

    @property
    def score(self) -> float:
        """Chess score (1 / 0.5 / 0) — kept for reference, never as the sole KPI."""
        return 1.0 if self.win else (0.5 if self.draw else 0.0)


def classify_episode(
    board: chess.Board, agent_color: chess.Color, truncated: bool
) -> EpisodeResult:
    """Turn a finished board into an :class:`EpisodeResult`."""
    outcome = board.outcome(claim_draw=True)
    plies = len(board.move_stack)

    if outcome is None:
        # Ran out of moves without a game-over condition.
        return EpisodeResult(False, True, False, plies, "TRUNCATED")

    termination = "TRUNCATED" if truncated else outcome.termination.name
    if outcome.winner is None:
        return EpisodeResult(False, True, False, plies, termination)
    if outcome.winner == agent_color:
        return EpisodeResult(True, False, False, plies, termination)
    return EpisodeResult(False, False, True, plies, termination)


class RollingStats:
    """Rolling window of episode results with separated W/D/L."""

    def __init__(self, window: int = 200):
        self.window = window
        self._results: deque[EpisodeResult] = deque(maxlen=window)
        self.total_episodes = 0

    def __len__(self) -> int:
        return len(self._results)

    @property
    def is_full(self) -> bool:
        return len(self._results) == self.window

    def add(self, result: EpisodeResult) -> None:
        self._results.append(result)
        self.total_episodes += 1

    def clear(self) -> None:
        self._results.clear()

    # ------------------------------------------------------------- metrics
    @property
    def wins(self) -> int:
        return sum(r.win for r in self._results)

    @property
    def draws(self) -> int:
        return sum(r.draw for r in self._results)

    @property
    def losses(self) -> int:
        return sum(r.loss for r in self._results)

    @property
    def win_rate(self) -> float:
        """Pure win fraction. THIS is the promotion criterion, not `score`."""
        return self.wins / len(self._results) if self._results else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / len(self._results) if self._results else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / len(self._results) if self._results else 0.0

    @property
    def score(self) -> float:
        return (
            sum(r.score for r in self._results) / len(self._results)
            if self._results
            else 0.0
        )

    @property
    def mean_plies(self) -> float:
        return (
            sum(r.plies for r in self._results) / len(self._results)
            if self._results
            else 0.0
        )

    @property
    def terminations(self) -> Counter:
        return Counter(r.termination for r in self._results)

    @property
    def conversion_failure_rate(self) -> float:
        """Fraction of episodes lost to stalemate / repetition / 50-move / K-vs-K.

        In a forced-mate endgame every one of these is the agent throwing away
        a won position, which is exactly the failure mode worth watching.
        """
        if not self._results:
            return 0.0
        n = sum(r.termination in CONVERSION_FAILURES for r in self._results)
        return n / len(self._results)

    # -------------------------------------------------------------- output
    def as_dict(self) -> dict:
        return {
            "n": len(self._results),
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "draw_rate": round(self.draw_rate, 4),
            "loss_rate": round(self.loss_rate, 4),
            "score": round(self.score, 4),
            "mean_plies": round(self.mean_plies, 2),
            "conversion_failure_rate": round(self.conversion_failure_rate, 4),
            "terminations": dict(self.terminations.most_common()),
        }

    def summary_line(self) -> str:
        return (
            f"W/D/L {self.wins:>4}/{self.draws:>4}/{self.losses:>4} | "
            f"win-rate {self.win_rate:5.3f} | plies {self.mean_plies:5.1f} | "
            f"conv-fail {self.conversion_failure_rate:5.3f}"
        )
