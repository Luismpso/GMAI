"""Opponent policies for the curriculum.

The training curriculum mirrors *domain randomization* from robotics RL:
instead of randomising physics, we randomise the adversary. The agent
graduates from a random mover, to a greedy material grabber, to frozen
copies of itself sampled from a checkpoint pool (self-play).
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod

import chess

from .rewards import PIECE_VALUES, material_balance


class Opponent(ABC):
    """Interface: given a board, return a legal move."""

    name: str = "opponent"

    @abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move: ...


class RandomOpponent(Opponent):
    """Uniform random legal move — curriculum stage 1."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))


class GreedyMaterialOpponent(Opponent):
    """1-ply material maximiser with random tie-breaking — stage 2.

    Prefers checkmate if available, otherwise plays the move with the best
    immediate material balance. Epsilon-random to avoid being fully
    deterministic (and therefore trivially exploitable).
    """

    name = "greedy"

    def __init__(self, epsilon: float = 0.1, seed: int | None = None):
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal)

        color = board.turn
        best_moves: list[chess.Move] = []
        best_score = -float("inf")
        for move in legal:
            board.push(move)
            if board.is_checkmate():
                score = float("inf")
            else:
                score = material_balance(board, color)
            board.pop()
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        return self._rng.choice(best_moves)


class AgentOpponent(Opponent):
    """Wrap a (frozen) agent so it can act as the adversary — stage 3."""

    name = "self-play"

    def __init__(self, agent):
        self.agent = agent

    def select_move(self, board: chess.Board) -> chess.Move:
        from .encoding import action_to_move  # local import: avoid cycle

        action = self.agent.act(board, greedy=True)
        return action_to_move(action, board)


class OpponentPool:
    """Checkpoint pool for self-play with opponent randomization.

    Keeps up to ``capacity`` frozen snapshots; ``sample`` returns a random
    one so the learner faces a distribution of past selves rather than
    overfitting to its latest incarnation (the classic self-play trap).
    """

    def __init__(self, capacity: int = 5, seed: int | None = None):
        self.capacity = capacity
        self._pool: list[AgentOpponent] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._pool)

    def add_snapshot(self, agent) -> None:
        frozen = copy.deepcopy(agent)
        frozen.epsilon = 0.0
        frozen.online.eval()
        self._pool.append(AgentOpponent(frozen))
        if len(self._pool) > self.capacity:
            self._pool.pop(0)

    def sample(self) -> AgentOpponent:
        if not self._pool:
            raise RuntimeError("opponent pool is empty — add a snapshot first")
        return self._rng.choice(self._pool)
