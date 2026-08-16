"""Reward design.

Terminal rewards: +1 win / -1 loss / ``draw_reward`` on draws.

Potential-based reward shaping (Ng, Harada & Russell, 1999):

    F(s, s') = gamma * Phi(s') - Phi(s)

The policy-invariance theorem requires ``Phi(terminal) = 0``. Skipping the
shaping term on the final transition — an easy mistake, and one this codebase
made — breaks the telescoping sum and voids the guarantee. The terminal term
is therefore *included* with ``Phi = 0``, so the discounted shaping along
any episode collapses to ``-Phi(s_0)`` — a constant independent of the policy,
which is exactly what makes the optimal policy unchanged.

Two potentials are provided:

``material_potential``
    Normalised material balance. Right for full chess.
``endgame_potential``
    Material (dominant: don't hang the queen) + drive the enemy king towards
    the edge + bring your own king closer. The classic mating heuristic,
    expressed as a potential so it stays policy-invariant.
"""

from __future__ import annotations

from typing import Callable

import chess

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
# Maximum realistic material advantage used for normalisation:
# 8P + 2N + 2B + 2R + 1Q = 39.
_MAX_MATERIAL = 39.0

# Weights for the endgame potential. Material dominates: losing the queen in
# KQ vs K turns a forced win into a dead draw, so it must outweigh any amount
# of king manoeuvring.
W_MATERIAL = 2.0
W_EDGE = 0.30
W_PROXIMITY = 0.20

PotentialFn = Callable[[chess.Board, chess.Color], float]


def material_balance(board: chess.Board, color: chess.Color) -> float:
    """Material of ``color`` minus material of the opponent, in pawns."""
    balance = 0.0
    for piece_type, value in PIECE_VALUES.items():
        balance += value * len(board.pieces(piece_type, color))
        balance -= value * len(board.pieces(piece_type, not color))
    return balance


def material_potential(board: chess.Board, color: chess.Color) -> float:
    """Phi(s) for full chess: normalised material balance in [-1, 1]."""
    return material_balance(board, color) / _MAX_MATERIAL


def _edge_distance(square: int) -> float:
    """Chebyshev distance from the board centre, normalised to [0, 1].

    0.0 = dead centre, 1.0 = on the rim. Driving the defending king here is
    a precondition for mate with a queen or rook.
    """
    file_, rank = chess.square_file(square), chess.square_rank(square)
    d = max(abs(file_ - 3.5), abs(rank - 3.5))  # in [0.5, 3.5]
    return (d - 0.5) / 3.0


def _king_proximity(board: chess.Board, color: chess.Color) -> float:
    """Closeness of the two kings, normalised to [0, 1] (1.0 = adjacent).

    The attacking king must join in; a queen or rook alone cannot mate.
    """
    ours, theirs = board.king(color), board.king(not color)
    if ours is None or theirs is None:
        return 0.0
    return (7 - chess.square_distance(ours, theirs)) / 7.0


def endgame_potential(board: chess.Board, color: chess.Color) -> float:
    """Phi(s) for forced-mate endgames: material + edge-drive + king proximity."""
    enemy_king = board.king(not color)
    if enemy_king is None:
        return 0.0
    return (
        W_MATERIAL * material_potential(board, color)
        + W_EDGE * _edge_distance(enemy_king)
        + W_PROXIMITY * _king_proximity(board, color)
    )


def shaping_reward(
    board_before: chess.Board,
    board_after: chess.Board,
    color: chess.Color,
    gamma: float,
    potential_fn: PotentialFn = material_potential,
    after_is_terminal: bool = False,
) -> float:
    """Potential-based shaping term ``F(s, s') = gamma * Phi(s') - Phi(s)``.

    ``after_is_terminal`` forces ``Phi(s') = 0``, which is required for the
    policy-invariance theorem to hold in episodic tasks. Truncation is *not*
    termination: a time-limit cut-off should pass ``False`` here, because the
    episode is still notionally ongoing.
    """
    phi_after = 0.0 if after_is_terminal else potential_fn(board_after, color)
    return gamma * phi_after - potential_fn(board_before, color)


def terminal_reward(
    board: chess.Board, color: chess.Color, draw_reward: float = 0.0
) -> float:
    """Reward for ``color`` in a finished game (0.0 if not over)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.0
    if outcome.winner is None:
        return draw_reward
    return 1.0 if outcome.winner == color else -1.0


# Backwards-compatible alias (older code and notebooks import `potential`).
potential = material_potential

POTENTIALS: dict[str, PotentialFn] = {
    "material": material_potential,
    "endgame": endgame_potential,
    "none": lambda board, color: 0.0,
}
