"""Reward design.

Terminal rewards: +1 win / -1 loss / ``draw_reward`` on draws.

Optional *potential-based reward shaping* (Ng, Harada & Russell, 1999) on
the material balance: F(s, s') = gamma * phi(s') - phi(s). Because the
shaping term telescopes along a trajectory, the optimal policy is provably
unchanged — the agent just receives a denser learning signal early on.
"""

from __future__ import annotations

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


def material_balance(board: chess.Board, color: chess.Color) -> float:
    """Material of ``color`` minus material of the opponent, in pawns."""
    balance = 0.0
    for piece_type, value in PIECE_VALUES.items():
        balance += value * len(board.pieces(piece_type, color))
        balance -= value * len(board.pieces(piece_type, not color))
    return balance


def potential(board: chess.Board, color: chess.Color) -> float:
    """phi(s): normalised material balance in [-1, 1]."""
    return material_balance(board, color) / _MAX_MATERIAL


def shaping_reward(
    board_before: chess.Board,
    board_after: chess.Board,
    color: chess.Color,
    gamma: float,
) -> float:
    """Potential-based shaping term F(s, s') = gamma * phi(s') - phi(s)."""
    return gamma * potential(board_after, color) - potential(board_before, color)


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
