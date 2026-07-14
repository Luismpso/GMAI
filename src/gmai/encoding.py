"""Board and move encoding.

State  : 18 binary planes of shape (18, 8, 8), always from the point of
         view of the side to move (the board is flipped for Black), so the
         network learns a single, colour-agnostic representation.
Action : ``from_square * 64 + to_square`` -> 4096 discrete actions.
         Promotions default to a queen (under-promotion is a documented
         simplification, see README roadmap).

Plane layout
------------
 0-5   own pieces      (P, N, B, R, Q, K)
 6-11  opponent pieces (P, N, B, R, Q, K)
 12    side to move    (all ones if White to move, zeros otherwise)
 13-16 castling rights (own K-side, own Q-side, opp K-side, opp Q-side)
 17    en-passant target square
"""

from __future__ import annotations

import chess
import numpy as np

N_PLANES = 18
N_ACTIONS = 64 * 64  # from-square x to-square
PIECE_ORDER = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def _oriented_square(square: int, pov_white: bool) -> int:
    """Mirror the square vertically when encoding from Black's perspective."""
    return square if pov_white else chess.square_mirror(square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode ``board`` as an (18, 8, 8) float32 tensor from the mover's POV."""
    pov_white = board.turn == chess.WHITE
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        sq = _oriented_square(square, pov_white)
        row, col = divmod(sq, 8)
        own = piece.color == board.turn
        base = 0 if own else 6
        planes[base + PIECE_ORDER.index(piece.piece_type), row, col] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    us, them = board.turn, not board.turn
    planes[13, :, :] = float(board.has_kingside_castling_rights(us))
    planes[14, :, :] = float(board.has_queenside_castling_rights(us))
    planes[15, :, :] = float(board.has_kingside_castling_rights(them))
    planes[16, :, :] = float(board.has_queenside_castling_rights(them))

    if board.ep_square is not None:
        sq = _oriented_square(board.ep_square, pov_white)
        row, col = divmod(sq, 8)
        planes[17, row, col] = 1.0

    return planes


def move_to_action(move: chess.Move, board: chess.Board) -> int:
    """Map a ``chess.Move`` to a discrete action id (mover's POV)."""
    pov_white = board.turn == chess.WHITE
    from_sq = _oriented_square(move.from_square, pov_white)
    to_sq = _oriented_square(move.to_square, pov_white)
    return from_sq * 64 + to_sq


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Map an action id back to a legal ``chess.Move`` on ``board``.

    Promotions are resolved to a queen. Raises ``ValueError`` if the action
    does not correspond to a legal move in the current position.
    """
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action {action} outside [0, {N_ACTIONS})")

    pov_white = board.turn == chess.WHITE
    from_sq = _oriented_square(action // 64, pov_white)
    to_sq = _oriented_square(action % 64, pov_white)

    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        return move

    promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if promo in board.legal_moves:
        return promo

    raise ValueError(f"action {action} is illegal in this position")


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (4096,) with ``True`` for legal actions."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_action(move, board)] = True
    return mask
