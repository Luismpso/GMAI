"""Board representation.

Shards store piece bitboards (12 uint64 plus 4 metadata bytes). This module
expands them into the planes the network consumes, in bulk, on the fly.

Storing planes directly would cost 4.6 KB per position — 460 GB for 100 million
— against roughly 5 bytes compressed as bitboards. Expansion is a couple of
numpy operations and happens per batch, so it costs far less than the I/O it
saves.

Planes are always from the point of view of the side to move: for Black the
board is mirrored vertically and the colours swapped, so the network sees one
colour-agnostic problem instead of two mirrored ones.

    0-5    own pieces      (P N B R Q K)
    6-11   opponent pieces (P N B R Q K)
    12     side to move (all ones when White, for orientation debugging)
    13-16  castling rights (own K, own Q, opponent K, opponent Q)
    17     en-passant target square
    18     halfmove clock, scaled to [0, 1]
"""

from __future__ import annotations

import chess
import numpy as np

N_PLANES = 19
N_ACTIONS = 64 * 64
PIECE_ORDER = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# Mirroring a square vertically is a XOR with 56 on the square index, which
# vectorises to a reshape-and-flip on the (8, 8) plane.
_BIT = np.arange(64, dtype=np.uint64)


def bitboards_to_planes(boards: np.ndarray, metas: np.ndarray) -> np.ndarray:
    """Expand a batch of stored positions into network input planes.

    Parameters
    ----------
    boards : (B, 12) uint64
        Piece bitboards, White first then Black, in ``PIECE_ORDER``.
    metas : (B, 4) uint8
        ``(turn, castling_bits, ep_square_or_64, halfmove_clock)``.

    Returns
    -------
    (B, 19, 8, 8) float32, from the side-to-move's point of view.
    """
    batch = boards.shape[0]
    # Unpack each bitboard into 64 bits: (B, 12, 64)
    bits = ((boards[:, :, None] >> _BIT[None, None, :]) & np.uint64(1)).astype(np.float32)
    squares = bits.reshape(batch, 12, 8, 8)

    white_to_move = metas[:, 0].astype(bool)
    planes = np.zeros((batch, N_PLANES, 8, 8), dtype=np.float32)

    # Own pieces first. For Black to move, swap the colour blocks.
    own = np.where(white_to_move[:, None, None, None], squares[:, :6], squares[:, 6:])
    opp = np.where(white_to_move[:, None, None, None], squares[:, 6:], squares[:, :6])
    planes[:, 0:6] = own
    planes[:, 6:12] = opp
    planes[:, 12] = white_to_move[:, None, None].astype(np.float32)

    castling = metas[:, 1]
    white_k = (castling & 1) > 0
    white_q = (castling & 2) > 0
    black_k = (castling & 4) > 0
    black_q = (castling & 8) > 0
    planes[:, 13] = np.where(white_to_move, white_k, black_k)[:, None, None]
    planes[:, 14] = np.where(white_to_move, white_q, black_q)[:, None, None]
    planes[:, 15] = np.where(white_to_move, black_k, white_k)[:, None, None]
    planes[:, 16] = np.where(white_to_move, black_q, white_q)[:, None, None]

    ep = metas[:, 2].astype(np.int32)
    has_ep = ep < 64
    if has_ep.any():
        idx = np.flatnonzero(has_ep)
        rows, cols = np.divmod(ep[idx], 8)
        planes[idx, 17, rows, cols] = 1.0

    planes[:, 18] = (metas[:, 3].astype(np.float32) / 100.0)[:, None, None]

    # Flip the board for Black so "forward" is always up.
    flip = ~white_to_move
    if flip.any():
        planes[flip] = planes[flip][:, :, ::-1, :]
    return planes


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a live ``chess.Board`` the same way the shards are decoded.

    Used at inference time. Must agree with :func:`bitboards_to_planes`
    exactly — a mismatch here trains on one representation and plays on
    another, which shows up as an agent that scores well in validation and
    blunders in real games. ``tests/test_encoding.py`` pins the two together.
    """
    bitboards = np.array(
        [
            int(board.pieces(pt, colour))
            for colour in (chess.WHITE, chess.BLACK)
            for pt in PIECE_ORDER
        ],
        dtype=np.uint64,
    )[None, :]
    castling = (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    )
    meta = np.array(
        [
            [
                int(board.turn),
                castling,
                board.ep_square if board.ep_square is not None else 64,
                min(board.halfmove_clock, 255),
            ]
        ],
        dtype=np.uint8,
    )
    return bitboards_to_planes(bitboards, meta)[0]


def move_to_action(move: chess.Move, board: chess.Board) -> int:
    """`from * 64 + to`, mirrored for Black so it matches the flipped planes."""
    if board.turn == chess.WHITE:
        return move.from_square * 64 + move.to_square
    return chess.square_mirror(move.from_square) * 64 + chess.square_mirror(
        move.to_square
    )


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Inverse of :func:`move_to_action`, resolving promotions to a queen."""
    from_sq, to_sq = divmod(action, 64)
    if board.turn != chess.WHITE:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        return move
    promotion = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if promotion in board.legal_moves:
        return promotion
    raise ValueError(f"action {action} is not legal in {board.fen()}")


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_action(move, board)] = True
    return mask
