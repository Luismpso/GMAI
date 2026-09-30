"""Training and inference must see the same position and the same move.

A mismatch here is silent: loss goes down, validation accuracy looks fine on
the half of the data that happens to agree, and the model plays badly on the
other half. This caught a real bug — every label for Black-to-move positions
was in the wrong frame.
"""

import random

import chess
import numpy as np
import pytest

from chessnet.dataset import orient_actions
from chessnet.encoding import (
    PIECE_ORDER,
    action_to_move,
    bitboards_to_planes,
    encode_board,
    move_to_action,
)


def _stored(board: chess.Board, move: chess.Move):
    """Exactly what scripts/extract_lichess.py writes for one position."""
    bitboards = np.array(
        [
            int(board.pieces(pt, c))
            for c in (chess.WHITE, chess.BLACK)
            for pt in PIECE_ORDER
        ],
        dtype=np.uint64,
    )
    castling = (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    )
    meta = np.array(
        [
            int(board.turn),
            castling,
            board.ep_square if board.ep_square is not None else 64,
            min(board.halfmove_clock, 255),
        ],
        dtype=np.uint8,
    )
    return bitboards, meta, move.from_square * 64 + move.to_square


def _random_positions(n: int = 80, seed: int = 0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 50)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        moves = list(board.legal_moves)
        if moves:
            out.append((board.copy(), rng.choice(moves)))
    return out


@pytest.fixture(scope="module")
def samples():
    positions = _random_positions()
    assert any(b.turn == chess.BLACK for b, _ in positions)
    assert any(b.turn == chess.WHITE for b, _ in positions)
    return positions


def test_planes_match_between_training_and_inference(samples):
    stored = [_stored(b, m) for b, m in samples]
    train = bitboards_to_planes(
        np.stack([s[0] for s in stored]), np.stack([s[1] for s in stored])
    )
    play = np.stack([encode_board(b) for b, _ in samples])
    assert np.array_equal(train, play)


def test_labels_match_between_training_and_inference(samples):
    stored = [_stored(b, m) for b, m in samples]
    labels = orient_actions(
        np.array([s[2] for s in stored], dtype=np.int64),
        np.array([s[1][0] for s in stored]),
    )
    expected = np.array([move_to_action(m, b) for b, m in samples])
    assert np.array_equal(labels, expected)


def test_every_training_label_decodes_to_the_move_played(samples):
    stored = [_stored(b, m) for b, m in samples]
    labels = orient_actions(
        np.array([s[2] for s in stored], dtype=np.int64),
        np.array([s[1][0] for s in stored]),
    )
    for (board, move), label in zip(samples, labels, strict=True):
        decoded = action_to_move(int(label), board)
        assert (decoded.from_square, decoded.to_square) == (
            move.from_square,
            move.to_square,
        )


def test_orientation_is_a_no_op_for_white():
    actions = np.array([12 * 64 + 28, 6 * 64 + 21])
    assert np.array_equal(orient_actions(actions, np.array([1, 1])), actions)
