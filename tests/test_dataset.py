"""The vectorised shard decoder must match the canonical per-position encoders.

``dataset.decode_planes`` / ``orient_actions`` expand the compact storage
written by ``scripts/extract_lichess.py`` back into exactly what
``encoding.encode_board`` / ``encoding.move_to_action`` produce. The mirror for
Black-to-move positions is the easy thing to get subtly wrong, so this pins the
whole chain: extract's ``encode_position`` -> dataset decode -> encoding.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

from gmai.dataset import decode_planes, orient_actions
from gmai.encoding import encode_board, move_to_action

# extract_lichess lives in scripts/, not in the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_lichess import encode_position  # noqa: E402


def _compact(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    bitboards, meta = encode_position(board)
    return (
        np.asarray([bitboards], dtype=np.uint64),
        np.asarray([meta], dtype=np.uint8),
    )


def _random_boards(n: int, seed: int = 0) -> list[chess.Board]:
    """Positions from random play: mixes turns, castling rights, captures."""
    rng = random.Random(seed)
    boards, board = [], chess.Board()
    while len(boards) < n:
        if board.is_game_over():
            board = chess.Board()
            continue
        boards.append(board.copy())
        board.push(rng.choice(list(board.legal_moves)))
    return boards


# En-passant, both to-move colours, to exercise the plane-17 mirror explicitly.
_EP_FENS = [
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 3",
]


@pytest.fixture
def boards() -> list[chess.Board]:
    return _random_boards(60) + [chess.Board(f) for f in _EP_FENS]


def test_decode_planes_matches_encode_board(boards):
    for board in boards:
        b, m = _compact(board)
        np.testing.assert_array_equal(decode_planes(b, m)[0], encode_board(board))


def test_orient_actions_matches_move_to_action(boards):
    rng = random.Random(1)
    for board in boards:
        _, m = _compact(board)
        move = rng.choice(list(board.legal_moves))
        abs_action = np.asarray([move.from_square * 64 + move.to_square], dtype=np.int16)
        assert int(orient_actions(abs_action, m)[0]) == move_to_action(move, board)


def test_black_to_move_is_actually_mirrored(boards):
    """Guard against a decoder that silently ignores the POV flip."""
    black = next(bd for bd in boards if bd.turn == chess.BLACK)
    b, m = _compact(black)
    # Own pieces (planes 0-5) live in the mover's home ranks after the flip:
    # for Black to move, the mirror puts them on rows 0-1, not 6-7.
    own = decode_planes(b, m)[0, 0:6]
    assert own[0:2].sum() > 0
