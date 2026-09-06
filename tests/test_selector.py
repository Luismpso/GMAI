"""Move selection: legality, top-k, the policy/value blend and mate handling.

These run against a randomly initialised network — the point is the *selection
logic*, not playing strength, so untrained weights are fine.
"""

from __future__ import annotations

import chess

from gmai.model import PolicyValueNet
from gmai.selector import MoveSelector


def _net():
    return PolicyValueNet(channels=16, n_blocks=2, hidden=64)


def test_returns_legal_move_and_is_seed_deterministic():
    board = chess.Board()
    net = _net()
    m1 = MoveSelector(net, seed=42).select(board.copy())
    m2 = MoveSelector(net, seed=42).select(board.copy())
    assert m1 in board.legal_moves
    assert m1 == m2  # same net, same seed -> same sampled move


def test_candidates_capped_at_top_k():
    board = chess.Board()  # 20 legal moves from the start
    cands = MoveSelector(_net(), top_k=3, seed=0).candidates(board)
    assert len(cands) == 3
    assert all(c.move in board.legal_moves for c in cands)


def test_plays_an_available_mate_in_one():
    # Back-rank mate: any rook move to the 8th rank is checkmate. With top_k
    # covering every legal move, the mate must be among the candidates and its
    # +inf success has to win the blend outright.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    sel = MoveSelector(_net(), top_k=board.legal_moves.count(), seed=0)
    move = sel.select(board)
    board.push(move)
    assert board.is_checkmate()


def test_select_returns_none_when_game_is_over():
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate
        board.push_uci(uci)
    assert board.is_checkmate()
    assert MoveSelector(_net(), seed=0).select(board) is None
