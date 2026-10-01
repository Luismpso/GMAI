"""Monte Carlo tree search: correctness that does not depend on a trained net.

The network here is tiny and random on purpose. Anything the search gets right
(mates, legality, bookkeeping) is the search's doing, not the network's.
"""

import io
import sys

import chess
import numpy as np
import pytest
import torch

from chessnet.model import ChessNet
from chessnet.play import Player, parse_go, time_budget, uci_loop
from chessnet.search import MCTS, terminal_value


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    model = ChessNet(channels=8, blocks=1)
    model.eval()
    return model


@pytest.fixture
def mcts(net):
    return MCTS(net, "cpu", batch_size=8)


# ------------------------------------------------------------ game rules
@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        # Fool's mate: White to move and checkmated.
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", -1.0),
        # Stalemate: Black to move, not in check, no legal moves.
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.0),
        # Bare kings.
        ("8/8/8/4k3/8/8/8/4K3 w - - 0 1", 0.0),
        # Fifty-move rule.
        ("8/8/8/4k3/8/8/8/R3K3 w - - 100 80", 0.0),
        (chess.STARTING_FEN, None),
    ],
)
def test_terminal_value(fen, expected):
    assert terminal_value(chess.Board(fen)) == expected


# ------------------------------------------------------------- tactics
def test_finds_back_rank_mate_for_white(mcts):
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    result = mcts.run(board, nodes=600)
    assert result.move == chess.Move.from_uci("d1d8")
    assert result.q > 0.9


def test_finds_back_rank_mate_for_black(mcts):
    # Same pattern with colours swapped: catches any sign error in backup.
    board = chess.Board("3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    result = mcts.run(board, nodes=600)
    assert result.move == chess.Move.from_uci("d8d1")
    assert result.q > 0.9


# ---------------------------------------------------------- bookkeeping
def test_result_is_legal_and_visits_add_up(mcts):
    fen = "r1b1kb1r/pppp1ppp/2n2n2/3qp3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 5"
    board = chess.Board(fen)
    result = mcts.run(board, nodes=200)
    assert board.fen() == fen, "search must leave the caller's board untouched"
    assert result.move in board.legal_moves
    # The first simulation expands the root; every later one visits one child.
    assert sum(result.visits.values()) == result.nodes - 1
    assert result.pv[0] == result.move
    replay = board.copy()
    for move in result.pv:
        assert move in replay.legal_moves
        replay.push(move)


def test_single_legal_move_terminates(mcts):
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")  # only Kxg2
    result = mcts.run(board, nodes=50)
    assert result.move == chess.Move.from_uci("h1g2")


def test_temperature_sampling_stays_legal(mcts):
    board = chess.Board()
    rng = np.random.default_rng(1)
    for _ in range(3):
        result = mcts.run(board, nodes=40, temperature=1.0, rng=rng)
        assert result.move in board.legal_moves


def test_refuses_finished_games(mcts):
    with pytest.raises(ValueError):
        mcts.run(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), nodes=10)


# ------------------------------------------------------------------ UCI
def test_parse_go_and_time_budget():
    params = parse_go(["wtime", "60000", "btime", "50000", "winc", "0", "binc", "0"])
    assert params == {"wtime": 60000, "btime": 50000, "winc": 0, "binc": 0}
    assert time_budget(chess.Board(), params) == pytest.approx(2.0)
    assert time_budget(chess.Board(), {"movetime": 250}) == pytest.approx(0.25)
    assert time_budget(chess.Board(), {}) is None
    # Never spend more than a quarter of what is left.
    assert time_budget(chess.Board(), {"wtime": 400, "movestogo": 1}) == pytest.approx(
        0.1
    )


def test_uci_go_nodes_returns_legal_move(tmp_path, monkeypatch, net):
    path = tmp_path / "tiny.pt"
    net.save(path)
    player = Player(str(path), device="cpu", nodes=32)
    script = "uci\nposition startpos moves e2e4\ngo nodes 32\nquit\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(script))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    uci_loop(player)
    lines = out.getvalue().splitlines()
    best = [ln for ln in lines if ln.startswith("bestmove")][-1].split()[1]
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(best) in board.legal_moves
    assert any(ln.startswith("info depth") and " pv " in ln for ln in lines)
