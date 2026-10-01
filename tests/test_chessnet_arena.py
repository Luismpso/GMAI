"""Matches against Stockfish: openings, rating maths and one real short match."""

import io
import math
import shutil
from pathlib import Path

import chess.pgn
import pytest
import torch

from chessnet.arena import (
    OPENINGS,
    fit_rating,
    opening_board,
    run_match,
    score_to_elo,
    summarise,
)
from chessnet.model import ChessNet
from chessnet.play import Player


def _find_stockfish():
    found = shutil.which("stockfish")
    if found:
        return found
    for candidate in ("/usr/games/stockfish", "/usr/local/bin/stockfish"):
        if Path(candidate).exists():
            return candidate
    return None


STOCKFISH = _find_stockfish()


def test_openings_are_legal_distinct_and_eight_plies():
    positions = set()
    for name, line in OPENINGS:
        board = opening_board(line)
        assert board.ply() == 8, name
        positions.add(board.board_fen())
    assert len(positions) == len(OPENINGS)


def test_score_to_elo():
    assert score_to_elo(0.5) == pytest.approx(0)
    assert score_to_elo(0.75) == pytest.approx(400 * math.log10(3))
    assert score_to_elo(0.25) == pytest.approx(-score_to_elo(0.75))


def test_single_level_fit_matches_the_classic_formula():
    # With one opponent rating, the likelihood peaks exactly at
    # level + 400 log10(s / (1 - s)).
    results = [(2000, 1.0)] * 3 + [(2000, 0.5)] * 2 + [(2000, 0.0)]
    rating, low, high = fit_rating(results)
    assert rating == pytest.approx(2000 + score_to_elo(4 / 6), abs=1)
    assert low < rating < high


def test_fit_recovers_a_symmetric_result():
    results = (
        [(1500, 1.0)] * 6 + [(1500, 0.0)] * 2
        + [(1800, 1.0)] * 4 + [(1800, 0.0)] * 4
        + [(2100, 1.0)] * 2 + [(2100, 0.0)] * 6
    )  # fmt: skip
    rating, low, high = fit_rating(results)
    assert rating == pytest.approx(1800, abs=1)
    assert low < 1800 < high


def test_fit_reports_open_bounds_for_perfect_scores():
    _, low, high = fit_rating([(1500, 1.0)] * 10)
    assert high is None and low > 1500
    _, low, high = fit_rating([(1500, 0.0)] * 10)
    assert low is None and high < 1500


@pytest.mark.skipif(STOCKFISH is None, reason="Stockfish not installed")
def test_short_match_against_stockfish(tmp_path):
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    ChessNet(channels=8, blocks=1).save(checkpoint)
    player = Player(str(checkpoint), device="cpu")

    records = run_match(
        player, STOCKFISH, levels=[1320], games=2, nodes=16, movetime=None,
        sf_time=0.01, max_plies=60, out_dir=tmp_path / "arena",
    )  # fmt: skip

    assert len(records) == 2
    assert [r["chessnet_white"] for r in records] == [True, False]
    assert all(r["score"] in (0.0, 0.5, 1.0) for r in records)
    assert "estimated strength" in summarise(records)

    pgn = io.StringIO((tmp_path / "arena" / "games.pgn").read_text())
    first = chess.pgn.read_game(pgn)
    played = [m.uci() for m in first.mainline_moves()][:8]
    expected = [m.uci() for m in opening_board(OPENINGS[0][1]).move_stack]
    assert played == expected, "games must start from the chosen opening"
    assert chess.pgn.read_game(pgn) is not None, "both games are saved"
