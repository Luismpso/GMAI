"""Tests for the exact endgame solver and the supervised warm start.

The solver is validated against *external* ground truth: KQ vs K is a mate in
at most 10 moves, KR vs K in at most 16. Those are textbook results, not
something the implementation could accidentally agree with.
"""

import random
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from gmai.agent import DQNAgent
from gmai.encoding import N_ACTIONS
from gmai.endgames import sample_endgame
from gmai.tablebase import (
    DRAW,
    SOLVABLE,
    EndgameTable,
    get_table,
    index,
    unindex,
)
from gmai.warmstart import build_dataset, dtm_quality, pretrain

CACHE = Path("tablebases")
# Textbook maxima for optimal play, in plies.
KNOWN_MAX_DTM = {"KQvK": 20, "KRvK": 32}


def _table_or_skip(kind: str) -> EndgameTable:
    path = CACHE / f"{kind}.npz"
    if not path.exists():
        pytest.skip(f"{path} not built (run: python -m gmai.tablebase --kind {kind})")
    return EndgameTable.load(path)


class TestIndexing:
    def test_round_trip(self):
        for parts in [(0, 0, 0, 0), (12, 40, 3, 1), (63, 63, 63, 1)]:
            assert unindex(index(*parts)) == parts

    def test_indices_are_unique(self):
        seen = {index(wk, bk, pc, t)
                for wk in range(0, 64, 7)
                for bk in range(0, 64, 5)
                for pc in range(0, 64, 3)
                for t in (0, 1)}
        expected = len(range(0, 64, 7)) * len(range(0, 64, 5)) * len(range(0, 64, 3)) * 2
        assert len(seen) == expected


@pytest.mark.parametrize("kind", SOLVABLE)
class TestSolverGroundTruth:
    def test_max_dtm_matches_chess_theory(self, kind):
        """External validation: KQvK mates in <=10 moves, KRvK in <=16."""
        table = _table_or_skip(kind)
        assert int(table.dtm[table.dtm < DRAW].max()) == KNOWN_MAX_DTM[kind]

    def test_most_positions_are_won(self, kind):
        table = _table_or_skip(kind)
        won = int((table.dtm < DRAW).sum())
        assert won > 200_000  # the vast majority of legal states

    def test_optimal_play_mates_within_the_predicted_dtm(self, kind):
        """Play out optimal-vs-optimal and check the DTM prediction exactly."""
        table = _table_or_skip(kind)
        rng = random.Random(0)
        checked = 0

        for _ in range(40):
            pos = sample_endgame(kind, rng=rng)
            board, strong = pos.board, pos.strong_color
            if board.turn != strong:
                continue
            predicted = table.probe(board, strong)
            if predicted >= DRAW:
                continue

            start = len(board.move_stack)
            while not board.is_game_over(claim_draw=True):
                if len(board.move_stack) - start > 2 * KNOWN_MAX_DTM[kind]:
                    break
                if board.turn == strong:
                    move = table.best_moves(board, strong)[0]
                else:  # optimal defence: maximise the distance to mate
                    move, best = None, -1
                    for candidate in board.legal_moves:
                        board.push(candidate)
                        d = (
                            10**6
                            if board.is_game_over(claim_draw=True)
                            and not board.is_checkmate()
                            else table.probe(board, strong)
                        )
                        board.pop()
                        if d > best:
                            best, move = d, candidate
                board.push(move)

            assert board.is_checkmate(), board.fen()
            assert len(board.move_stack) - start == predicted
            checked += 1

        assert checked > 0


class TestTableLookup:
    def test_mate_in_one_is_dtm_one(self):
        table = _table_or_skip("KQvK")
        board = chess.Board("7k/8/6K1/8/8/8/Q7/8 w - - 0 1")  # Qa8#
        assert board.is_valid()  # a queen on a1 would already be giving check
        assert table.probe(board, chess.WHITE) == 1

    def test_stalemate_trap_is_drawn(self):
        table = _table_or_skip("KQvK")
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # already stalemate
        assert table.probe(board, chess.WHITE) >= DRAW

    def test_black_as_the_strong_side_is_mirrored(self):
        table = _table_or_skip("KQvK")
        white = chess.Board("7k/8/6K1/8/8/8/8/Q7 w - - 0 1")
        black = chess.Board("q7/8/8/8/8/6k1/8/7K b - - 0 1")
        assert table.probe(white, chess.WHITE) == table.probe(black, chess.BLACK)

    def test_best_moves_are_legal_and_reduce_dtm(self):
        table = _table_or_skip("KQvK")
        rng = random.Random(1)
        for _ in range(25):
            pos = sample_endgame("KQvK", rng=rng)
            board, strong = pos.board, pos.strong_color
            if board.turn != strong:
                continue
            before = table.probe(board, strong)
            if before >= DRAW:
                continue
            moves = table.best_moves(board, strong)
            assert moves
            for move in moves:
                assert move in board.legal_moves
            board.push(moves[0])
            after = 0 if board.is_checkmate() else table.probe(board, strong)
            assert after < before

    def test_probe_returns_draw_for_out_of_scope_shapes(self):
        table = _table_or_skip("KQvK")
        assert table.probe(chess.Board(), chess.WHITE) >= DRAW

    def test_save_and_load_round_trip(self, tmp_path):
        table = _table_or_skip("KQvK")
        path = tmp_path / "t.npz"
        table.save(path)
        restored = EndgameTable.load(path)
        assert restored.kind == table.kind
        assert np.array_equal(restored.dtm, table.dtm)

    def test_get_table_refuses_four_piece_endgames(self):
        assert get_table("KRRvK") is None


class TestWarmStart:
    @pytest.fixture
    def data(self):
        table = _table_or_skip("KQvK")
        return build_dataset("KQvK", table, n_positions=200, seed=0)

    def test_dataset_shapes(self, data):
        n = len(data.targets)
        assert n == 200
        assert data.states.shape == (n, 18, 8, 8)
        assert data.masks.shape == (n, N_ACTIONS)

    def test_targets_are_legal_actions(self, data):
        assert data.masks[np.arange(len(data.targets)), data.targets].all()

    def test_pretraining_reduces_loss(self, data):
        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        history = pretrain(agent, data, epochs=4, batch_size=64, verbose=False)
        assert history[-1] < history[0]

    def test_pretraining_syncs_the_target_network(self, data):
        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        pretrain(agent, data, epochs=1, batch_size=64, verbose=False)
        for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
            assert torch.equal(po, pt)


class TestDtmQuality:
    def test_rates_sum_to_one(self):
        table = _table_or_skip("KQvK")
        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        q = dtm_quality(agent, "KQvK", table, n_positions=40, seed=0)
        total = q["optimal_rate"] + q["suboptimal_rate"] + q["throw_away_rate"]
        assert total == pytest.approx(1.0, abs=1e-3)
        assert q["n"] == 40

    def test_optimal_player_scores_perfectly(self):
        """A player following the table must have a 100% optimal rate."""
        table = _table_or_skip("KQvK")

        class OptimalAgent:
            def act(self, board, greedy=True):
                from gmai.encoding import move_to_action

                strong = board.turn
                return move_to_action(table.best_moves(board, strong)[0], board)

        q = dtm_quality(OptimalAgent(), "KQvK", table, n_positions=60, seed=2)
        assert q["optimal_rate"] == 1.0
        assert q["throw_away_rate"] == 0.0
