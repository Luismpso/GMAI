import chess
import pytest

from gmai.endgames import ENDGAME_ORDER, ENDGAME_SPECS, sample_endgame
from gmai.metrics import RollingStats, classify_episode


def _mated_board():
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(san)
    return board


class TestClassifyEpisode:
    def test_win_for_the_mating_side(self):
        board = _mated_board()
        result = classify_episode(board, chess.BLACK, truncated=False)
        assert result.win and not result.draw and not result.loss
        assert result.termination == "CHECKMATE"
        assert result.score == 1.0

    def test_loss_for_the_mated_side(self):
        result = classify_episode(_mated_board(), chess.WHITE, truncated=False)
        assert result.loss and not result.win
        assert result.score == 0.0

    def test_stalemate_is_a_draw(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        result = classify_episode(board, chess.WHITE, truncated=False)
        assert result.draw
        assert result.termination == "STALEMATE"

    def test_unfinished_game_is_truncated(self):
        result = classify_episode(chess.Board(), chess.WHITE, truncated=True)
        assert result.draw
        assert result.termination == "TRUNCATED"

    def test_plies_counted(self):
        result = classify_episode(_mated_board(), chess.BLACK, truncated=False)
        assert result.plies == 4


class TestRollingStats:
    def _stats(self, wins, draws, losses, window=100):
        stats = RollingStats(window=window)
        mate, stale = _mated_board(), chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        for _ in range(wins):
            stats.add(classify_episode(mate, chess.BLACK, False))
        for _ in range(draws):
            stats.add(classify_episode(stale, chess.WHITE, False))
        for _ in range(losses):
            stats.add(classify_episode(mate, chess.WHITE, False))
        return stats

    def test_wdl_are_separated(self):
        stats = self._stats(3, 5, 2)
        assert (stats.wins, stats.draws, stats.losses) == (3, 5, 2)

    def test_win_rate_ignores_draws_but_score_does_not(self):
        """The whole point of separating them: these two numbers differ."""
        stats = self._stats(1, 8, 1)
        assert stats.win_rate == pytest.approx(0.1)
        assert stats.score == pytest.approx(0.5)  # the metric that hid the bugs

    def test_conversion_failure_rate_counts_stalemates(self):
        stats = self._stats(2, 8, 0)
        assert stats.conversion_failure_rate == pytest.approx(0.8)

    def test_window_evicts_oldest(self):
        stats = self._stats(50, 50, 0, window=10)
        assert len(stats) == 10
        assert stats.total_episodes == 100

    def test_clear_resets_window_not_total(self):
        stats = self._stats(5, 5, 0)
        stats.clear()
        assert len(stats) == 0 and stats.total_episodes == 10
        assert stats.win_rate == 0.0

    def test_is_full_flag(self):
        stats = self._stats(3, 0, 0, window=5)
        assert not stats.is_full
        stats = self._stats(5, 0, 0, window=5)
        assert stats.is_full

    def test_as_dict_has_separated_counts(self):
        d = self._stats(3, 5, 2).as_dict()
        assert d["wins"] == 3 and d["draws"] == 5 and d["losses"] == 2
        assert "win_rate" in d and "terminations" in d


class TestEndgameSampler:
    @pytest.mark.parametrize("kind", ENDGAME_ORDER)
    def test_positions_are_legal_and_playable(self, kind):
        import random

        rng = random.Random(0)
        for _ in range(120):
            pos = sample_endgame(kind, rng=rng)
            assert pos.board.is_valid(), pos.board.fen()
            assert not pos.board.is_game_over(claim_draw=True)
            assert list(pos.board.legal_moves)

    @pytest.mark.parametrize("kind,n_pieces", [("KQvK", 3), ("KRvK", 3), ("KRRvK", 4)])
    def test_piece_counts(self, kind, n_pieces):
        import random

        pos = sample_endgame(kind, rng=random.Random(1))
        assert len(pos.board.piece_map()) == n_pieces

    def test_kings_are_never_adjacent(self):
        import random

        rng = random.Random(2)
        for _ in range(150):
            board = sample_endgame("KQvK", rng=rng).board
            assert chess.square_distance(
                board.king(chess.WHITE), board.king(chess.BLACK)
            ) > 1

    def test_weak_side_has_only_a_king(self):
        import random

        rng = random.Random(3)
        for _ in range(60):
            pos = sample_endgame("KRRvK", rng=rng)
            weak_pieces = [
                p for p in pos.board.piece_map().values() if p.color != pos.strong_color
            ]
            assert len(weak_pieces) == 1
            assert weak_pieces[0].piece_type == chess.KING

    def test_strong_color_can_be_forced(self):
        import random

        rng = random.Random(4)
        for _ in range(20):
            pos = sample_endgame("KQvK", rng=rng, strong_color=chess.BLACK)
            assert pos.strong_color == chess.BLACK
            assert pos.board.pieces(chess.QUEEN, chess.BLACK)

    def test_max_moves_matches_spec(self):
        import random

        for kind, (_, max_moves) in ENDGAME_SPECS.items():
            pos = sample_endgame(kind, rng=random.Random(5))
            assert pos.max_moves == max_moves

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown endgame"):
            sample_endgame("KBNvK")

    def test_sampler_is_reproducible(self):
        import random

        a = sample_endgame("KQvK", rng=random.Random(7)).board.fen()
        b = sample_endgame("KQvK", rng=random.Random(7)).board.fen()
        assert a == b
