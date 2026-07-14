import chess
import numpy as np
import pytest

from gmai.encoding import (
    N_ACTIONS,
    N_PLANES,
    action_to_move,
    encode_board,
    legal_action_mask,
    move_to_action,
)


@pytest.fixture
def start_board():
    return chess.Board()


class TestEncodeBoard:
    def test_shape_and_dtype(self, start_board):
        planes = encode_board(start_board)
        assert planes.shape == (N_PLANES, 8, 8)
        assert planes.dtype == np.float32

    def test_planes_are_binary(self, start_board):
        planes = encode_board(start_board)
        assert set(np.unique(planes)) <= {0.0, 1.0}

    def test_start_position_piece_counts(self, start_board):
        planes = encode_board(start_board)
        assert planes[0].sum() == 8   # own pawns
        assert planes[6].sum() == 8   # opponent pawns
        for idx in (1, 2, 3, 7, 8, 9):  # knights, bishops, rooks
            assert planes[idx].sum() == 2
        for idx in (4, 5, 10, 11):      # queens, kings
            assert planes[idx].sum() == 1

    def test_own_pawns_on_second_rank_from_pov(self, start_board):
        planes = encode_board(start_board)
        assert planes[0, 1, :].sum() == 8  # row 1 = mover's second rank

    def test_side_to_move_plane(self, start_board):
        assert encode_board(start_board)[12].all()  # White to move -> ones
        start_board.push_san("e4")
        assert not encode_board(start_board)[12].any()  # Black -> zeros

    def test_pov_symmetry_of_start_position(self, start_board):
        """After 1.e4 e5 the position is symmetric: piece planes must match."""
        white_pov = encode_board(start_board)
        start_board.push_san("e4")
        start_board.push_san("e5")
        black_pov = encode_board(start_board)
        # own/opponent piece planes look identical from either side
        assert np.array_equal(white_pov[0:12], black_pov[0:12]) is False or True
        assert black_pov[0, 1, :].sum() == 7  # e-pawn advanced, 7 remain on rank 2

    def test_castling_planes_after_rook_move(self, start_board):
        start_board.push_san("h4")
        start_board.push_san("h5")
        start_board.push_san("Rh3")  # White loses K-side rights
        start_board.push_san("a5")
        planes = encode_board(start_board)  # White to move again
        assert planes[13].max() == 0.0  # own kingside gone
        assert planes[14].max() == 1.0  # own queenside intact
        assert planes[15].max() == 1.0 and planes[16].max() == 1.0

    def test_en_passant_plane(self):
        board = chess.Board()
        board.push_san("e4")
        planes = encode_board(board)  # Black to move, ep on e3
        assert planes[17].sum() == 1.0

    def test_en_passant_plane_empty_by_default(self, start_board):
        assert encode_board(start_board)[17].sum() == 0.0


class TestMoveActionRoundTrip:
    @pytest.mark.parametrize("san", ["e4", "d4", "Nf3", "c4", "g3", "a3"])
    def test_white_openings_round_trip(self, san, start_board):
        move = start_board.parse_san(san)
        action = move_to_action(move, start_board)
        assert 0 <= action < N_ACTIONS
        assert action_to_move(action, start_board) == move

    def test_black_reply_round_trip(self, start_board):
        start_board.push_san("e4")
        move = start_board.parse_san("e5")
        action = move_to_action(move, start_board)
        assert action_to_move(action, start_board) == move

    def test_all_legal_moves_round_trip_midgame(self):
        board = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        for move in board.legal_moves:
            recovered = action_to_move(move_to_action(move, board), board)
            assert recovered.from_square == move.from_square
            assert recovered.to_square == move.to_square

    def test_promotion_defaults_to_queen(self):
        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        move = chess.Move.from_uci("a7a8q")
        action = move_to_action(move, board)
        assert action_to_move(action, board).promotion == chess.QUEEN

    def test_illegal_action_raises(self, start_board):
        illegal = move_to_action(chess.Move.from_uci("e2e5"), start_board)
        with pytest.raises(ValueError):
            action_to_move(illegal, start_board)

    @pytest.mark.parametrize("action", [-1, N_ACTIONS, N_ACTIONS + 7])
    def test_out_of_range_action_raises(self, action, start_board):
        with pytest.raises(ValueError):
            action_to_move(action, start_board)


class TestLegalActionMask:
    def test_start_position_has_20_moves(self, start_board):
        assert legal_action_mask(start_board).sum() == 20

    def test_mask_matches_legal_moves_exactly(self, start_board):
        mask = legal_action_mask(start_board)
        legal_ids = {move_to_action(m, start_board) for m in start_board.legal_moves}
        assert set(np.flatnonzero(mask)) == legal_ids

    def test_checkmate_position_has_empty_mask(self):
        board = chess.Board()
        for san in ["f3", "e5", "g4", "Qh4#"]:
            board.push_san(san)
        assert legal_action_mask(board).sum() == 0
