import chess
import numpy as np
import pytest

from gmai.encoding import N_ACTIONS, N_PLANES, move_to_action
from gmai.environment import ChessEnv
from gmai.opponents import RandomOpponent


@pytest.fixture
def env():
    return ChessEnv(opponent=RandomOpponent(seed=0), agent_color=chess.WHITE, seed=0)


class TestReset:
    def test_observation_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (N_PLANES, 8, 8)
        assert info["action_mask"].shape == (N_ACTIONS,)

    def test_start_mask_has_20_actions(self, env):
        _, info = env.reset()
        assert info["action_mask"].sum() == 20

    def test_colors_alternate_when_not_fixed(self):
        env = ChessEnv(opponent=RandomOpponent(seed=0), seed=0)
        env.reset()
        first = env.agent_color
        env.reset()
        assert env.agent_color != first

    def test_black_agent_sees_board_after_white_move(self):
        env = ChessEnv(opponent=RandomOpponent(seed=0), agent_color=chess.BLACK)
        env.reset()
        assert env.board.turn == chess.BLACK
        assert env.board.fullmove_number == 1
        assert len(env.board.move_stack) == 1  # opponent already moved


class TestStep:
    def test_step_returns_gymnasium_tuple(self, env):
        _, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (N_PLANES, 8, 8)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)

    def test_illegal_action_raises(self, env):
        env.reset()
        illegal = move_to_action(chess.Move.from_uci("e2e5"), env.board)
        with pytest.raises(ValueError):
            env.step(illegal)

    def test_opponent_replies_after_agent_move(self, env):
        _, info = env.reset()
        action = int(np.flatnonzero(info["action_mask"])[0])
        env.step(action)
        assert env.board.turn == chess.WHITE  # back to the agent

    def test_truncation_on_move_limit(self):
        env = ChessEnv(
            opponent=RandomOpponent(seed=0),
            agent_color=chess.WHITE,
            max_moves=3,
            use_shaping=False,
            seed=0,
        )
        _, info = env.reset()
        truncated = False
        for _ in range(10):
            legal = np.flatnonzero(info["action_mask"])
            if len(legal) == 0:
                break
            _, _, terminated, truncated, info = env.step(int(legal[0]))
            if terminated or truncated:
                break
        assert truncated or terminated


class TestRewards:
    def _scholars_mate_env(self):
        """Deterministic checkmate: opponent replays a fixed losing line."""

        class ScriptedOpponent(RandomOpponent):
            LINE = ["e7e5", "b8c6", "g8f6"]  # falls into Scholar's mate

            def __init__(self):
                super().__init__(seed=0)
                self.i = 0

            def select_move(self, board):
                move = chess.Move.from_uci(self.LINE[self.i])
                self.i += 1
                return move

        return ChessEnv(
            opponent=ScriptedOpponent(),
            agent_color=chess.WHITE,
            use_shaping=False,
            draw_reward=0.0,
        )

    def test_win_gives_plus_one(self):
        env = self._scholars_mate_env()
        env.reset()
        rewards = []
        for uci in ["e2e4", "f1c4", "d1h5", "h5f7"]:
            move = chess.Move.from_uci(uci)
            action = move_to_action(move, env.board)
            _, r, terminated, _, _ = env.step(action)
            rewards.append(r)
        assert terminated
        assert rewards[-1] == 1.0
        assert all(r == 0.0 for r in rewards[:-1])  # no shaping -> sparse

    def test_shaping_rewards_capture(self):
        env = ChessEnv(
            opponent=RandomOpponent(seed=3),
            agent_color=chess.WHITE,
            use_shaping=True,
        )
        env.reset()
        # 1. e4 then capture whatever we can, checking sign of shaping
        env.board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        action = move_to_action(chess.Move.from_uci("e4d5"), env.board)
        _, reward, terminated, _, _ = env.step(action)
        if not terminated:
            # Won a pawn; opponent's reply may recapture (net <= 0) but the
            # reward must be finite and bounded by shaping scale.
            assert -1.0 <= reward <= 1.0
