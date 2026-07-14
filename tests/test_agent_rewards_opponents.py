import chess
import numpy as np
import pytest
import torch

from gmai.agent import DQNAgent
from gmai.encoding import N_ACTIONS, action_to_move, legal_action_mask
from gmai.opponents import (
    GreedyMaterialOpponent,
    OpponentPool,
    RandomOpponent,
)
from gmai.replay_buffer import ReplayBuffer
from gmai.rewards import (
    material_balance,
    potential,
    shaping_reward,
    terminal_reward,
)


@pytest.fixture(scope="module")
def tiny_agent():
    torch.manual_seed(0)
    return DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)


class TestRewards:
    def test_start_material_is_zero(self):
        board = chess.Board()
        assert material_balance(board, chess.WHITE) == 0.0
        assert potential(board, chess.WHITE) == 0.0

    @pytest.mark.parametrize(
        "fen,white_balance",
        [
            ("k7/8/8/8/8/8/8/KQ6 w - - 0 1", 9.0),    # extra queen
            ("k7/8/8/8/8/8/8/KR6 w - - 0 1", 5.0),    # extra rook
            ("kn6/8/8/8/8/8/8/KB6 w - - 0 1", 0.0),   # bishop vs knight
            ("kq6/8/8/8/8/8/8/KR6 w - - 0 1", -4.0),  # rook vs queen
        ],
    )
    def test_material_balance_values(self, fen, white_balance):
        board = chess.Board(fen)
        assert material_balance(board, chess.WHITE) == white_balance
        assert material_balance(board, chess.BLACK) == -white_balance

    def test_potential_is_antisymmetric(self):
        board = chess.Board("k7/8/8/8/8/8/8/KR6 w - - 0 1")
        assert potential(board, chess.WHITE) == -potential(board, chess.BLACK)

    def test_shaping_telescopes_along_trajectory(self):
        """Sum of shaping terms == gamma^T * phi(s_T) - phi(s_0) (policy invariance)."""
        gamma = 0.9
        board = chess.Board()
        boards = [board.copy()]
        for san in ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qd8"]:
            board.push_san(san)
            boards.append(board.copy())

        total = sum(
            (gamma ** t)
            * shaping_reward(boards[t], boards[t + 1], chess.WHITE, gamma)
            for t in range(len(boards) - 1)
        )
        T = len(boards) - 1
        expected = (gamma ** T) * potential(boards[-1], chess.WHITE) - potential(
            boards[0], chess.WHITE
        )
        assert total == pytest.approx(expected, abs=1e-9)

    def test_terminal_rewards(self):
        board = chess.Board()
        for san in ["f3", "e5", "g4", "Qh4#"]:
            board.push_san(san)
        assert terminal_reward(board, chess.BLACK) == 1.0
        assert terminal_reward(board, chess.WHITE) == -1.0
        assert terminal_reward(chess.Board(), chess.WHITE) == 0.0

    def test_draw_reward_on_stalemate(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
        assert terminal_reward(board, chess.WHITE, draw_reward=-0.1) == -0.1


class TestOpponents:
    def test_random_opponent_plays_legal(self):
        opp = RandomOpponent(seed=0)
        board = chess.Board()
        for _ in range(30):
            if board.is_game_over():
                break
            move = opp.select_move(board)
            assert move in board.legal_moves
            board.push(move)

    def test_greedy_takes_free_queen(self):
        opp = GreedyMaterialOpponent(epsilon=0.0, seed=0)
        board = chess.Board("k7/8/8/3q4/4P3/8/8/K7 w - - 0 1")
        assert opp.select_move(board) == chess.Move.from_uci("e4d5")

    def test_greedy_prefers_mate_over_material(self):
        opp = GreedyMaterialOpponent(epsilon=0.0, seed=0)
        # Qxe5 wins a bishop, but Rd8 is back-rank mate (verified position).
        board = chess.Board("6k1/5ppp/8/4b3/8/6Q1/5PPP/3R2K1 w - - 0 1")
        assert opp.select_move(board) == chess.Move.from_uci("d1d8")

    def test_pool_capacity_and_sampling(self, tiny_agent):
        pool = OpponentPool(capacity=2, seed=0)
        with pytest.raises(RuntimeError):
            pool.sample()
        for _ in range(3):
            pool.add_snapshot(tiny_agent)
        assert len(pool) == 2
        board = chess.Board()
        move = pool.sample().select_move(board)
        assert move in board.legal_moves

    def test_pool_snapshots_are_frozen_copies(self, tiny_agent):
        pool = OpponentPool(capacity=1, seed=0)
        pool.add_snapshot(tiny_agent)
        frozen = pool.sample().agent
        assert frozen is not tiny_agent
        assert frozen.epsilon == 0.0


class TestAgent:
    def test_act_returns_legal_action(self, tiny_agent):
        board = chess.Board()
        mask = legal_action_mask(board)
        for greedy in (False, True):
            action = tiny_agent.act(board, greedy=greedy)
            assert 0 <= action < N_ACTIONS
            assert mask[action]

    def test_greedy_act_is_deterministic(self, tiny_agent):
        board = chess.Board()
        actions = {tiny_agent.act(board, greedy=True) for _ in range(5)}
        assert len(actions) == 1

    def test_act_move_is_pushable(self, tiny_agent):
        board = chess.Board()
        move = action_to_move(tiny_agent.act(board, greedy=True), board)
        board.push(move)  # must not raise

    def test_epsilon_decays_to_floor(self):
        agent = DQNAgent(
            channels=8, n_blocks=2, hidden=32,
            epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=10,
            device="cpu", seed=0,
        )
        for _ in range(50):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.1)

    def _filled_buffer(self, n=32):
        rng = np.random.default_rng(0)
        buf = ReplayBuffer(capacity=n, seed=0)
        board = chess.Board()
        from gmai.encoding import encode_board

        state = encode_board(board)
        for _ in range(n):
            action = int(rng.integers(0, N_ACTIONS))
            buf.push(state, action, float(rng.normal()), state,
                     legal_action_mask(board), 0.0)
        return buf

    def test_train_step_returns_finite_loss(self, tiny_agent):
        batch = self._filled_buffer().sample(8)
        loss, td = tiny_agent.train_step(batch)
        assert np.isfinite(loss)
        assert td.shape == (8,)

    def test_target_sync_copies_weights(self, tiny_agent):
        with torch.no_grad():
            for p in tiny_agent.online.parameters():
                p.add_(1.0)
        tiny_agent.sync_target()
        for po, pt in zip(
            tiny_agent.online.parameters(), tiny_agent.target.parameters()
        ):
            assert torch.equal(po, pt)

    def test_save_and_load_round_trip(self, tiny_agent, tmp_path):
        path = tmp_path / "ckpt.pt"
        tiny_agent.save(path)
        fresh = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=1)
        fresh.load(path)
        for po, pf in zip(
            tiny_agent.online.parameters(), fresh.online.parameters()
        ):
            assert torch.equal(po, pf)
