"""Regression tests for the five bugs found in the first version.

Each test here fails against the original code and passes against the fix.
These are the most valuable tests in the repo: they encode *why* 2100
episodes of training produced nothing measurable.
"""

import chess
import numpy as np
import pytest
import torch

from gmai.agent import DQNAgent
from gmai.encoding import N_ACTIONS, legal_action_mask, move_to_action
from gmai.endgames import make_sampler
from gmai.environment import ChessEnv
from gmai.model import DuelingChessNet
from gmai.opponents import RandomOpponent
from gmai.replay_buffer import ReplayBuffer
from gmai.rewards import endgame_potential, material_potential, shaping_reward


# ----------------------------------------------- dueling baseline
class TestDuelingMaskedAdvantage:
    """The advantage baseline must average over LEGAL actions only."""

    @pytest.fixture
    def net(self):
        torch.manual_seed(0)
        return DuelingChessNet(channels=16, n_blocks=2, hidden=64).eval()

    def test_baseline_is_zero_over_legal_actions(self, net):
        x = torch.rand(3, 18, 8, 8)
        mask = torch.zeros(3, N_ACTIONS, dtype=torch.bool)
        mask[:, :25] = True

        q = net(x, mask)
        v = net.value_head(net.trunk(x))
        centred = ((q - v) * mask).sum(dim=1) / mask.sum(dim=1)
        assert torch.allclose(centred, torch.zeros_like(centred), atol=1e-4)

    def test_illegal_outputs_do_not_shift_legal_q_values(self, net):
        """Averaging over all 4096 outputs let ~4000 untrained values move every Q."""
        x = torch.rand(1, 18, 8, 8)
        mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
        mask[0, :20] = True
        q_before = net(x, mask)[0, :20].clone()

        # Perturb only illegal outputs; masked Q-values must be unchanged.
        with torch.no_grad():
            net.advantage_head[-1].bias[100:] += 50.0
        q_after = net(x, mask)[0, :20]
        assert torch.allclose(q_before, q_after, atol=1e-4)

    def test_mask_with_a_single_legal_action_is_safe(self, net):
        mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
        mask[0, 7] = True
        q = net(torch.rand(1, 18, 8, 8), mask)
        assert torch.isfinite(q).all()


# ------------------------------------------------- normalisation
class TestNoBatchNorm:
    """BatchNorm's running stats are wrong for DQN; GroupNorm replaces it."""

    def test_no_batchnorm_modules(self):
        net = DuelingChessNet(channels=32, n_blocks=3, hidden=64)
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                       for m in net.modules())
        assert any(isinstance(m, torch.nn.GroupNorm) for m in net.modules())

    def test_train_and_eval_modes_agree(self):
        """With BatchNorm, acting (batch of 1) used different stats than learning."""
        torch.manual_seed(0)
        net = DuelingChessNet(channels=16, n_blocks=2, hidden=64)
        x = torch.rand(1, 18, 8, 8)
        mask = torch.ones(1, N_ACTIONS, dtype=torch.bool)

        net.train()
        q_train = net(x, mask)
        net.eval()
        q_eval = net(x, mask)
        assert torch.allclose(q_train, q_eval, atol=1e-5)

    def test_group_count_divides_odd_channel_counts(self):
        for channels in (8, 12, 16, 48, 64):
            net = DuelingChessNet(channels=channels, n_blocks=2, hidden=32)
            assert net(torch.rand(2, 18, 8, 8)).shape == (2, N_ACTIONS)


# ------------------------------------ truncation vs. termination
class TestTruncationIsNotTerminal:
    """A time-limit cut-off must still bootstrap the successor's value."""

    def _agent(self):
        torch.manual_seed(0)
        return DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0,
                        gamma=0.9)

    def _batch(self, terminated: float, reward: float = 0.0):
        rng = np.random.default_rng(0)
        buf = ReplayBuffer(capacity=8, seed=0)
        state = rng.random((18, 8, 8), dtype=np.float32)
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[:12] = True
        for _ in range(8):
            buf.push(state, mask, 3, reward, state, mask, terminated)
        return buf.sample(8)

    def test_terminated_flag_zeroes_the_bootstrap(self):
        agent = self._agent()
        _, td_terminal = agent.train_step(self._batch(terminated=1.0))
        agent2 = self._agent()
        _, td_bootstrap = agent2.train_step(self._batch(terminated=0.0))
        # Different targets => different TD errors. If truncation were treated
        # as terminal these would be identical.
        assert not np.allclose(td_terminal, td_bootstrap)

    def test_env_reports_truncation_separately_from_termination(self):
        env = ChessEnv(
            opponent=RandomOpponent(seed=0),
            position_sampler=make_sampler("KQvK", seed=0),
            use_shaping=False,
        )
        _, info = env.reset()
        saw_step = False
        for _ in range(200):
            legal = np.flatnonzero(info["action_mask"])
            if len(legal) == 0:
                break
            _, _, terminated, truncated, info = env.step(int(legal[0]))
            saw_step = True
            assert not (terminated and truncated)  # never both
            if terminated or truncated:
                break
        assert saw_step

    def test_buffer_stores_terminated_not_done(self):
        batch = self._batch(terminated=0.0)
        assert hasattr(batch, "terminated")
        assert not hasattr(batch, "dones")
        assert batch.masks.shape == (8, N_ACTIONS)


# --------------------------------------------- reward shaping
class TestShapingIsPolicyInvariant:
    """Phi(terminal) = 0 is required for the Ng et al. theorem to apply."""

    def _play(self, ucis, fen=None):
        board = chess.Board(fen) if fen else chess.Board()
        boards = [board.copy()]
        for uci in ucis:
            board.push(chess.Move.from_uci(uci))
            boards.append(board.copy())
        return boards

    def test_discounted_shaping_telescopes_to_minus_phi_of_start(self):
        gamma = 0.9
        boards = self._play(
            ["a2a7", "e8f8", "h6g6", "f8g8", "a7b7", "g8h8", "b7b8"],
            fen="4k3/8/7K/8/8/8/Q7/8 w - - 0 1",
        )
        assert boards[-1].is_checkmate()

        T = len(boards) - 1
        total = sum(
            (gamma**t)
            * shaping_reward(
                boards[t], boards[t + 1], chess.WHITE, gamma,
                potential_fn=endgame_potential,
                after_is_terminal=(t == T - 1),
            )
            for t in range(T)
        )
        assert total == pytest.approx(
            -endgame_potential(boards[0], chess.WHITE), abs=1e-9
        )

    def test_skipping_the_terminal_term_breaks_the_identity(self):
        """Dropping the terminal term is the easy mistake — kept as a witness."""
        gamma = 0.9
        boards = self._play(
            ["a2a7", "e8f8", "h6g6", "f8g8", "a7b7", "g8h8", "b7b8"],
            fen="4k3/8/7K/8/8/8/Q7/8 w - - 0 1",
        )
        T = len(boards) - 1
        broken = sum(
            (gamma**t)
            * shaping_reward(boards[t], boards[t + 1], chess.WHITE, gamma,
                             potential_fn=endgame_potential)
            for t in range(T - 1)  # last term dropped, as before
        )
        assert broken != pytest.approx(
            -endgame_potential(boards[0], chess.WHITE), abs=1e-6
        )

    def test_truncation_keeps_the_successor_potential(self):
        """Truncating is not terminating: Phi(s') must still count."""
        # A position with a real material imbalance, so Phi(s') != 0.
        before = chess.Board("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1")
        after = chess.Board("4k3/8/8/8/8/Q7/8/4K3 b - - 1 1")

        truncated_term = shaping_reward(
            before, after, chess.WHITE, 0.99,
            potential_fn=material_potential, after_is_terminal=False,
        )
        terminal_term = shaping_reward(
            before, after, chess.WHITE, 0.99,
            potential_fn=material_potential, after_is_terminal=True,
        )
        assert material_potential(after, chess.WHITE) > 0
        assert truncated_term != terminal_term
        # Forcing Phi(s') = 0 removes exactly the discounted successor term.
        assert truncated_term - terminal_term == pytest.approx(
            0.99 * material_potential(after, chess.WHITE)
        )


# ------------------------------------------ endgame potential
class TestEndgamePotential:
    def test_losing_the_queen_drops_the_potential(self):
        with_queen = chess.Board("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1")
        without = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        assert endgame_potential(with_queen, chess.WHITE) > endgame_potential(
            without, chess.WHITE
        )

    def test_enemy_king_on_the_rim_scores_higher(self):
        centre = chess.Board("8/8/8/3k4/8/8/Q7/4K3 w - - 0 1")
        corner = chess.Board("k7/8/8/8/8/8/Q7/4K3 w - - 0 1")
        assert endgame_potential(corner, chess.WHITE) > endgame_potential(
            centre, chess.WHITE
        )

    def test_closer_own_king_scores_higher(self):
        far = chess.Board("k7/8/8/8/8/8/Q7/7K w - - 0 1")
        near = chess.Board("k7/8/1K6/8/8/8/Q7/8 w - - 0 1")
        assert endgame_potential(near, chess.WHITE) > endgame_potential(
            far, chess.WHITE
        )
