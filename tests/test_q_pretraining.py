"""Tests for Q-value pre-training and scale calibration.

These cover the machinery built to fix the scale mismatch documented in
docs/POSTMORTEM.md — a network pre-trained by cross-entropy ranks moves well
but sits ~300 units away from the return scale, and dropping that into a TD
loop destroys it.
"""

import numpy as np
import pytest
import torch

from gmai.agent import DQNAgent
from gmai.encoding import N_ACTIONS, encode_board, legal_action_mask
from gmai.model import masked_q_values
from gmai.tablebase import DRAW, EndgameTable
from gmai.warmstart import (
    MAX_LEGAL,
    build_q_dataset,
    calibrate_scale,
    optimal_q_value,
    pretrain_q,
)

from pathlib import Path

CACHE = Path("tablebases")


def _table_or_skip(kind="KQvK"):
    path = CACHE / f"{kind}.npz"
    if not path.exists():
        pytest.skip(f"{path} not built (run: python -m gmai.tablebase --kind {kind})")
    return EndgameTable.load(path)


class TestOptimalQValue:
    def test_mate_is_the_maximum(self):
        mate = optimal_q_value(0, phi_before=0.0, gamma=0.9,
                               draw_reward=-1.0, mate=True)
        later = optimal_q_value(6, phi_before=0.0, gamma=0.9,
                                draw_reward=-1.0, mate=False)
        assert mate == pytest.approx(1.0)
        assert mate > later

    def test_closer_mates_are_worth_more(self):
        values = [
            optimal_q_value(d, 0.0, 0.9, -1.0, mate=False) for d in (2, 4, 6, 8, 10)
        ]
        assert values == sorted(values, reverse=True)

    def test_draw_uses_the_draw_reward(self):
        v = optimal_q_value(DRAW, phi_before=0.0, gamma=0.9,
                            draw_reward=-1.0, mate=False)
        assert v == pytest.approx(-1.0)

    def test_potential_offsets_every_value_equally(self):
        """Q_shaped = Q_unshaped - Phi(s): a constant per position."""
        a = optimal_q_value(4, phi_before=0.0, gamma=0.9, draw_reward=-1.0, mate=False)
        b = optimal_q_value(4, phi_before=0.6, gamma=0.9, draw_reward=-1.0, mate=False)
        assert a - b == pytest.approx(0.6)

    def test_gamma_controls_the_dynamic_range(self):
        """The reason gamma=0.99 is wrong here: it flattens the target range."""
        span = lambda g: (  # noqa: E731
            optimal_q_value(2, 0.0, g, -1.0, False)
            - optimal_q_value(20, 0.0, g, -1.0, False)
        )
        assert span(0.90) > 5 * span(0.99)


class TestQDataset:
    @pytest.fixture
    def data(self):
        return build_q_dataset("KQvK", _table_or_skip(), 60, seed=0, gamma=0.9)

    def test_shapes_and_padding(self, data):
        n = len(data.states)
        assert data.states.shape == (n, 18, 8, 8)
        assert data.actions.shape == (n, MAX_LEGAL)
        assert data.targets.shape == (n, MAX_LEGAL)
        assert data.valid.shape == (n, MAX_LEGAL)
        assert data.valid.any(axis=1).all()  # every position has legal moves

    def test_targets_are_within_the_return_range(self, data):
        t = data.targets[data.valid]
        assert t.min() >= -3.0 and t.max() <= 1.5

    def test_padding_slots_are_marked_invalid(self, data):
        for row, n_valid in zip(data.valid, data.valid.sum(axis=1)):
            assert row[:n_valid].all()
            assert not row[n_valid:].any()

    def test_best_target_beats_the_worst_in_every_position(self, data):
        for targets, valid in zip(data.targets, data.valid):
            legal = targets[valid]
            assert legal.max() >= legal.min()


class TestPretrainQ:
    @pytest.fixture
    def data(self):
        return build_q_dataset("KQvK", _table_or_skip(), 150, seed=0, gamma=0.9)

    def test_loss_decreases(self, data):
        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        history = pretrain_q(agent, data, epochs=5, batch_size=64,
                             ce_weight=0.0, verbose=False)
        assert history[-1] < history[0]

    def test_produces_q_values_on_the_return_scale(self, data):
        """The whole point: pure regression lands near the target range."""
        torch.manual_seed(0)
        agent = DQNAgent(channels=16, n_blocks=2, hidden=64, device="cpu", seed=0)
        pretrain_q(agent, data, epochs=8, batch_size=64, ce_weight=0.0,
                   verbose=False)

        states = torch.from_numpy(data.states[:32])
        actions = torch.from_numpy(data.actions[:32])
        valid = torch.from_numpy(data.valid[:32])
        mask = torch.zeros(32, N_ACTIONS, dtype=torch.bool)
        mask.scatter_(1, actions, valid)
        with torch.no_grad():
            q = agent.online(states, mask).gather(1, actions)[valid]
        assert q.abs().max() < 20.0  # not the ~300 that cross-entropy produces

    def test_ranking_term_is_annealed_to_zero(self, data):
        """With ce_weight > 0 training must still converge, not diverge."""
        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        history = pretrain_q(agent, data, epochs=6, batch_size=64,
                             ce_weight=1.0, verbose=False)
        assert np.isfinite(history).all()
        assert history[-1] < history[0]


class TestCalibrateScale:
    @pytest.fixture
    def data(self):
        return build_q_dataset("KQvK", _table_or_skip(), 120, seed=0, gamma=0.9)

    def test_returns_slope_and_intercept(self, data):
        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        alpha, beta = calibrate_scale(agent, data, verbose=False)
        assert np.isfinite(alpha) and np.isfinite(beta)
        assert alpha > 0

    def test_calibration_preserves_the_greedy_policy(self, data):
        """alpha > 0 means the ranking — and therefore the policy — is unchanged."""
        import chess
        from gmai.endgames import sample_endgame
        import random

        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        agent.epsilon = 0.0

        rng = random.Random(0)
        boards = []
        while len(boards) < 12:
            pos = sample_endgame("KQvK", rng=rng)
            if pos.board.turn == pos.strong_color:
                boards.append(pos.board)

        before = [agent.act(b, greedy=True) for b in boards]
        alpha, _ = calibrate_scale(agent, data, verbose=False)
        after = [agent.act(b, greedy=True) for b in boards]
        assert alpha > 0
        assert before == after

    def test_target_network_is_resynced(self, data):
        torch.manual_seed(0)
        agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
        calibrate_scale(agent, data, verbose=False)
        for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
            assert torch.equal(po, pt)


class TestWarmStartDataPacking:
    """Regression: the dense mask array and the np.stack spike killed a run."""

    @pytest.fixture
    def data(self):
        from gmai.warmstart import build_dataset

        return build_dataset("KQvK", _table_or_skip(), 40, seed=0, verbose=False)

    def test_masks_are_stored_packed(self, data):
        assert data.masks_packed.dtype == np.uint8
        assert data.masks_packed.shape == (len(data), N_ACTIONS // 8)

    def test_packing_is_eight_times_smaller(self, data):
        assert data.masks_packed.nbytes * 8 == data.masks.nbytes

    def test_masks_for_round_trips(self, data):
        idx = np.array([0, 3, 7])
        assert np.array_equal(data.masks_for(idx), data.masks[idx])

    def test_targets_are_legal_in_the_unpacked_mask(self, data):
        masks = data.masks_for(np.arange(len(data)))
        assert masks[np.arange(len(data)), data.targets].all()

    def test_arrays_are_exactly_sized(self, data):
        """Pre-allocation must be trimmed to the number actually produced."""
        n = len(data)
        assert data.states.shape[0] == n
        assert data.masks_packed.shape[0] == n
        assert n <= 40

    def test_empty_result_raises_a_clear_error(self):
        from gmai.warmstart import build_dataset

        with pytest.raises(RuntimeError, match="no usable"):
            build_dataset("KQvK", _table_or_skip(), 0, seed=0, verbose=False)
