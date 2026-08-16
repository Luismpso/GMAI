import chess
import numpy as np
import pytest
import torch

from gmai.encoding import N_ACTIONS, N_PLANES, encode_board, legal_action_mask
from gmai.model import DuelingChessNet, masked_q_values
from gmai.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, SumTree


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return DuelingChessNet(channels=16, n_blocks=2, hidden=64).eval()


class TestModel:
    def test_output_shape(self, net):
        x = torch.rand(4, N_PLANES, 8, 8)
        assert net(x).shape == (4, N_ACTIONS)

    def test_batch_of_one_in_eval_mode(self, net):
        x = torch.rand(1, N_PLANES, 8, 8)
        assert net(x).shape == (1, N_ACTIONS)  # BatchNorm safe in eval

    def test_dueling_advantage_is_mean_centered_over_legal_actions(self, net):
        """Q - V must average ~0 over the LEGAL actions."""
        x = torch.rand(2, N_PLANES, 8, 8)
        mask = torch.zeros(2, N_ACTIONS, dtype=torch.bool)
        mask[:, :40] = True
        v = net.value_head(net.trunk(x))
        centered = ((net(x, mask) - v) * mask).sum(dim=1) / mask.sum(dim=1)
        assert torch.allclose(centered, torch.zeros_like(centered), atol=1e-4)

    def test_masked_argmax_is_always_legal(self, net):
        board = chess.Board()
        state = torch.from_numpy(encode_board(board)).unsqueeze(0)
        mask = torch.from_numpy(legal_action_mask(board)).unsqueeze(0)
        q = masked_q_values(net(state, mask), mask)
        best = int(q.argmax(dim=1))
        assert mask[0, best]

    def test_masked_values_are_min_float(self, net):
        q = torch.zeros(1, N_ACTIONS)
        mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
        mask[0, 5] = True
        out = masked_q_values(q, mask)
        assert out[0, 5] == 0.0
        assert out[0, 0] == torch.finfo(torch.float32).min


def _dummy_transition(rng, reward=0.0, terminated=0.0):
    """(state, mask, action, reward, next_state, next_mask, terminated)."""
    state = rng.random((N_PLANES, 8, 8), dtype=np.float32)
    next_state = rng.random((N_PLANES, 8, 8), dtype=np.float32)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[rng.integers(0, N_ACTIONS, size=10)] = True
    next_mask = np.zeros(N_ACTIONS, dtype=bool)
    next_mask[rng.integers(0, N_ACTIONS, size=10)] = True
    action = int(np.flatnonzero(mask)[0])
    return state, mask, action, reward, next_state, next_mask, terminated


class TestReplayBuffer:
    def test_push_and_len(self):
        rng = np.random.default_rng(0)
        buf = ReplayBuffer(capacity=10, seed=0)
        for _ in range(5):
            buf.push(*_dummy_transition(rng))
        assert len(buf) == 5

    def test_capacity_wraparound(self):
        rng = np.random.default_rng(0)
        buf = ReplayBuffer(capacity=8, seed=0)
        for _ in range(20):
            buf.push(*_dummy_transition(rng))
        assert len(buf) == 8

    def test_sample_shapes(self):
        rng = np.random.default_rng(1)
        buf = ReplayBuffer(capacity=64, seed=0)
        for _ in range(64):
            buf.push(*_dummy_transition(rng))
        batch = buf.sample(16)
        assert batch.states.shape == (16, N_PLANES, 8, 8)
        assert batch.actions.shape == (16,)
        assert batch.masks.shape == (16, N_ACTIONS)       # current-state mask
        assert batch.next_masks.shape == (16, N_ACTIONS)
        assert batch.terminated.shape == (16,)            # not `dones`
        assert batch.actions.dtype == np.int64


class TestSumTree:
    def test_total_is_sum_of_priorities(self):
        tree = SumTree(capacity=4)
        for p in (1.0, 2.0, 3.0, 4.0):
            tree.add(p)
        assert tree.total == pytest.approx(10.0)

    def test_update_propagates_to_root(self):
        tree = SumTree(capacity=4)
        leaves = [tree.add(1.0) for _ in range(4)]
        tree.update(leaves[0], 5.0)
        assert tree.total == pytest.approx(8.0)

    def test_get_respects_proportions(self):
        tree = SumTree(capacity=2)
        tree.add(1.0)
        tree.add(99.0)
        hits = sum(
            tree.get(v)[1] > 50 for v in np.linspace(0.5, tree.total - 0.5, 100)
        )
        assert hits > 90  # ~99% of mass in the second leaf


class TestPrioritizedReplay:
    def test_sample_returns_weights_and_indices(self):
        rng = np.random.default_rng(2)
        buf = PrioritizedReplayBuffer(capacity=32, seed=0)
        for _ in range(32):
            buf.push(*_dummy_transition(rng))
        batch = buf.sample(8)
        assert batch.weights is not None and batch.weights.shape == (8,)
        assert batch.indices is not None and batch.indices.shape == (8,)
        assert 0 < batch.weights.max() <= 1.0

    def test_update_priorities_changes_tree(self):
        rng = np.random.default_rng(3)
        buf = PrioritizedReplayBuffer(capacity=16, seed=0)
        for _ in range(16):
            buf.push(*_dummy_transition(rng))
        batch = buf.sample(4)
        before = buf._tree.total
        buf.update_priorities(batch.indices, np.full(4, 10.0))
        assert buf._tree.total > before

    def test_beta_anneals_towards_one(self):
        rng = np.random.default_rng(4)
        buf = PrioritizedReplayBuffer(capacity=16, beta=0.4, beta_increment=0.1, seed=0)
        for _ in range(16):
            buf.push(*_dummy_transition(rng))
        for _ in range(3):
            buf.sample(4)
        assert buf.beta == pytest.approx(0.7)
