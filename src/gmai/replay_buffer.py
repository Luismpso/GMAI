"""Experience replay.

``ReplayBuffer``          — classic uniform replay (Mnih et al. 2015).
``PrioritizedReplayBuffer`` — proportional PER (Schaul et al. 2016) backed
by a SumTree for O(log n) sampling, with importance-sampling weights to
correct the induced bias.

Transitions store the *next-state action mask* so the Double DQN target can
argmax over legal moves only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Batch:
    states: np.ndarray        # (B, 18, 8, 8) float32
    actions: np.ndarray       # (B,)          int64
    rewards: np.ndarray       # (B,)          float32
    next_states: np.ndarray   # (B, 18, 8, 8) float32
    next_masks: np.ndarray    # (B, 4096)     bool
    dones: np.ndarray         # (B,)          float32
    weights: np.ndarray = field(default=None)  # PER IS weights
    indices: np.ndarray = field(default=None)  # PER tree indices


class ReplayBuffer:
    """Uniform circular replay buffer."""

    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = capacity
        self._storage: list[tuple] = []
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, state, action, reward, next_state, next_mask, done) -> None:
        item = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(next_mask, dtype=bool),
            float(done),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(item)
        else:
            self._storage[self._pos] = item
        self._pos = (self._pos + 1) % self.capacity

    def _collate(self, items: list[tuple]) -> Batch:
        s, a, r, ns, nm, d = zip(*items)
        return Batch(
            states=np.stack(s),
            actions=np.asarray(a, dtype=np.int64),
            rewards=np.asarray(r, dtype=np.float32),
            next_states=np.stack(ns),
            next_masks=np.stack(nm),
            dones=np.asarray(d, dtype=np.float32),
        )

    def sample(self, batch_size: int) -> Batch:
        idx = self._rng.integers(0, len(self._storage), size=batch_size)
        return self._collate([self._storage[i] for i in idx])


class SumTree:
    """Binary tree where each parent stores the sum of its children."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.n_entries = 0
        self._write = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        leaf = self._write + self.capacity - 1
        self.update(leaf, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return leaf

    def update(self, leaf: int, priority: float) -> None:
        change = priority - self.tree[leaf]
        self.tree[leaf] = priority
        parent = leaf
        while parent != 0:
            parent = (parent - 1) // 2
            self.tree[parent] += change

    def get(self, value: float) -> tuple[int, float]:
        """Descend the tree; return (leaf_index, priority) for ``value``."""
        idx = 0
        while True:
            left, right = 2 * idx + 1, 2 * idx + 2
            if left >= len(self.tree):
                return idx, float(self.tree[idx])
            idx = left if value <= self.tree[left] else right
            if idx == right:
                value -= self.tree[left]


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER: P(i) ∝ (|delta_i| + eps)^alpha."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-5,
        eps: float = 1e-2,
        seed: int | None = None,
    ):
        super().__init__(capacity, seed=seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self._tree = SumTree(capacity)
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, next_mask, done) -> None:
        super().push(state, action, reward, next_state, next_mask, done)
        self._tree.add(self._max_priority**self.alpha)

    def sample(self, batch_size: int) -> Batch:
        self.beta = min(1.0, self.beta + self.beta_increment)
        segment = self._tree.total / batch_size

        leaves, priorities, items = [], [], []
        for i in range(batch_size):
            value = self._rng.uniform(segment * i, segment * (i + 1))
            leaf, priority = self._tree.get(value)
            data_idx = leaf - (self._tree.capacity - 1)
            data_idx = min(data_idx, len(self._storage) - 1)
            leaves.append(leaf)
            priorities.append(priority)
            items.append(self._storage[data_idx])

        batch = self._collate(items)
        probs = np.asarray(priorities) / max(self._tree.total, 1e-12)
        weights = (len(self._storage) * probs) ** (-self.beta)
        batch.weights = (weights / weights.max()).astype(np.float32)
        batch.indices = np.asarray(leaves, dtype=np.int64)
        return batch

    def update_priorities(self, leaves: np.ndarray, td_errors: np.ndarray) -> None:
        for leaf, delta in zip(leaves, np.abs(td_errors)):
            priority = float((delta + self.eps) ** self.alpha)
            self._tree.update(int(leaf), priority)
            self._max_priority = max(self._max_priority, delta + self.eps)
