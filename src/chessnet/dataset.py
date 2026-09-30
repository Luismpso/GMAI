"""Shard-backed dataset.

Shards hold bitboards, not planes, so the whole training set fits in memory —
100 million positions is about 1 GB as ``uint64`` bitboards against 460 GB as
float32 planes. Expansion happens per batch on the CPU while the GPU works on
the previous one.

``ShardDataset`` loads every shard it finds, holds an index, and yields batches
of already-expanded planes. Held-out validation comes from a fixed slice, so
the split is identical across runs without storing a separate file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .encoding import N_ACTIONS, bitboards_to_planes


def orient_actions(actions: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """Mirror absolute ``from*64+to`` actions for positions with Black to move.

    Vertical mirroring of a square index is ``sq ^ 56``. Must agree exactly
    with :func:`chessnet.encoding.move_to_action`, which is what inference uses.
    """
    black = turn == 0
    from_sq, to_sq = np.divmod(actions, 64)
    mirrored = (from_sq ^ 56) * 64 + (to_sq ^ 56)
    return np.where(black, mirrored, actions)


class ShardDataset:
    def __init__(
        self,
        data_dir: str | Path,
        val_fraction: float = 0.02,
        seed: int = 0,
        limit: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        shards = sorted(self.data_dir.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"no shards in {self.data_dir}")

        boards, metas, actions, results = [], [], [], []
        total = 0
        for path in shards:
            with np.load(path) as data:
                boards.append(data["boards"])
                metas.append(data["metas"])
                actions.append(data["actions"])
                results.append(data["results"])
            total += len(actions[-1])
            if limit is not None and total >= limit:
                break

        self.boards = np.concatenate(boards)
        self.metas = np.concatenate(metas)
        # Shards store moves in absolute board coordinates. The planes are
        # oriented for the side to move (mirrored when Black is on move), so
        # the labels must be mirrored the same way or half the dataset teaches
        # the wrong move. Done here, at load time, so existing shards stay valid.
        self.actions = orient_actions(
            np.concatenate(actions).astype(np.int64),
            np.concatenate(metas)[:, 0],
        )
        self.results = np.concatenate(results).astype(np.float32)
        if limit is not None:
            self.boards = self.boards[:limit]
            self.metas = self.metas[:limit]
            self.actions = self.actions[:limit]
            self.results = self.results[:limit]

        n = len(self.actions)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        n_val = int(n * val_fraction)
        self.val_idx = order[:n_val]
        self.train_idx = order[n_val:]

        manifest = self.data_dir / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planes = bitboards_to_planes(self.boards[idx], self.metas[idx])
        return (
            torch.from_numpy(planes),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.results[idx]),
        )

    def iter_batches(
        self,
        batch_size: int,
        train: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        idx = self.train_idx if train else self.val_idx
        if shuffle:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(idx)
        for start in range(0, len(idx), batch_size):
            chunk = idx[start : start + batch_size]
            if len(chunk) < 2:  # BatchNorm-free, but keep degenerate batches out
                continue
            yield self.batch(chunk)

    def summary(self) -> str:
        return (
            f"{len(self):,} positions from {self.data_dir} "
            f"({self.n_train:,} train / {self.n_val:,} val)"
        )


def action_stats(dataset: ShardDataset, top: int = 10) -> list[tuple[int, int]]:
    """Most frequent actions — a sanity check on the label distribution."""
    counts = np.bincount(dataset.actions, minlength=N_ACTIONS)
    order = np.argsort(counts)[::-1][:top]
    return [(int(a), int(counts[a])) for a in order]
