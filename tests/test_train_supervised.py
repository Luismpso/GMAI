"""Shard splitting and an end-to-end training smoke on synthetic shards."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_lichess import encode_position  # noqa: E402
from train_supervised import TrainConfig, _split_shards, run_training  # noqa: E402


def _random_boards(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards, board = [], chess.Board()
    while len(boards) < n:
        if board.is_game_over():
            board = chess.Board()
            continue
        boards.append(board.copy())
        board.push(rng.choice(list(board.legal_moves)))
    return boards


def _write_shard(path: Path, boards: list[chess.Board], seed: int = 0) -> None:
    rng = random.Random(seed)
    bb, mt, ac, rs = [], [], [], []
    for board in boards:
        bitboards, meta = encode_position(board)
        move = rng.choice(list(board.legal_moves))
        bb.append(bitboards)
        mt.append(meta)
        ac.append(move.from_square * 64 + move.to_square)
        rs.append(rng.choice([-1, 0, 1]))
    np.savez(
        path,
        boards=np.asarray(bb, dtype=np.uint64),
        metas=np.asarray(mt, dtype=np.uint8),
        actions=np.asarray(ac, dtype=np.int16),
        results=np.asarray(rs, dtype=np.int8),
    )


def test_split_shards_val_zero_trains_on_everything(tmp_path):
    for i in range(3):
        _write_shard(tmp_path / f"shard_{i:04d}.npz", _random_boards(4, seed=i))
    train, val = _split_shards(TrainConfig(data=tmp_path, val_shards=0))
    assert len(train) == 3 and val == []  # regression: -0 slice must not empty train
    train, val = _split_shards(TrainConfig(data=tmp_path, val_shards=1))
    assert len(train) == 2 and len(val) == 1


def test_run_training_smoke(tmp_path):
    _write_shard(tmp_path / "shard_0000.npz", _random_boards(32, seed=1), seed=1)
    _write_shard(tmp_path / "shard_0001.npz", _random_boards(32, seed=2), seed=2)
    out = tmp_path / "final.pt"
    result = run_training(
        TrainConfig(
            data=tmp_path,
            out=out,
            epochs=1,
            batch_size=8,
            channels=16,
            n_blocks=2,
            hidden=64,
            val_shards=1,
            max_steps=3,
            log_every=1,
            device="cpu",
        )
    )
    assert out.exists()
    train_loss = result["history"][0]["train"]["loss"]
    assert train_loss > 0 and math.isfinite(train_loss)


def test_resume_continues_from_saved_step(tmp_path):
    """A killed 3-day run must pick up where it left off, not restart."""
    _write_shard(tmp_path / "shard_0000.npz", _random_boards(32, seed=1), seed=1)
    _write_shard(tmp_path / "shard_0001.npz", _random_boards(32, seed=2), seed=2)
    common = dict(
        data=tmp_path,
        out=tmp_path / "final.pt",
        batch_size=8,
        channels=16,
        n_blocks=2,
        hidden=64,
        val_shards=1,
        ckpt_every=2,
        device="cpu",
    )
    run_training(TrainConfig(max_steps=4, **common))
    assert (tmp_path / "last.pt").exists()  # full state saved for resume

    resumed = run_training(
        TrainConfig(total_steps=6, resume=tmp_path / "last.pt", **common)
    )
    assert resumed["steps"] == 6  # continued from 4, did not restart at 0
