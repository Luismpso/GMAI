"""Interrupting and resuming training must change nothing.

The strongest check available: a run stopped mid-epoch and resumed from its
last.pt has to end with bit-for-bit the same weights as a run that never
stopped. Anything not restored exactly (optimizer moments, learning-rate
schedule, batch order, position in the epoch) makes the weights diverge.
"""

import json
import random

import chess
import numpy as np
import pytest
import torch

from chessnet.encoding import board_arrays
from chessnet.model import ChessNet
from chessnet.train import build_parser, train


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("shards")
    rng = random.Random(0)
    boards, metas, actions, results = [], [], [], []
    while len(actions) < 2400:
        board, outcome = chess.Board(), rng.choice([-1, 0, 1])
        for _ in range(60):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            bitboards, meta = board_arrays(board)
            boards.append(bitboards)
            metas.append(meta)
            actions.append(move.from_square * 64 + move.to_square)
            results.append(outcome if board.turn == chess.WHITE else -outcome)
            board.push(move)
    np.savez_compressed(
        path / "shard_0000.npz",
        boards=np.array(boards),
        metas=np.array(metas),
        actions=np.array(actions, dtype=np.int16),
        results=np.array(results, dtype=np.int8),
    )
    return path


def _args(data_dir, out, *extra):
    # 2400 positions, 2% validation, batch 64 -> 37 batches per epoch, so a
    # 60-step run crosses an epoch boundary and step 20 is mid-epoch.
    return build_parser().parse_args(
        [
            "--data", str(data_dir), "--out", str(out), "--channels", "8",
            "--blocks", "1", "--batch-size", "64", "--eval-every", "10",
            "--warmup-steps", "5", "--log-every", "1000", "--device", "cpu",
            "--val-fraction", "0.02", "--epochs", "5", *extra,
        ]
    )  # fmt: skip


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)  # deterministic CPU reductions
    yield
    torch.set_num_threads(previous)


def test_resumed_run_matches_uninterrupted_run(data_dir, tmp_path):
    straight = train(_args(data_dir, tmp_path / "a", "--max-steps", "60"))

    interrupted = train(_args(data_dir, tmp_path / "b", "--max-steps", "20"))
    resumed = train(
        _args(data_dir, tmp_path / "b", "--max-steps", "60",
              "--resume", str(interrupted / "last.pt"))
    )  # fmt: skip
    assert resumed == interrupted, "a resumed run continues in its own directory"

    a, b = _weights(straight / "last.pt"), _weights(resumed / "last.pt")
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} diverged after resuming"

    steps_a = [e["step"] for e in json.loads((straight / "history.json").read_text())]
    steps_b = [e["step"] for e in json.loads((resumed / "history.json").read_text())]
    assert steps_a == steps_b == [10, 20, 30, 40, 50, 60]


def test_init_from_starts_at_the_checkpoint_accuracy(data_dir, tmp_path):
    source = train(_args(data_dir, tmp_path / "src", "--max-steps", "30"))
    expected = torch.load(source / "best.pt", weights_only=False)["metrics"]["top1"]

    fresh = train(
        _args(data_dir, tmp_path / "new", "--max-steps", "1",
              "--init-from", str(source / "best.pt"))
    )  # fmt: skip
    start = torch.load(fresh / "best.pt", weights_only=False)
    assert start["step"] == 0
    assert start["metrics"]["top1"] == expected


def test_resume_rejects_checkpoints_without_optimizer_state(data_dir, tmp_path):
    old = tmp_path / "old.pt"
    ChessNet(channels=8, blocks=1).save(old, step=5)
    with pytest.raises(SystemExit, match="--init-from"):
        train(_args(data_dir, tmp_path / "x", "--resume", str(old)))


def test_max_hours_cannot_reset_loaded_weights(data_dir, tmp_path):
    old = tmp_path / "w.pt"
    ChessNet(channels=8, blocks=1).save(old, step=5)
    with pytest.raises(SystemExit, match="max-hours"):
        train(
            _args(data_dir, tmp_path / "y", "--init-from", str(old), "--max-hours", "1")
        )
