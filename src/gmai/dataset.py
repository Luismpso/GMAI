"""Training dataset over extracted Lichess shards.

``scripts/extract_lichess.py`` stores positions **compactly**: twelve piece
bitboards plus four metadata bytes per position, in *absolute* board
coordinates. The network, however, consumes the 18-plane encoding from
``encoding.py``, which is written from the **mover's point of view** — the
board (and therefore the move) is mirrored vertically whenever Black is to
move.

This module bridges the two. :func:`decode_planes` and :func:`orient_actions`
expand the compact storage into exactly what :func:`gmai.encoding.encode_board`
and :func:`gmai.encoding.move_to_action` would produce, but vectorised over a
whole batch so the dataloader is not the bottleneck. ``test_dataset.py`` pins
this equivalence against the canonical (slow) encoders.

Storage contract (see ``extract_lichess.encode_position``)
----------------------------------------------------------
``boards``  : (N, 12) uint64 — bitboards, order ``[W_P,W_N,W_B,W_R,W_Q,W_K,
              B_P,B_N,B_B,B_R,B_Q,B_K]``. Bit ``i`` is square ``i``.
``metas``   : (N, 4)  uint8  — ``[turn (1=White), castling_bits, ep_square
              (64 if none), halfmove_clock]``. Castling bits: 0=WK,1=WQ,2=BK,3=BQ.
``actions`` : (N,)    int16  — ``from_square * 64 + to_square`` (absolute).
``results`` : (N,)    int8   — game result already from the mover's POV
              (+1 the mover's side won, -1 lost, 0 draw).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .encoding import N_PLANES

_SQUARE_MIRROR = np.uint8(56)  # chess.square_mirror(sq) == sq ^ 56 (flip rank)


def _bits_to_planes(bitboards: np.ndarray) -> np.ndarray:
    """(N, 12) uint64 bitboards -> (N, 12, 8, 8) float32, bit ``i`` at square ``i``."""
    bb = bitboards.astype(np.uint64)
    shifts = np.arange(64, dtype=np.uint64)
    bits = ((bb[:, :, None] >> shifts) & np.uint64(1)).astype(np.float32)
    # square i -> (row=i//8, col=i%8), matching divmod(sq, 8) in encode_board.
    return bits.reshape(bitboards.shape[0], 12, 8, 8)


def decode_planes(boards: np.ndarray, metas: np.ndarray) -> np.ndarray:
    """Expand compact storage into (N, 18, 8, 8) planes from the mover's POV.

    Reproduces :func:`gmai.encoding.encode_board` exactly, vectorised.
    """
    n = boards.shape[0]
    pieces = _bits_to_planes(boards)  # (N, 12, 8, 8): 0-5 White, 6-11 Black
    white, black = pieces[:, 0:6], pieces[:, 6:12]

    turn = metas[:, 0].astype(bool)  # True: White to move
    w, b = turn, ~turn
    planes = np.zeros((n, N_PLANES, 8, 8), dtype=np.float32)

    # Own pieces in planes 0-5, opponent in 6-11. For Black to move, "own" is
    # Black and the board is mirrored vertically (flip the rank axis, -2).
    planes[w, 0:6], planes[w, 6:12] = white[w], black[w]
    planes[b, 0:6] = np.flip(black[b], axis=-2)
    planes[b, 6:12] = np.flip(white[b], axis=-2)

    planes[w, 12] = 1.0  # side-to-move plane: ones iff White to move

    cast = metas[:, 1].astype(np.uint8)
    wk = ((cast >> 0) & 1).astype(np.float32)
    wq = ((cast >> 1) & 1).astype(np.float32)
    bk = ((cast >> 2) & 1).astype(np.float32)
    bq = ((cast >> 3) & 1).astype(np.float32)
    # Planes 13-16: own K-side, own Q-side, opp K-side, opp Q-side.
    planes[w, 13], planes[w, 14] = wk[w][:, None, None], wq[w][:, None, None]
    planes[w, 15], planes[w, 16] = bk[w][:, None, None], bq[w][:, None, None]
    planes[b, 13], planes[b, 14] = bk[b][:, None, None], bq[b][:, None, None]
    planes[b, 15], planes[b, 16] = wk[b][:, None, None], wq[b][:, None, None]

    # En-passant (plane 17) is rare; scatter the few that exist, mirrored for Black.
    ep = metas[:, 2].astype(np.int64)
    for i in np.nonzero(ep != 64)[0]:
        sq = int(ep[i])
        if not turn[i]:
            sq ^= 56
        planes[i, 17, sq // 8, sq % 8] = 1.0

    return planes


def orient_actions(actions: np.ndarray, metas: np.ndarray) -> np.ndarray:
    """Map absolute ``from*64+to`` actions to the mover's POV (mirror for Black).

    Reproduces :func:`gmai.encoding.move_to_action`, vectorised.
    """
    a = actions.astype(np.int64)
    frm, to = a // 64, a % 64
    b = ~metas[:, 0].astype(bool)  # Black to move
    frm = frm.copy()
    to = to.copy()
    frm[b] ^= 56
    to[b] ^= 56
    return frm * 64 + to


def list_shards(root: str | Path) -> list[Path]:
    """Sorted ``shard_*.npz`` files under ``root``."""
    return sorted(Path(root).glob("shard_*.npz"))


class ShardIterableDataset:
    """Streams decoded ``(planes, action, value)`` batches from shard files.

    An :class:`~torch.utils.data.IterableDataset`: shards are visited in a
    (per-epoch shuffled) order, each is loaded once and its rows shuffled in
    memory, then decoded batch by batch. Peak memory is one shard, not the
    whole dataset — which matters because the expanded planes are ~1.1 KB each
    while the compact rows are ~100 bytes.
    """

    def __init__(
        self,
        shards: list[Path],
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 0,
    ):
        import torch  # local import: keeps torch off the module import path

        self._torch = torch
        self.shards = list(shards)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        torch = self._torch
        rng = np.random.default_rng(self.seed + self._epoch)
        order = (
            rng.permutation(len(self.shards)) if self.shuffle else range(len(self.shards))
        )

        for si in order:
            data = np.load(self.shards[si])
            boards, metas = data["boards"], data["metas"]
            actions, results = data["actions"], data["results"]
            n = len(actions)
            rows = rng.permutation(n) if self.shuffle else np.arange(n)

            for start in range(0, n, self.batch_size):
                idx = rows[start : start + self.batch_size]
                planes = decode_planes(boards[idx], metas[idx])
                acts = orient_actions(actions[idx], metas[idx])
                vals = results[idx].astype(np.float32)
                yield (
                    torch.from_numpy(planes),
                    torch.from_numpy(acts).long(),
                    torch.from_numpy(vals),
                )
