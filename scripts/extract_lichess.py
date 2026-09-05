"""Extract training positions from a Lichess database archive.

Reads a local file (``.pgn.zst`` or plain ``.pgn``) or streams a monthly
archive from database.lichess.org.

Parsing is the bottleneck, not I/O. One Python process manages roughly 1 600
positions per second, which turns 50 million positions into nine hours. Here a
single process reads the file and splits it on game boundaries while a pool of
workers parses the blocks, which scales close to linearly with cores.

Positions are stored as **piece bitboards** rather than encoded planes. The
shards compress to a few bytes per position instead of 4.6 KB, so tens of
millions fit in a few hundred megabytes; the dataloader expands them into
planes on the fly.

    python extract_lichess.py --file data/raw/lichess_2025-01.pgn.zst \
        --positions 50000000 --min-elo 2000 --workers 10
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn
import numpy as np

BASE_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)

# Time controls worth learning from. Bullet is played on reflex rather than
# judgement, and correspondence games involve outside analysis; neither
# reflects the decision we want the network to imitate.
GOOD_TIME_CONTROLS = frozenset({"rapid", "classical", "blitz"})

PIECE_ORDER = (
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
)
RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

_CONFIG: dict = {}  # set once per worker by the pool initialiser


def open_archive(path: Path | None, month: str | None):
    """Text stream over the PGN, from disk or over the network."""
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".zst":
            import zstandard as zstd

            reader = zstd.ZstdDecompressor().stream_reader(path.open("rb"))
            return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        return path.open("r", encoding="utf-8", errors="replace")

    import zstandard as zstd

    url = BASE_URL.format(month=month)
    print(f"streaming {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "chessnet/0.1"})
    reader = zstd.ZstdDecompressor().stream_reader(
        urllib.request.urlopen(request, timeout=60)
    )
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def iter_blocks(stream, games_per_block: int) -> Iterator[str]:
    """Split the stream into blocks of whole games without parsing them.

    Games start with a line beginning ``[Event``. Detecting that is cheap, which
    keeps the reader from becoming the bottleneck it exists to avoid.
    """
    lines: list[str] = []
    count = 0
    for line in stream:
        if line.startswith("[Event ") and lines:
            count += 1
            if count >= games_per_block:
                yield "".join(lines)
                lines, count = [], 0
        lines.append(line)
    if lines:
        yield "".join(lines)


def init_worker(config: dict) -> None:
    global _CONFIG
    _CONFIG = config


def classify_time_control(tc: str) -> str:
    if not tc or tc == "-":
        return "correspondence"
    try:
        base, inc = tc.split("+")
        total = int(base) + 40 * int(inc)
    except (ValueError, AttributeError):
        return "unknown"
    if total < 179:
        return "bullet"
    if total < 479:
        return "blitz"
    if total < 1499:
        return "rapid"
    return "classical"


def encode_position(board: chess.Board) -> tuple[list[int], list[int]]:
    bitboards = [
        int(board.pieces(pt, colour))
        for colour in (chess.WHITE, chess.BLACK)
        for pt in PIECE_ORDER
    ]
    castling = (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(board.has_queenside_castling_rights(chess.BLACK)) << 3
    )
    meta = [
        int(board.turn),
        castling,
        board.ep_square if board.ep_square is not None else 64,
        min(board.halfmove_clock, 255),
    ]
    return bitboards, meta


def parse_block(block: str):
    """Parse one block of games. Returns arrays plus (games_read, games_kept)."""
    min_elo = _CONFIG["min_elo"]
    max_gap = _CONFIG["max_elo_gap"]
    skip_plies = _CONFIG["skip_opening_plies"]

    boards, metas, actions, results = [], [], [], []
    read = kept = 0

    stream = io.StringIO(block)
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception:
            break
        if game is None:
            break
        read += 1

        headers = game.headers
        try:
            white = int(headers.get("WhiteElo", 0))
            black = int(headers.get("BlackElo", 0))
        except ValueError:
            continue
        if min(white, black) < min_elo or abs(white - black) > max_gap:
            continue
        if classify_time_control(headers.get("TimeControl", "")) not in GOOD_TIME_CONTROLS:
            continue
        result = RESULTS.get(headers.get("Result", "*"))
        if result is None or headers.get("Termination", "") == "Abandoned":
            continue

        board = game.board()
        added = False
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= skip_plies:
                bitboards, meta = encode_position(board)
                boards.append(bitboards)
                metas.append(meta)
                actions.append(move.from_square * 64 + move.to_square)
                results.append(result if board.turn == chess.WHITE else -result)
                added = True
            board.push(move)
        kept += added

    if not actions:
        return (
            np.empty((0, 12), dtype=np.uint64),
            np.empty((0, 4), dtype=np.uint8),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int8),
            read, kept,
        )
    return (
        np.asarray(boards, dtype=np.uint64),
        np.asarray(metas, dtype=np.uint8),
        np.asarray(actions, dtype=np.int16),
        np.asarray(results, dtype=np.int8),
        read, kept,
    )


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.index = 0
        self.total_bytes = 0
        self._buffers: list[tuple] = []
        self._count = 0

    def add(self, arrays: tuple) -> None:
        if len(arrays[2]) == 0:
            return
        self._buffers.append(arrays)
        self._count += len(arrays[2])
        if self._count >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffers:
            return
        path = self.out_dir / f"shard_{self.index:04d}.npz"
        np.savez_compressed(
            path,
            boards=np.concatenate([b[0] for b in self._buffers]),
            metas=np.concatenate([b[1] for b in self._buffers]),
            actions=np.concatenate([b[2] for b in self._buffers]),
            results=np.concatenate([b[3] for b in self._buffers]),
        )
        size = path.stat().st_size
        self.total_bytes += size
        print(f"    {path.name}: {self._count:,} positions, {size / 1e6:.0f} MB",
              flush=True)
        self.index += 1
        self._buffers, self._count = [], 0


def extract(args) -> None:
    config = {
        "min_elo": args.min_elo,
        "max_elo_gap": args.max_elo_gap,
        "skip_opening_plies": args.skip_opening_plies,
    }
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"source  : {args.file or args.month}")
    print(f"target  : {args.positions:,} positions")
    print(f"filter  : both players >= {args.min_elo} Elo, gap <= {args.max_elo_gap}")
    print(f"workers : {workers}")
    print(f"output  : {args.out}\n")

    stream = open_archive(args.file, args.month)
    writer = ShardWriter(args.out, args.shard_size)

    positions = games_read = games_kept = 0
    started = time.time()
    next_report = 500_000

    pool = mp.Pool(workers, initializer=init_worker, initargs=(config,))
    try:
        blocks = iter_blocks(stream, args.games_per_block)
        for *arrays, read, kept in pool.imap_unordered(parse_block, blocks, chunksize=1):
            writer.add(tuple(arrays))
            positions += len(arrays[2])
            games_read += read
            games_kept += kept

            if positions >= next_report:
                elapsed = time.time() - started
                rate = positions / elapsed
                left = (args.positions - positions) / max(rate, 1)
                print(f"  {positions:,}/{args.positions:,} | {rate:,.0f} pos/s "
                      f"| {games_read:,} read, {games_kept / max(games_read, 1):.1%} kept "
                      f"| ~{left / 60:.0f} min left", flush=True)
                next_report += 500_000

            if positions >= args.positions:
                break
    except KeyboardInterrupt:
        print("\ninterrupted; flushing what we have")
    finally:
        pool.terminate()
        pool.join()
        writer.flush()
        stream.close()

    elapsed = time.time() - started
    (args.out / "manifest.json").write_text(json.dumps({
        "source": str(args.file or args.month),
        "min_elo": args.min_elo,
        "max_elo_gap": args.max_elo_gap,
        "skip_opening_plies": args.skip_opening_plies,
        "time_controls": sorted(GOOD_TIME_CONTROLS),
        "games_read": games_read,
        "games_kept": games_kept,
        "positions": positions,
        "shards": writer.index,
        "bytes": writer.total_bytes,
        "seconds": round(elapsed, 1),
    }, indent=2) + "\n")

    print(f"\n{positions:,} positions from {games_kept:,} games in {elapsed / 60:.1f} min")
    print(f"{writer.index} shards, {writer.total_bytes / 1e6:.0f} MB total")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="local .pgn.zst or .pgn")
    source.add_argument("--month", help="stream from lichess, e.g. 2026-07")
    ap.add_argument("--out", type=Path, default=Path("data/train"))
    ap.add_argument("--positions", type=int, default=50_000_000)
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--max-elo-gap", type=int, default=300)
    ap.add_argument("--skip-opening-plies", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=2_000_000)
    ap.add_argument("--games-per-block", type=int, default=500)
    ap.add_argument("--workers", type=int, default=None)
    extract(ap.parse_args())


if __name__ == "__main__":
    mp.freeze_support()  # required on Windows
    main()
