"""Exact endgame solver by retrograde analysis.

Instead of downloading gigabytes of Syzygy tablebases, GMAI *computes* the
ground truth for its own scope. A three-piece endgame (K + piece vs. K) has
only 64x64x64x2 = 524 288 encodable states, of which ~448 000 are legal, so
full backward induction runs in seconds and fits in memory.

The result is a **distance-to-mate (DTM)** table: for every position, the
exact number of plies to mate under optimal play by both sides, or "draw".

This unlocks two things that a self-play-only project cannot have:

1. **Supervised warm start** — (position, optimal move) pairs for free, to
   pre-train the advantage head before RL begins. This is what AlphaGo did
   with human games; here the teacher is provably optimal.
2. **An interpretable evaluation metric** — the fraction of moves that
   *worsen* the DTM. Elo tells you a model is bad; "17% of its moves throw
   away progress towards mate" tells you how.

Supported: ``KQvK``, ``KRvK``. ``KRRvK`` has four pieces (~29M states) and is
out of scope for this solver — it falls back to self-play only.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

DRAW = np.iinfo(np.int16).max
UNKNOWN = -1

SOLVABLE = ("KQvK", "KRvK")
PIECE_OF = {"KQvK": chess.QUEEN, "KRvK": chess.ROOK}

# Ray directions as (file_delta, rank_delta).
ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def index(wk: int, bk: int, pc: int, turn: int) -> int:
    """turn: 0 = strong side to move, 1 = weak side to move."""
    return ((wk * 64 + bk) * 64 + pc) * 2 + turn


def unindex(idx: int) -> tuple[int, int, int, int]:
    turn = idx & 1
    idx >>= 1
    pc = idx & 63
    idx >>= 6
    bk = idx & 63
    wk = idx >> 6
    return wk, bk, pc, turn


N_STATES = 64 * 64 * 64 * 2


class EndgameTable:
    """Exact DTM table for a three-piece endgame.

    ``dtm[i]`` is the number of plies to mate with optimal play from both
    sides, or :data:`DRAW`. Positions are indexed by :func:`index` with the
    strong side normalised to White.
    """

    def __init__(self, kind: str, dtm: np.ndarray):
        self.kind = kind
        self.dtm = dtm

    # ------------------------------------------------------------- lookup
    @staticmethod
    def _decompose(
        board: chess.Board, strong: chess.Color
    ) -> tuple[int, int, int, int] | None:
        """Map a board to (wk, bk, piece, turn) with the strong side as 'white'."""
        pieces = board.piece_map()
        if len(pieces) != 3:
            return None
        wk = bk = pc = None
        for square, piece in pieces.items():
            if piece.piece_type == chess.KING:
                if piece.color == strong:
                    wk = square
                else:
                    bk = square
            elif piece.color == strong:
                pc = square
            else:
                return None  # weak side has material: not this table
        if wk is None or bk is None or pc is None:
            return None
        turn = 0 if board.turn == strong else 1
        if strong == chess.BLACK:  # mirror so the strong side is always "White"
            wk, bk, pc = (
                chess.square_mirror(wk),
                chess.square_mirror(bk),
                chess.square_mirror(pc),
            )
        return wk, bk, pc, turn

    def probe(self, board: chess.Board, strong: chess.Color) -> int:
        """DTM in plies, or :data:`DRAW`. Returns ``DRAW`` for unknown shapes."""
        parts = self._decompose(board, strong)
        if parts is None:
            return DRAW
        return int(self.dtm[index(*parts)])

    def best_moves(self, board: chess.Board, strong: chess.Color) -> list[chess.Move]:
        """All moves that keep optimal play (minimise DTM for the strong side)."""
        if board.turn != strong:
            return []
        best, best_dtm = [], None
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                d = 0
            elif board.is_game_over(claim_draw=True):
                d = DRAW
            else:
                d = self.probe(board, strong)
            board.pop()
            if best_dtm is None or d < best_dtm:
                best_dtm, best = d, [move]
            elif d == best_dtm:
                best.append(move)
        return [] if best_dtm is None or best_dtm >= DRAW else best

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, kind=self.kind, dtm=self.dtm)

    @classmethod
    def load(cls, path: str | Path) -> "EndgameTable":
        data = np.load(path)
        return cls(str(data["kind"]), data["dtm"])


def _build_state_graph(piece_type: int):
    """Enumerate legal states and their successor edges.

    Move generation is delegated to ``python-chess`` rather than hand-rolled:
    this runs once (~20 s) and is then cached, and it removes an entire class
    of subtle legality bugs (pinned-ray king retreats, king-defended piece
    captures, stalemate detection).

    ``out_degree`` counts *all* legal moves, while ``successors`` only holds
    the edges that stay inside this endgame. A capture of the strong piece
    leaves K vs K — a dead draw — so it inflates ``out_degree`` without ever
    being resolvable. That is exactly the behaviour we want: the weak side is
    only lost when *every* move loses, and an escape into a drawn position
    keeps ``remaining`` above zero forever.
    """
    valid = np.zeros(N_STATES, dtype=bool)
    is_mate = np.zeros(N_STATES, dtype=bool)
    out_degree = np.zeros(N_STATES, dtype=np.int32)
    successors: list[list[int]] = [[] for _ in range(N_STATES)]

    board = chess.Board(None)
    white_king = chess.Piece(chess.KING, chess.WHITE)
    black_king = chess.Piece(chess.KING, chess.BLACK)
    white_piece = chess.Piece(piece_type, chess.WHITE)

    for wk in range(64):
        for bk in range(64):
            if bk == wk or chess.square_distance(wk, bk) <= 1:
                continue  # same square or adjacent kings
            for pc in range(64):
                if pc in (wk, bk):
                    continue
                for turn in (0, 1):
                    board.clear()
                    board.set_piece_at(wk, white_king)
                    board.set_piece_at(bk, black_king)
                    board.set_piece_at(pc, white_piece)
                    board.turn = chess.WHITE if turn == 0 else chess.BLACK
                    if not board.is_valid():
                        continue

                    i = index(wk, bk, pc, turn)
                    valid[i] = True

                    moves = list(board.legal_moves)
                    out_degree[i] = len(moves)
                    if not moves:
                        if board.is_check():
                            is_mate[i] = True   # weak side mated
                        continue                # stalemate: stays DRAW

                    for move in moves:
                        board.push(move)
                        pieces = board.piece_map()
                        if len(pieces) == 3:
                            nwk = board.king(chess.WHITE)
                            nbk = board.king(chess.BLACK)
                            npc = next(
                                sq for sq, p in pieces.items()
                                if p.piece_type != chess.KING
                            )
                            successors[i].append(
                                index(nwk, nbk, npc, 0 if board.turn else 1)
                            )
                        # else: the piece was captured -> K vs K -> draw sink
                        board.pop()

    return valid, successors, out_degree, is_mate


def solve(kind: str, verbose: bool = False) -> EndgameTable:
    """Compute the exact DTM table for ``kind`` by backward induction.

    Standard retrograde analysis over the game graph:

    * a mate is a resolved state at distance 0;
    * a **strong-side** state is won as soon as *one* successor is won;
    * a **weak-side** state is lost only when *every* legal move leads to a
      won state, tracked with a per-state countdown of unresolved moves.

    Anything never resolved is a draw.
    """
    if kind not in SOLVABLE:
        raise ValueError(f"{kind} has more than 3 pieces; solvable: {SOLVABLE}")

    valid, successors, out_degree, is_mate = _build_state_graph(PIECE_OF[kind])

    predecessors: list[list[int]] = [[] for _ in range(N_STATES)]
    for i in range(N_STATES):
        if valid[i]:
            for j in successors[i]:
                predecessors[j].append(i)

    remaining = out_degree.copy()
    dtm = np.full(N_STATES, DRAW, dtype=np.int16)
    resolved = np.zeros(N_STATES, dtype=bool)

    frontier = [int(i) for i in np.flatnonzero(is_mate)]
    dtm[frontier] = 0
    resolved[frontier] = True

    depth = 0
    while frontier:
        depth += 1
        nxt: list[int] = []
        for i in frontier:
            d = int(dtm[i])
            for p in predecessors[i]:
                if resolved[p]:
                    continue
                if p & 1 == 0:
                    # Strong side to move: a single winning move suffices.
                    dtm[p] = d + 1
                    resolved[p] = True
                    nxt.append(p)
                else:
                    # Weak side to move: lost only if every move loses.
                    remaining[p] -= 1
                    if remaining[p] == 0:
                        dtm[p] = d + 1
                        resolved[p] = True
                        nxt.append(p)
        frontier = nxt
        if verbose and frontier:
            print(f"  depth {depth:>3}: {len(frontier):>7} states resolved")

    dtm[~valid] = DRAW
    if verbose:
        n_valid = int(valid.sum())
        won = int((dtm[valid] < DRAW).sum())
        print(
            f"{kind}: {n_valid:,} legal states | {won:,} won ({won / n_valid:.1%}) "
            f"| max DTM {int(dtm[dtm < DRAW].max())} plies"
        )
    return EndgameTable(kind, dtm)


def get_table(kind: str, cache_dir: str | Path = "tablebases", verbose: bool = False):
    """Load the table from ``cache_dir`` or solve and cache it."""
    if kind not in SOLVABLE:
        return None
    path = Path(cache_dir) / f"{kind}.npz"
    if path.exists():
        return EndgameTable.load(path)
    table = solve(kind, verbose=verbose)
    table.save(path)
    return table


def main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Solve an endgame exactly")
    parser.add_argument("--kind", default="KQvK", choices=SOLVABLE)
    parser.add_argument("--cache-dir", default="tablebases")
    args = parser.parse_args()
    get_table(args.kind, cache_dir=args.cache_dir, verbose=True)


if __name__ == "__main__":  # pragma: no cover
    main()
