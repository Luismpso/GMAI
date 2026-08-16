"""Forced-mate endgame position generator.

Scope decision: GMAI is trained on **forced-mate endgames**, not full chess.
This is the regime where a search-free DQN actually works — the horizon is
short (<20 moves), the reward is reachable by exploration, and the win-rate
against a random defender has real dynamic range.

Supported: KQ vs K, KR vs K, KRR vs K.

Every sampled position is guaranteed to be:
  * legal (``board.is_valid()``): kings not adjacent, side-not-to-move not in check
  * non-terminal: the side to move has at least one legal move
  * a theoretical win for the strong side
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess

# Move budget per level: generous enough for a correct mate, tight enough
# that aimless shuffling is punished. KQ vs K mates in <=10 from any position
# with optimal play; KR vs K in <=16.
ENDGAME_SPECS: dict[str, tuple[list[int], int]] = {
    # name      strong-side pieces (besides the king)      max_moves
    "KQvK": ([chess.QUEEN], 30),
    "KRvK": ([chess.ROOK], 40),
    "KRRvK": ([chess.ROOK, chess.ROOK], 40),
}
ENDGAME_ORDER = ("KQvK", "KRvK", "KRRvK")


@dataclass(frozen=True)
class EndgamePosition:
    board: chess.Board
    kind: str
    strong_color: chess.Color
    max_moves: int


def _random_squares(rng: random.Random, k: int) -> list[int]:
    return rng.sample(range(64), k)


def sample_endgame(
    kind: str = "KQvK",
    rng: random.Random | None = None,
    strong_color: chess.Color | None = None,
    strong_to_move: bool | None = None,
    max_attempts: int = 1000,
) -> EndgamePosition:
    """Sample a random legal, non-terminal position of the given endgame type.

    Parameters
    ----------
    kind:
        One of ``KQvK``, ``KRvK``, ``KRRvK``.
    strong_color:
        Colour holding the extra material. Random if ``None``.
    strong_to_move:
        Whether the strong side moves first. Random if ``None``.
    """
    if kind not in ENDGAME_SPECS:
        raise ValueError(f"unknown endgame {kind!r}; expected one of {ENDGAME_ORDER}")

    rng = rng or random.Random()
    pieces, max_moves = ENDGAME_SPECS[kind]

    for _ in range(max_attempts):
        strong = (
            strong_color
            if strong_color is not None
            else rng.choice([chess.WHITE, chess.BLACK])
        )
        weak = not strong
        to_move = (
            strong_to_move if strong_to_move is not None else rng.choice([True, False])
        )

        squares = _random_squares(rng, 2 + len(pieces))
        board = chess.Board(None)  # empty board
        board.set_piece_at(squares[0], chess.Piece(chess.KING, strong))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, weak))
        for piece_type, square in zip(pieces, squares[2:]):
            board.set_piece_at(square, chess.Piece(piece_type, strong))

        board.turn = strong if to_move else weak
        board.castling_rights = chess.BB_EMPTY
        board.halfmove_clock = 0
        board.fullmove_number = 1

        # is_valid() rejects adjacent kings and a side-not-to-move left in check.
        if not board.is_valid():
            continue
        # Reject positions that are already over (mate/stalemate) or where the
        # weak king can simply grab the piece for an instant draw.
        if board.is_game_over(claim_draw=True):
            continue
        if board.turn == weak and _weak_king_can_capture_everything(board, weak):
            continue

        return EndgamePosition(board, kind, strong, max_moves)

    raise RuntimeError(f"could not sample a valid {kind} position in {max_attempts} tries")


def _weak_king_can_capture_everything(board: chess.Board, weak: chess.Color) -> bool:
    """True if the lone king can capture the strong side's last piece for free."""
    remaining = [
        sq for sq, p in board.piece_map().items()
        if p.color != weak and p.piece_type != chess.KING
    ]
    if len(remaining) != 1:
        return False
    target = remaining[0]
    return any(
        m.to_square == target for m in board.legal_moves
    )


def make_sampler(kind: str, seed: int | None = None, **kwargs):
    """Return a zero-argument callable producing fresh :class:`EndgamePosition`."""
    rng = random.Random(seed)

    def sampler() -> EndgamePosition:
        return sample_endgame(kind, rng=rng, **kwargs)

    return sampler


def main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Sample endgame positions")
    parser.add_argument("--kind", default="KQvK", choices=list(ENDGAME_SPECS))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fen-only", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for i in range(args.n):
        pos = sample_endgame(args.kind, rng=rng)
        if args.fen_only:
            print(pos.board.fen())
        else:
            side = "White" if pos.strong_color == chess.WHITE else "Black"
            print(f"--- {args.kind} #{i + 1} | strong side: {side} | {pos.board.fen()}")
            print(pos.board.unicode(borders=True))


if __name__ == "__main__":  # pragma: no cover
    main()
