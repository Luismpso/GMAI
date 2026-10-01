"""Measure playing strength with matches against Stockfish at fixed Elo levels.

Stockfish can play below full strength (``UCI_LimitStrength``) at a chosen
``UCI_Elo``. Playing games at a few levels and fitting one rating to all the
results gives an estimate of ChessNet's strength with a confidence interval::

    python -m chessnet.arena --checkpoint runs/<run>/best.pt \\
        --stockfish C:/tools/stockfish/stockfish.exe --nodes 800

Games start from a fixed set of common openings, each played once with each
colour, so every level faces the same positions and colour advantage cancels
out. The network never saw the first eight plies of games in training, so the
openings also start it where its knowledge starts.

Everything is saved under ``runs/arena-<time>/``: a PGN with all the games
and a JSON summary.

The Elo is on Stockfish's own scale, which is anchored to computer rating
lists rather than to Lichess or FIDE, so treat the absolute number as
indicative. Differences between runs of this script (with and without search,
one checkpoint against another) are the reliable part.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from .play import Player

# Eight plies each: the network's training data starts at ply nine.
OPENINGS: list[tuple[str, str]] = [
    ("Ruy Lopez", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6"),
    ("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7"),
    ("Sicilian Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6"),
    ("King's Indian", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6"),
    ("French Winawer", "e4 e6 d4 d5 Nc3 Bb4 e5 c5"),
    ("Slav", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4"),
    ("Caro-Kann", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5"),
    ("English", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5"),
    ("Italian", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6"),
    ("Nimzo-Indian", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O"),
    ("Scandinavian", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6"),
    ("London System", "d4 d5 Bf4 Nf6 e3 c5 c3 Nc6"),
    ("Grunfeld", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5"),
    ("Scotch", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6"),
    ("Pirc", "e4 d6 d4 Nf6 Nc3 g6 Be2 Bg7"),
    ("Dutch Leningrad", "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7"),
]


def opening_board(line: str) -> chess.Board:
    board = chess.Board()
    for san in line.split():
        board.push_san(san)
    return board


# ------------------------------------------------------------------ ratings
def score_to_elo(score: float) -> float:
    """Elo difference implied by an average score, under the logistic model."""
    score = min(max(score, 1e-9), 1 - 1e-9)
    return 400 * math.log10(score / (1 - score))


def fit_rating(
    results: list[tuple[float, float]], lo: int = 0, hi: int = 4000
) -> tuple[float, float | None, float | None]:
    """Maximum-likelihood rating from (opponent Elo, score) pairs.

    Draws count as half a win. Returns the rating and a 95% profile-likelihood
    interval; a bound is None when it runs off the grid, which happens when
    every game was won (no upper bound) or lost (no lower bound).
    """
    grid = np.arange(lo, hi + 1, dtype=np.float64)
    opp = np.array([r[0] for r in results], dtype=np.float64)[:, None]
    score = np.array([r[1] for r in results], dtype=np.float64)[:, None]
    expected = 1 / (1 + 10 ** ((opp - grid[None, :]) / 400))
    expected = np.clip(expected, 1e-12, 1 - 1e-12)
    loglik = (score * np.log(expected) + (1 - score) * np.log(1 - expected)).sum(axis=0)
    best = int(loglik.argmax())
    inside = np.flatnonzero(loglik >= loglik[best] - 1.92)  # chi2(1) / 2 at 95%
    low = float(grid[inside[0]]) if inside[0] > 0 else None
    high = float(grid[inside[-1]]) if inside[-1] < len(grid) - 1 else None
    return float(grid[best]), low, high


def describe_rating(rating: float, low: float | None, high: float | None) -> str:
    if low is None and high is None:
        return "undetermined"
    if high is None:
        return f"above {low:.0f} (won every game: play stronger levels)"
    if low is None:
        return f"below {high:.0f} (lost every game: play weaker levels)"
    return f"{rating:.0f} Elo (95% interval {low:.0f} to {high:.0f})"


# -------------------------------------------------------------------- games
def play_game(
    player: Player,
    engine: chess.engine.SimpleEngine,
    opening: tuple[str, str],
    chessnet_white: bool,
    nodes: int,
    movetime: float | None,
    sf_time: float,
    max_plies: int,
) -> tuple[float, chess.Board, str, list[float]]:
    """One game from the given opening. Returns ChessNet's score (1, 0.5, 0),
    the final board, how the game ended and the search speeds measured."""
    board = opening_board(opening[1])
    speeds: list[float] = []
    game_marker = object()  # makes python-chess send ucinewgame once per game
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == chessnet_white:
            move, _, _ = player.select(board, nodes=nodes, movetime=movetime)
            if player.last_search is not None:
                speeds.append(player.last_search.nps)
        else:
            move = engine.play(
                board, chess.engine.Limit(time=sf_time), game=game_marker
            ).move
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5, board, "move limit", speeds
    ending = outcome.termination.name.lower().replace("_", " ")
    if outcome.winner is None:
        return 0.5, board, ending, speeds
    return (1.0 if outcome.winner == chessnet_white else 0.0), board, ending, speeds


def run_match(
    player: Player,
    stockfish: str,
    levels: list[int],
    games: int,
    nodes: int,
    movetime: float | None,
    sf_time: float,
    max_plies: int,
    out_dir: Path,
) -> list[dict]:
    """Play ``games`` games at each Stockfish level; save PGN and JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path, json_path = out_dir / "games.pgn", out_dir / "results.json"
    if movetime:
        ours = f"ChessNet (search {movetime:g}s/move)"
    elif nodes > 0:
        ours = f"ChessNet (search {nodes} nodes)"
    else:
        ours = "ChessNet (policy only)"

    records: list[dict] = []
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    try:
        if "UCI_Elo" not in engine.options:
            raise SystemExit(
                f"{stockfish} has no UCI_Elo option; use Stockfish 12 or newer"
            )
        option = engine.options["UCI_Elo"]
        sf_name = engine.id.get("name", "Stockfish")
        print(f"{ours} vs {sf_name}, {sf_time:g} s/move | {games} games per level")
        print(f"games: {pgn_path}\n")

        for level in levels:
            elo = min(max(level, option.min), option.max)
            if elo != level:
                print(
                    f"  ({sf_name} plays between {option.min} and {option.max}; using {elo})"
                )
            settings = {"UCI_LimitStrength": True, "UCI_Elo": elo}
            settings.update(
                {k: v for k, v in (("Threads", 1), ("Hash", 16)) if k in engine.options}
            )
            engine.configure(settings)

            total = 0.0
            for i in range(games):
                opening = OPENINGS[(i // 2) % len(OPENINGS)]
                white = i % 2 == 0
                score, board, ending, speeds = play_game(
                    player, engine, opening, white, nodes, movetime, sf_time, max_plies
                )
                total += score

                game = chess.pgn.Game.from_board(board)
                game.headers.update(
                    Event="ChessNet vs Stockfish",
                    Round=str(len(records) + 1),
                    White=ours if white else f"{sf_name} (Elo {elo})",
                    Black=f"{sf_name} (Elo {elo})" if white else ours,
                    Opening=opening[0],
                    Termination=ending,
                )
                if ending == "move limit":
                    game.headers["Result"] = "1/2-1/2"
                with pgn_path.open("a", encoding="utf-8") as f:
                    print(game, file=f, end="\n\n")

                records.append(
                    {
                        "stockfish_elo": elo,
                        "opening": opening[0],
                        "chessnet_white": white,
                        "score": score,
                        "ending": ending,
                        "moves": (board.ply() + 1) // 2,
                        "nodes_per_s": float(np.mean(speeds)) if speeds else None,
                    }
                )
                json_path.write_text(json.dumps(records, indent=2))

                speed = f" | {np.mean(speeds):,.0f} nodes/s" if speeds else ""
                verdict = {1.0: "won", 0.5: "draw", 0.0: "lost"}[score]
                print(
                    f"  SF {elo} | game {i + 1:>2}/{games} | {opening[0]:<23} | "
                    f"{'white' if white else 'black'} | {verdict:<4} ({ending}, "
                    f"{(board.ply() + 1) // 2} moves) | {total:g}/{i + 1}{speed}",
                    flush=True,
                )
            print()
    finally:
        engine.quit()
    return records


def summarise(records: list[dict]) -> str:
    lines = ["level   games    W    D    L   score   performance"]
    for elo in sorted({r["stockfish_elo"] for r in records}):
        scores = [r["score"] for r in records if r["stockfish_elo"] == elo]
        n, mean = len(scores), sum(scores) / len(scores)
        wins, draws = scores.count(1.0), scores.count(0.5)
        if 0 < mean < 1:
            perf = f"{elo + score_to_elo(mean):.0f}"
        else:
            perf = "all won" if mean == 1 else "all lost"
        lines.append(
            f"{elo:>5} {n:>7} {wins:>4} {draws:>4} {n - wins - draws:>4} "
            f"{mean:>6.0%}   {perf}"
        )
    rating = fit_rating([(r["stockfish_elo"], r["score"]) for r in records])
    lines.append(f"\nestimated strength: {describe_rating(*rating)}")
    lines.append(
        "(Stockfish's Elo scale: compare runs with each other, not with Lichess)"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--stockfish", default="stockfish", help="path to the Stockfish binary"
    )
    ap.add_argument("--elo", type=int, nargs="+", default=[1500, 1800, 2100, 2400])
    ap.add_argument("--games", type=int, default=12, help="games per level (even)")
    ap.add_argument(
        "--nodes", type=int, default=800, help="search simulations; 0 = policy only"
    )
    ap.add_argument(
        "--movetime", type=float, default=None, help="search seconds per move"
    )
    ap.add_argument(
        "--sf-time", type=float, default=0.5, help="Stockfish seconds per move"
    )
    ap.add_argument(
        "--max-plies", type=int, default=400, help="adjudicate a draw after this"
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    games = args.games + args.games % 2  # every opening with both colours
    player = Player(args.checkpoint, device=args.device, nodes=args.nodes)
    out_dir = Path(args.out) / f"arena-{time.strftime('%Y%m%d-%H%M%S')}"
    records: list[dict] = []
    try:
        records = run_match(
            player,
            args.stockfish,
            args.elo,
            games,
            args.nodes,
            args.movetime,
            args.sf_time,
            args.max_plies,
            out_dir,
        )
    except KeyboardInterrupt:
        print("\ninterrupted")
        results = out_dir / "results.json"
        if results.exists():
            records = json.loads(results.read_text())
    if records:
        print(summarise(records))


if __name__ == "__main__":
    main()
