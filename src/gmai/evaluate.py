"""Endgame evaluation.

Reports **W/D/L separately** plus termination breakdown, against:

* the random defender the agent was trained on;
* the *optimal* defender within the endgame (a 2-ply king that maximises
  survival), which is far harder and closer to the truth;

and computes a random-agent baseline for the same positions, so every number
has something honest to be compared against.

Usage:
    python -m gmai.evaluate --checkpoint runs/<run>/final.pt --games 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess

from .agent import DQNAgent
from .encoding import action_to_move
from .endgames import ENDGAME_ORDER, sample_endgame
from .metrics import RollingStats, classify_episode
from .opponents import Opponent, RandomOpponent


class StubbornKingOpponent(Opponent):
    """Defender that maximises distance from the enemy king and the edge.

    Not tablebase-optimal, but a much stiffer test than random: it runs for
    the centre and grabs any hanging piece, which punishes an agent that has
    only learned to beat a defender that wanders.
    """

    name = "stubborn"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        me = board.turn
        best, best_score = [], -float("inf")
        for move in board.legal_moves:
            captures = board.is_capture(move)
            board.push(move)
            if board.is_stalemate() or board.is_insufficient_material():
                score = 1e6  # a draw is a total win for the defender
            elif board.is_checkmate():
                score = -1e6
            else:
                king = board.king(me)
                enemy = board.king(not me)
                centre = -max(
                    abs(chess.square_file(king) - 3.5),
                    abs(chess.square_rank(king) - 3.5),
                )
                score = 10.0 * centre + 2.0 * chess.square_distance(king, enemy)
                if captures:
                    score += 500.0
            board.pop()
            if score > best_score:
                best_score, best = score, [move]
            elif score == best_score:
                best.append(move)
        return self._rng.choice(best)


def play_endgame(
    policy, opponent: Opponent, kind: str, rng: random.Random
) -> tuple[chess.Board, chess.Color, bool]:
    """Play one endgame. ``policy(board) -> move``. Returns (board, colour, truncated)."""
    position = sample_endgame(kind, rng=rng)
    board, strong, limit = position.board, position.strong_color, position.max_moves
    start_ply = len(board.move_stack)

    truncated = False
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) - start_ply >= 2 * limit:
            truncated = True
            break
        move = policy(board) if board.turn == strong else opponent.select_move(board)
        board.push(move)
    return board, strong, truncated


def evaluate_kind(
    policy, opponent: Opponent, kind: str, games: int, seed: int
) -> RollingStats:
    rng = random.Random(seed)
    stats = RollingStats(window=games)
    for _ in range(games):
        board, strong, truncated = play_endgame(policy, opponent, kind, rng)
        stats.add(classify_episode(board, strong, truncated))
    return stats


def run_report(agent: DQNAgent | None, games: int = 200, seed: int = 0) -> dict:
    """Full report: agent and random baseline, vs. two defenders, per endgame."""
    rng_baseline = random.Random(seed + 7)

    def agent_policy(board: chess.Board) -> chess.Move:
        return action_to_move(agent.act(board, greedy=True), board)

    def random_policy(board: chess.Board) -> chess.Move:
        return rng_baseline.choice(list(board.legal_moves))

    report: dict = {}
    for kind in ENDGAME_ORDER:
        report[kind] = {}
        for opponent in (RandomOpponent(seed=seed), StubbornKingOpponent(seed=seed)):
            entry = {
                "random_baseline": evaluate_kind(
                    random_policy, opponent, kind, games, seed
                ).as_dict()
            }
            if agent is not None:
                entry["agent"] = evaluate_kind(
                    agent_policy, opponent, kind, games, seed
                ).as_dict()
            report[kind][opponent.name] = entry
    return report


def print_report(report: dict) -> None:
    print(f"\n{'endgame':<8} {'defender':<10} {'who':<10} "
          f"{'W':>4} {'D':>4} {'L':>4} {'win-rate':>9} {'plies':>7}")
    print("-" * 62)
    for kind, defenders in report.items():
        for defender, entries in defenders.items():
            for who, r in entries.items():
                print(
                    f"{kind:<8} {defender:<10} {who:<10} "
                    f"{r['wins']:>4} {r['draws']:>4} {r['losses']:>4} "
                    f"{r['win_rate']:>9.3f} {r['mean_plies']:>7.1f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GMAI on endgames")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the report as JSON")
    args = parser.parse_args()

    agent = None
    if args.checkpoint:
        agent = DQNAgent.from_checkpoint(args.checkpoint)
        agent.epsilon = 0.0

    report = run_report(agent, games=args.games, seed=args.seed)
    print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
