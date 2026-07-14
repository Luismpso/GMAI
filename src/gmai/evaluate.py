"""Arena evaluation.

Plays N games (colours alternating) against each baseline and reports
W/D/L, score and an Elo *difference* estimate from the logistic model:

    elo_diff = 400 * log10(score / (1 - score))

Usage:
    python -m gmai.evaluate --checkpoint runs/<run>/final.pt --games 100
"""

from __future__ import annotations

import argparse
import math

import chess

from .agent import DQNAgent
from .encoding import action_to_move
from .opponents import GreedyMaterialOpponent, Opponent, RandomOpponent


def play_game(
    agent: DQNAgent,
    opponent: Opponent,
    agent_color: chess.Color,
    max_moves: int = 200,
) -> float:
    """Return 1 / 0.5 / 0 from the agent's point of view."""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > max_moves:
            break
        if board.turn == agent_color:
            move = action_to_move(agent.act(board, greedy=True), board)
        else:
            move = opponent.select_move(board)
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == agent_color else 0.0


def elo_diff(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)  # avoid infinities
    return 400.0 * math.log10(score / (1.0 - score))


def run_arena(agent: DQNAgent, games: int = 100, seed: int = 0) -> dict:
    baselines: list[Opponent] = [
        RandomOpponent(seed=seed),
        GreedyMaterialOpponent(seed=seed),
    ]
    report: dict[str, dict] = {}
    for opponent in baselines:
        wins = draws = losses = 0
        for g in range(games):
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            result = play_game(agent, opponent, color)
            wins += result == 1.0
            draws += result == 0.5
            losses += result == 0.0
        score = (wins + 0.5 * draws) / games
        report[opponent.name] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": round(score, 3),
            "elo_diff": round(elo_diff(score), 1),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GMAI in the arena")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent = DQNAgent()
    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    report = run_arena(agent, games=args.games, seed=args.seed)
    print(f"{'opponent':<10} {'W':>4} {'D':>4} {'L':>4} {'score':>7} {'ΔElo':>8}")
    for name, r in report.items():
        print(
            f"{name:<10} {r['wins']:>4} {r['draws']:>4} {r['losses']:>4} "
            f"{r['score']:>7.3f} {r['elo_diff']:>+8.1f}"
        )


if __name__ == "__main__":
    main()
