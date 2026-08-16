"""Play against a trained agent in the terminal.

Usage:
    python -m gmai.play --checkpoint runs/<run>/final.pt --color white
Moves are entered in SAN (e4, Nf3, O-O) or UCI (e2e4).
"""

from __future__ import annotations

import argparse

import chess

from .agent import DQNAgent
from .encoding import action_to_move


def read_human_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("your move > ").strip()
        try:
            return board.parse_san(raw)
        except ValueError:
            pass
        try:
            move = chess.Move.from_uci(raw)
            if move in board.legal_moves:
                return move
        except ValueError:
            pass
        print("  illegal or unparsable — try again (SAN like Nf3, or UCI like g1f3)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play vs GMAI")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--color", choices=["white", "black"], default="white")
    args = parser.parse_args()

    agent = DQNAgent.from_checkpoint(args.checkpoint)
    agent.epsilon = 0.0

    human = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()

    print(board.unicode(borders=True), "\n")
    while not board.is_game_over(claim_draw=True):
        if board.turn == human:
            move = read_human_move(board)
        else:
            move = action_to_move(agent.act(board, greedy=True), board)
            print(f"GMAI plays {board.san(move)}")
        board.push(move)
        print(board.unicode(borders=True), "\n")

    outcome = board.outcome(claim_draw=True)
    print(f"Game over: {outcome.result()} ({outcome.termination.name})")


if __name__ == "__main__":
    main()
