"""UCI protocol adapter.

Exposes the trained agent as a standard UCI engine, so it can be driven by
any chess GUI (Arena, Cute Chess, En Croissant, BanksiaGUI) or bridged to
Lichess with ``lichess-bot``.

The agent is a *policy*, not a search: it answers in one forward pass, so
``go`` ignores time-control parameters (wtime/btime/movetime) and replies
immediately. That is legal UCI behaviour — the engine is simply very fast.

Usage:
    python -m gmai.uci --checkpoint runs/<run>/final.pt

Then in the GUI, register this command as a UCI engine. Manual smoke test:

    uci
    isready
    position startpos moves e2e4
    go
    quit
"""

from __future__ import annotations

import argparse
import sys

import chess

from .agent import DQNAgent
from .encoding import action_to_move
from .model import masked_q_values

ENGINE_NAME = "GMAI (Grand Master AI)"
ENGINE_AUTHOR = "Luis Miguel Pereira Silva"


def _send(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _parse_position(board: chess.Board, tokens: list[str]) -> chess.Board:
    """Handle ``position [startpos | fen <FEN>] [moves <m1> <m2> ...]``."""
    if not tokens:
        return board

    if tokens[0] == "startpos":
        board = chess.Board()
        rest = tokens[1:]
    elif tokens[0] == "fen":
        fen = " ".join(tokens[1:7])
        board = chess.Board(fen)
        rest = tokens[7:]
    else:
        return board

    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            try:
                board.push(chess.Move.from_uci(uci))
            except ValueError:
                break  # malformed input: keep what we have
    return board


def _best_move(agent: DQNAgent, board: chess.Board) -> chess.Move | None:
    if board.is_game_over(claim_draw=True):
        return None
    action = agent.act(board, greedy=True)
    return action_to_move(action, board)


def uci_loop(agent: DQNAgent) -> None:
    board = chess.Board()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            _send(f"id name {ENGINE_NAME}")
            _send(f"id author {ENGINE_AUTHOR}")
            _send("uciok")

        elif cmd == "isready":
            _send("readyok")

        elif cmd == "ucinewgame":
            board = chess.Board()

        elif cmd == "position":
            board = _parse_position(board, tokens[1:])

        elif cmd == "go":
            move = _best_move(agent, board)
            _send(f"bestmove {move.uci() if move else '0000'}")

        elif cmd == "stop":
            move = _best_move(agent, board)
            _send(f"bestmove {move.uci() if move else '0000'}")

        elif cmd == "eval":  # non-standard, handy for debugging
            _send(f"info string fen {board.fen()}")

        elif cmd == "quit":
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GMAI as a UCI engine")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    agent = DQNAgent.from_checkpoint(args.checkpoint, device=args.device)
    agent.epsilon = 0.0
    agent.online.eval()

    uci_loop(agent)


if __name__ == "__main__":
    main()
