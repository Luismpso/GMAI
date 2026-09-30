"""Move selection and UCI adapter.

The policy alone plays: one forward pass, mask the illegal moves, take the
argmax. Sampling from the distribution instead of taking the maximum gives more
varied play, which matters for a bot that will face the same opponents
repeatedly.

    python -m chessnet.play --checkpoint runs/<run>/best.pt --uci
    python -m chessnet.play --checkpoint runs/<run>/best.pt --fen "..."
"""

from __future__ import annotations

import argparse
import sys

import chess
import numpy as np
import torch

from .encoding import action_to_move, encode_board, legal_action_mask
from .model import ChessNet, masked_policy

ENGINE_NAME = "ChessNet"
ENGINE_AUTHOR = "Luis Miguel Pereira Silva"


class Player:
    def __init__(
        self, checkpoint: str, device: str | None = None, temperature: float = 0.0
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = ChessNet.load(checkpoint, device=str(self.device)).to(self.device)
        self.model.eval()
        self.temperature = temperature

    @torch.no_grad()
    def select(self, board: chess.Board) -> tuple[chess.Move, float, float]:
        """Return (move, probability assigned to it, value estimate)."""
        planes = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(legal_action_mask(board)).unsqueeze(0).to(self.device)

        logits, value = self.model(planes)
        logits = masked_policy(logits, mask)

        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=1)[0]
            action = int(torch.multinomial(probs, 1).item())
        else:
            probs = torch.softmax(logits, dim=1)[0]
            action = int(logits.argmax(dim=1).item())

        return action_to_move(action, board), float(probs[action]), float(value[0])

    def top_moves(self, board: chess.Board, k: int = 5):
        """The k moves the policy likes best, for inspection."""
        with torch.no_grad():
            planes = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
            mask = torch.from_numpy(legal_action_mask(board)).unsqueeze(0).to(self.device)
            logits = masked_policy(self.model(planes)[0], mask)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        best = np.argsort(probs)[::-1][:k]
        out = []
        for action in best:
            try:
                out.append(
                    (
                        board.san(action_to_move(int(action), board)),
                        float(probs[action]),
                    )
                )
            except ValueError:
                continue
        return out


def uci_loop(player: Player) -> None:
    board = chess.Board()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            print(f"id name {ENGINE_NAME}")
            print(f"id author {ENGINE_AUTHOR}")
            print("uciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            board = chess.Board()
        elif command == "position":
            board = _parse_position(tokens[1:])
        elif command in ("go", "stop"):
            if board.is_game_over(claim_draw=True):
                print("bestmove 0000", flush=True)
            else:
                move, prob, value = player.select(board)
                print(
                    f"info depth 1 score cp {int(value * 300)} string policy {prob:.3f}"
                )
                print(f"bestmove {move.uci()}", flush=True)
        elif command == "quit":
            break


def _parse_position(tokens: list[str]) -> chess.Board:
    if not tokens:
        return chess.Board()
    if tokens[0] == "startpos":
        board, rest = chess.Board(), tokens[1:]
    elif tokens[0] == "fen":
        board, rest = chess.Board(" ".join(tokens[1:7])), tokens[7:]
    else:
        return chess.Board()
    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            try:
                board.push(chess.Move.from_uci(uci))
            except ValueError:
                break
    return board


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--uci", action="store_true")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    args = ap.parse_args()

    player = Player(args.checkpoint, args.device, args.temperature)
    if args.uci:
        uci_loop(player)
        return

    board = chess.Board(args.fen)
    print(board.unicode(borders=True))
    move, prob, value = player.select(board)
    print(f"\nbest: {board.san(move)} (p={prob:.3f}) | value {value:+.3f}")
    print("top moves:")
    for san, p in player.top_moves(board):
        print(f"  {san:<8} {p:.3f}")


if __name__ == "__main__":
    main()
