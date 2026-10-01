"""Move selection and UCI adapter.

Two ways to choose a move:

* **Policy only** (``--nodes 0``, the default): one forward pass, mask the
  illegal moves, take the most likely one. Instant, and plays the way the
  training data played.
* **Search** (``--nodes N``): Monte Carlo tree search guided by the policy and
  the value head (see :mod:`chessnet.search`). Slower per move, much harder to
  trick tactically.

    python -m chessnet.play --checkpoint runs/<run>/best.pt --fen "..."
    python -m chessnet.play --checkpoint runs/<run>/best.pt --fen "..." --nodes 800
    python -m chessnet.play --checkpoint runs/<run>/best.pt --uci --nodes 800

In UCI mode with search enabled, ``go nodes N``, ``go movetime MS`` and clock
controls (``wtime``/``btime``/``winc``/``binc``) are honoured.
"""

from __future__ import annotations

import argparse
import sys

import chess
import numpy as np
import torch

from .encoding import N_PLANES, action_to_move, encode_board, legal_action_mask
from .model import ChessNet, masked_policy
from .search import MCTS, SearchResult

ENGINE_NAME = "ChessNet"
ENGINE_AUTHOR = "Luis Miguel Pereira Silva"


class Player:
    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        temperature: float = 0.0,
        nodes: int = 0,
        c_puct: float = 1.8,
        batch_size: int = 32,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = ChessNet.load(checkpoint, device=str(self.device)).to(self.device)
        self.model.eval()
        self.temperature = temperature
        self.nodes = nodes
        self.search = MCTS(self.model, self.device, c_puct=c_puct, batch_size=batch_size)
        self.last_search: SearchResult | None = None
        if self.device.type == "cuda":
            self._warm_up(batch_size)

    @torch.inference_mode()
    def _warm_up(self, batch_size: int) -> None:
        """Pay the GPU's one-off start-up costs now rather than on the first move.

        The first CUDA calls in a process load kernels and set up libraries,
        which can take about a second: enough to lose time on the clock.
        """
        for n in (1, batch_size):
            self.model(torch.zeros(n, N_PLANES, 8, 8, device=self.device))
        torch.cuda.synchronize()

    def select(
        self,
        board: chess.Board,
        nodes: int | None = None,
        movetime: float | None = None,
    ) -> tuple[chess.Move, float, float]:
        """Return (move, confidence in it, value estimate for the side to move).

        Confidence is the policy probability without search, or the move's
        share of root visits with it.
        """
        nodes = self.nodes if nodes is None else nodes
        if nodes <= 0 and movetime is None:
            self.last_search = None
            return self._policy_move(board)
        result = self.search.run(
            board,
            nodes=nodes if nodes > 0 else 10**9,
            movetime=movetime,
            temperature=self.temperature,
        )
        self.last_search = result
        share = result.visits[result.move] / max(1, sum(result.visits.values()))
        return result.move, share, result.q

    @torch.no_grad()
    def _policy_move(self, board: chess.Board) -> tuple[chess.Move, float, float]:
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
        out = []
        for action in np.argsort(probs)[::-1][:k]:
            try:
                out.append(
                    (board.san(action_to_move(int(action), board)), float(probs[action]))
                )
            except ValueError:
                continue
        return out


# --------------------------------------------------------------------- UCI
def parse_go(tokens: list[str]) -> dict[str, int]:
    """``go`` arguments as integers; flags such as ``infinite`` map to 1."""
    params: dict[str, int] = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        if i + 1 < len(tokens) and tokens[i + 1].lstrip("-").isdigit():
            params[key] = int(tokens[i + 1])
            i += 2
        else:
            params[key] = 1
            i += 1
    return params


def time_budget(board: chess.Board, params: dict[str, int]) -> float | None:
    """Seconds to spend on this move, from UCI ``go`` parameters."""
    if "movetime" in params:
        return max(0.01, params["movetime"] / 1000)
    left = params.get("wtime" if board.turn == chess.WHITE else "btime")
    if left is None:
        return None
    inc = params.get("winc" if board.turn == chess.WHITE else "binc", 0)
    moves_to_go = params.get("movestogo", 30)
    budget = left / 1000 / max(moves_to_go, 1) + 0.8 * inc / 1000
    return max(0.02, min(budget, 0.25 * left / 1000))  # never risk the flag


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
        elif command == "go":
            if board.is_game_over(claim_draw=True):
                print("bestmove 0000", flush=True)
                continue
            params = parse_go(tokens[1:])
            if player.nodes <= 0 and "nodes" not in params:
                move, prob, value = player.select(board, nodes=0)
                print(
                    f"info depth 1 score cp {int(value * 300)} string policy {prob:.3f}"
                )
            else:
                nodes = params.get("nodes", player.nodes or 10**9)
                move, _, _ = player.select(
                    board, nodes=nodes, movetime=time_budget(board, params)
                )
                r = player.last_search
                print(
                    f"info depth {len(r.pv)} nodes {r.nodes} nps {int(r.nps)} "
                    f"time {int(r.seconds * 1000)} score cp {r.centipawns} "
                    f"pv {' '.join(m.uci() for m in r.pv)}"
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--nodes", type=int, default=0, help="search simulations (0 = policy only)"
    )
    ap.add_argument(
        "--movetime", type=float, default=None, help="search time limit in seconds"
    )
    ap.add_argument("--c-puct", type=float, default=1.8)
    ap.add_argument("--uci", action="store_true")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    args = ap.parse_args()

    player = Player(
        args.checkpoint, args.device, args.temperature, args.nodes, args.c_puct
    )
    if args.uci:
        uci_loop(player)
        return

    board = chess.Board(args.fen)
    print(board.unicode(borders=True))
    print("\npolicy (no search):")
    for san, p in player.top_moves(board):
        print(f"  {san:<8} {p:.3f}")

    if args.nodes > 0 or args.movetime:
        move, _, _ = player.select(board, movetime=args.movetime)
        r = player.last_search
        print(f"\nsearch: {r.nodes:,} nodes in {r.seconds:.2f}s ({r.nps:,.0f} nodes/s)")
        ranked = sorted(r.visits, key=r.visits.get, reverse=True)[:5]
        total = max(1, sum(r.visits.values()))
        for m in ranked:
            print(
                f"  {board.san(m):<8} visits {r.visits[m] / total:6.1%}  "
                f"expected {r.child_q[m]:+.3f}"
            )
        line = board.variation_san(r.pv)
        print(f"\nbest: {board.san(move)} | expected {r.q:+.3f} | line: {line}")
    else:
        move, prob, value = player.select(board)
        print(f"\nbest: {board.san(move)} (p={prob:.3f}) | value {value:+.3f}")


if __name__ == "__main__":
    main()
