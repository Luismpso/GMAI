"""Monte Carlo tree search over the policy-value network (AlphaZero-style PUCT).

The policy alone plays on instinct: one forward pass, best-looking move. Search
looks ahead. Each simulation walks down the tree choosing moves by

    Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

where P is the policy prior, Q the mean value seen below that move and N the
visit counts, then evaluates the position it reaches with the network and backs
the value up the path. Checkmate, stalemate and the draw rules are detected
exactly, so the search finds short tactics even where the value head is unsure.

Conventions, which the tests pin down:

* The network's value is from the point of view of the side to move, matching
  how the training targets were stored.
* A node's ``value_sum`` is from the point of view of the player who made the
  move *into* that node, so a parent picks the child with the highest Q.

Python is slow at tree walking, so leaves are evaluated in batches: several
simulations descend before one forward pass, with a *virtual loss* steering
each away from the paths the others took.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .encoding import N_ACTIONS, bitboards_to_planes, board_arrays, move_to_action

VIRTUAL_LOSS = 1.0
UNDERPROMOTION_SCALE = 0.05  # the action space folds every promotion into one index


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "terminal")

    def __init__(self, prior: float):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, Node] | None = None  # None until expanded
        self.terminal: float | None = None  # value for the side to move, if over

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    move: chess.Move
    q: float  # expected result for the side to move at the root, in [-1, 1]
    nodes: int
    seconds: float
    visits: dict[chess.Move, int] = field(default_factory=dict)
    child_q: dict[chess.Move, float] = field(default_factory=dict)
    pv: list[chess.Move] = field(default_factory=list)

    @property
    def nps(self) -> float:
        return self.nodes / max(self.seconds, 1e-9)

    @property
    def centipawns(self) -> int:
        """Value mapped to a centipawn-like score for UCI front-ends."""
        q = max(-0.99, min(0.99, self.q))
        return int(round(111.714640912 * math.tan(1.5620688421 * q)))


def terminal_value(
    board: chess.Board, moves: list[chess.Move] | None = None
) -> float | None:
    """Game-theoretic value for the side to move, or None if the game goes on."""
    if moves is None:
        moves = list(board.legal_moves)
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0.0
    if board.is_repetition(3):
        return 0.0
    return None


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        c_puct: float = 1.8,
        batch_size: int = 32,
        fpu_reduction: float = 0.2,
    ):
        self.model = model
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.fpu_reduction = fpu_reduction

    # ---------------------------------------------------------------- public
    def run(
        self,
        board: chess.Board,
        nodes: int = 800,
        movetime: float | None = None,
        temperature: float = 0.0,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> SearchResult:
        """Search from ``board`` (left unchanged) and pick a move.

        Stops after ``nodes`` simulations or ``movetime`` seconds, whichever
        comes first. ``temperature`` > 0 samples the move in proportion to
        visits**(1/T) instead of taking the most visited; ``noise`` mixes
        Dirichlet noise into the root priors, for variety in casual play.
        """
        if terminal_value(board) is not None:
            raise ValueError("the game is already over in this position")
        rng = rng or np.random.default_rng()
        work = board.copy()
        root = Node(1.0)
        started = time.perf_counter()
        done = 0
        nodes = max(1, nodes)

        noised = False
        while done < nodes:
            done += self._simulate_batch(root, work, min(self.batch_size, nodes - done))
            if noise > 0 and root.children and not noised:
                self._add_noise(root, noise, rng)
                noised = True
            if movetime is not None and time.perf_counter() - started >= movetime:
                break

        elapsed = time.perf_counter() - started
        return self._result(root, done, elapsed, temperature, rng)

    # ----------------------------------------------------------- simulations
    def _simulate_batch(self, root: Node, board: chess.Board, size: int) -> int:
        pending: list[tuple[list[Node], Node, tuple, np.ndarray, list, list]] = []
        pending_ids: set[int] = set()
        completed = 0

        for _ in range(size):
            path = self._select(root, board)
            leaf = path[-1]
            depth = len(path) - 1

            if leaf.terminal is not None:
                self._backup(path, leaf.terminal)
                completed += 1
            elif id(leaf) in pending_ids:
                # Another simulation in this batch is already evaluating this
                # leaf. Undo this one and evaluate what we have.
                self._revert(path)
                for _ in range(depth):
                    board.pop()
                break
            else:
                moves = list(board.legal_moves)
                value = terminal_value(board, moves)
                if value is not None:
                    leaf.terminal = value
                    leaf.children = {}
                    self._backup(path, value)
                    completed += 1
                else:
                    actions = [move_to_action(m, board) for m in moves]
                    mask = np.zeros(N_ACTIONS, dtype=bool)
                    mask[actions] = True
                    pending.append(
                        (path, leaf, board_arrays(board), mask, moves, actions)
                    )
                    pending_ids.add(id(leaf))

            for _ in range(depth):
                board.pop()

        if pending:
            planes = bitboards_to_planes(
                np.stack([p[2][0] for p in pending]), np.stack([p[2][1] for p in pending])
            )
            priors_batch, values = self._evaluate(
                planes, np.stack([p[3] for p in pending])
            )
            for (path, leaf, _, _, moves, actions), probs, value in zip(
                pending, priors_batch, values, strict=True
            ):
                self._expand(leaf, moves, actions, probs)
                self._backup(path, float(value))
                completed += 1
        return completed

    def _select(self, root: Node, board: chess.Board) -> list[Node]:
        node = root
        path = [node]
        self._apply_virtual_loss(node)
        while node.children:  # expanded and has moves
            move, node = self._best_child(node)
            board.push(move)
            path.append(node)
            self._apply_virtual_loss(node)
        return path

    def _best_child(self, node: Node) -> tuple[chess.Move, Node]:
        sqrt_n = math.sqrt(max(1, node.visits))
        # First-play urgency: unvisited moves start a little below the
        # parent's own value, so the search tries likely moves first.
        fpu = -node.q - self.fpu_reduction
        best_score = -math.inf
        best = None
        for move, child in node.children.items():
            q = child.q if child.visits else fpu
            score = q + self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            if score > best_score:
                best_score, best = score, (move, child)
        return best

    @staticmethod
    def _apply_virtual_loss(node: Node) -> None:
        node.visits += 1
        node.value_sum -= VIRTUAL_LOSS

    @staticmethod
    def _revert(path: list[Node]) -> None:
        for node in path:
            node.visits -= 1
            node.value_sum += VIRTUAL_LOSS

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        """``value`` is for the side to move at the leaf."""
        v = -value  # the leaf's stats belong to the player who moved into it
        for node in reversed(path):
            node.value_sum += VIRTUAL_LOSS + v
            v = -v

    @staticmethod
    def _expand(node: Node, moves: list, actions: list, probs: np.ndarray) -> None:
        priors = probs[actions].astype(np.float64)
        for i, move in enumerate(moves):
            if move.promotion is not None and move.promotion != chess.QUEEN:
                priors[i] *= UNDERPROMOTION_SCALE
        total = priors.sum()
        priors = priors / total if total > 0 else np.full(len(moves), 1 / len(moves))
        node.children = {m: Node(float(p)) for m, p in zip(moves, priors, strict=True)}

    @torch.inference_mode()
    def _evaluate(self, planes: np.ndarray, masks: np.ndarray):
        x = torch.from_numpy(planes).to(self.device)
        m = torch.from_numpy(masks).to(self.device)
        logits, value = self.model(x)
        logits = logits.float().masked_fill(~m, torch.finfo(torch.float32).min)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy(), value.float().cpu().numpy()

    @staticmethod
    def _add_noise(root: Node, eps: float, rng: np.random.Generator) -> None:
        children = list(root.children.values())
        noise = rng.dirichlet([0.3] * len(children))
        for child, n in zip(children, noise, strict=True):
            child.prior = (1 - eps) * child.prior + eps * float(n)

    # ---------------------------------------------------------------- result
    def _result(self, root, done, elapsed, temperature, rng) -> SearchResult:
        children = root.children or {}
        if not children:
            raise ValueError("no legal moves at the root")
        visits = {m: c.visits for m, c in children.items()}
        if temperature > 0:
            moves = list(visits)
            weights = np.array([visits[m] for m in moves], dtype=np.float64) ** (
                1 / temperature
            )
            if weights.sum() == 0:
                weights = np.ones(len(moves))
            move = moves[int(rng.choice(len(moves), p=weights / weights.sum()))]
        else:
            move = max(children, key=lambda m: (children[m].visits, children[m].q))

        pv = [move]
        node = children[move]
        while node.children:
            nxt = max(node.children, key=lambda m: node.children[m].visits)
            if node.children[nxt].visits == 0:
                break
            pv.append(nxt)
            node = node.children[nxt]

        return SearchResult(
            move=move,
            q=children[move].q,
            nodes=done,
            seconds=elapsed,
            visits=visits,
            child_q={m: c.q for m, c in children.items()},
            pv=pv,
        )
