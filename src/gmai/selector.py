"""Move selection for the policy/value network.

The bot does not play the single highest-probability move. Instead, following
the design agreed for this project, it:

1. asks the policy for a distribution over legal moves and keeps the **top-k**
   (default 3);
2. does a **one-ply lookahead** — plays each candidate and asks the value head
   how good the resulting position is (from *our* side, so a low value for the
   opponent is a high "success" for us);
3. **blends** the two signals — how human-like the move is (policy) and how
   likely it is to win (value) — into a weight per candidate;
4. **samples** among the candidates by that weight.

The result is varied, human-like play that still leans toward winning moves and
is harder to exploit than a deterministic ``argmax``. A checkmate in one is
always played outright. The blend and randomness are controlled by
``policy_weight``, ``value_weight`` and ``temperature``.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from .encoding import action_to_move, encode_board, legal_action_mask


class Candidate:
    """One considered move with the signals behind it (handy for UCI ``info``)."""

    __slots__ = ("move", "policy", "success", "weight")

    def __init__(self, move: chess.Move, policy: float, success: float):
        self.move = move
        self.policy = policy  # policy probability among the legal moves
        self.success = success  # -value(child): higher is better for us
        self.weight = 0.0  # blended sampling weight, filled in by the selector

    def __repr__(self) -> str:
        return (
            f"Candidate({self.move.uci()}, p={self.policy:.3f}, "
            f"s={self.success:+.3f}, w={self.weight:.3f})"
        )


class MoveSelector:
    def __init__(
        self,
        net,
        device: str = "cpu",
        top_k: int = 3,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        temperature: float = 0.7,
        seed: int | None = None,
    ):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.top_k = top_k
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.temperature = max(temperature, 1e-6)
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def candidates(self, board: chess.Board) -> list[Candidate]:
        """Top-k legal moves with policy probability and one-ply success."""
        planes = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, _ = self.net(planes)

        mask = legal_action_mask(board)
        legal_idx = np.flatnonzero(mask)
        mask_t = torch.from_numpy(mask).to(self.device)
        probs = torch.softmax(logits[0].masked_fill(~mask_t, float("-inf")), dim=0)
        legal_probs = probs[legal_idx].cpu().numpy()

        order = np.argsort(-legal_probs)[: self.top_k]
        top_actions = legal_idx[order]
        top_probs = legal_probs[order]
        moves = [action_to_move(int(a), board) for a in top_actions]

        # One-ply lookahead. A checkmate we deliver is +inf success (always play
        # it); any other game-over child is a draw for us, worth 0. Everything
        # else is scored by the value head, negated because after our move the
        # value is from the opponent's point of view.
        child_planes, fixed = [], np.full(len(moves), np.nan)
        for j, move in enumerate(moves):
            board.push(move)
            if board.is_checkmate():
                fixed[j] = np.inf
            elif board.is_game_over(claim_draw=True):
                fixed[j] = 0.0
            child_planes.append(encode_board(board))
            board.pop()

        child_t = torch.from_numpy(np.stack(child_planes)).to(self.device)
        _, child_values = self.net(child_t)
        success = -child_values.cpu().numpy().astype(np.float64)
        override = ~np.isnan(fixed)
        success[override] = fixed[override]

        return [
            Candidate(m, float(p), float(s))
            for m, p, s in zip(moves, top_probs, success, strict=True)
        ]

    def _weights(self, candidates: list[Candidate]) -> np.ndarray:
        """Blend policy and success into a normalised sampling distribution."""
        policy = np.array([c.policy for c in candidates], dtype=np.float64)
        success = np.array([c.success for c in candidates], dtype=np.float64)

        # A guaranteed mate dominates: play it deterministically.
        if np.isinf(success).any():
            w = np.zeros(len(candidates))
            w[np.argmax(success)] = 1.0
            return w

        score = self.policy_weight * np.log(policy + 1e-12) + self.value_weight * success
        score = (score - score.max()) / self.temperature
        w = np.exp(score)
        return w / w.sum()

    def select(self, board: chess.Board) -> chess.Move | None:
        """Sample a move for ``board`` (``None`` if the game is already over)."""
        if board.is_game_over(claim_draw=True):
            return None
        cands = self.candidates(board)
        weights = self._weights(cands)
        for c, w in zip(cands, weights, strict=True):
            c.weight = float(w)
        idx = int(self._rng.choice(len(cands), p=weights))
        return cands[idx].move
