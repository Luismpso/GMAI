"""Double DQN agent with legal-action masking.

Target (van Hasselt et al. 2016), with the argmax restricted to legal moves:

    a*  = argmax_{a legal} Q_online(s', a)
    y   = r + gamma * (1 - TERMINATED) * Q_target(s', a*)

Selection uses the online network, evaluation the target network — the
decoupling that removes vanilla DQN's maximisation bias.

The factor is ``1 - terminated``, **not** ``1 - done``. Truncating
an episode at the move limit is an artifact of the training loop, not a
property of the MDP: the successor state still has value and must be
bootstrapped. Treating truncation as terminal told the agent that the vast
majority of positions were worth exactly zero.

Every forward pass receives the action mask, because
the dueling baseline is the mean advantage over *legal* actions.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .encoding import encode_board, legal_action_mask
from .model import DuelingChessNet, masked_q_values
from .replay_buffer import Batch


class DQNAgent:
    def __init__(
        self,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_sync_every: int = 1_000,
        grad_clip: float = 10.0,
        channels: int = 64,
        n_blocks: int = 4,
        hidden: int = 512,
        device: str | None = None,
        seed: int | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Kept so checkpoints are self-describing (see save/from_checkpoint).
        self.arch = {"channels": channels, "n_blocks": n_blocks, "hidden": hidden}
        self.online = DuelingChessNet(channels, n_blocks, hidden).to(self.device)
        self.target = DuelingChessNet(channels, n_blocks, hidden).to(self.device)
        self.sync_target()
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(
            1, epsilon_decay_steps
        )
        self.target_sync_every = target_sync_every
        self.grad_clip = grad_clip
        self.train_steps = 0
        self._rng = random.Random(seed)

    # ------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, board: chess.Board, greedy: bool = False) -> int:
        """Masked epsilon-greedy action for the current position."""
        mask = legal_action_mask(board)
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.choice(np.flatnonzero(mask)))

        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        q = masked_q_values(self.online(state, mask_t), mask_t)
        return int(q.argmax(dim=1).item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    # ----------------------------------------------------------- learning
    def train_step(self, batch: Batch) -> tuple[float, np.ndarray]:
        """One gradient step; returns (loss, per-sample TD errors)."""
        device = self.device
        states = torch.from_numpy(batch.states).to(device)
        masks = torch.from_numpy(batch.masks).to(device)
        actions = torch.from_numpy(batch.actions).to(device)
        rewards = torch.from_numpy(batch.rewards).to(device)
        next_states = torch.from_numpy(batch.next_states).to(device)
        next_masks = torch.from_numpy(batch.next_masks).to(device)
        terminated = torch.from_numpy(batch.terminated).to(device)

        q_sa = self.online(states, masks).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Terminal successors have empty masks; give them a dummy legal
            # action so the argmax is well defined. The (1 - terminated)
            # factor zeroes their contribution anyway.
            safe_masks = next_masks.clone()
            empty = ~safe_masks.any(dim=1)
            safe_masks[empty, 0] = True

            next_q_online = masked_q_values(
                self.online(next_states, safe_masks), safe_masks
            )
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states, safe_masks).gather(1, best_actions)
            targets = (
                rewards
                + self.gamma * (1.0 - terminated) * next_q_target.squeeze(1)
            )

        td_errors = q_sa - targets
        if batch.weights is not None:  # PER importance-sampling correction
            weights = torch.from_numpy(batch.weights).to(device)
            loss = (weights * F.smooth_l1_loss(q_sa, targets, reduction="none")).mean()
        else:
            loss = F.smooth_l1_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_sync_every == 0:
            self.sync_target()

        return float(loss.item()), td_errors.detach().cpu().numpy()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # -------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "arch": self.arch,  # makes the checkpoint self-describing
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load weights into *this* agent.

        Raises ``ValueError`` if the checkpoint was trained with a different
        architecture — use :meth:`from_checkpoint` to rebuild it instead.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        arch = ckpt.get("arch")
        if arch is not None and arch != self.arch:
            raise ValueError(
                f"checkpoint architecture {arch} != agent architecture "
                f"{self.arch}; use DQNAgent.from_checkpoint(path) instead"
            )
        self.online.load_state_dict(ckpt["online"])
        self.sync_target()
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.train_steps = ckpt.get("train_steps", 0)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | None = None, **kwargs
    ) -> "DQNAgent":
        """Rebuild an agent with the architecture stored in the checkpoint.

        This is what evaluation, play and UCI entry points should use: it
        works regardless of which model size the checkpoint was trained with.
        """
        head = torch.load(path, map_location="cpu", weights_only=True)
        arch = head.get("arch", {})
        agent = cls(device=device, **{**arch, **kwargs})
        agent.load(path)
        return agent
