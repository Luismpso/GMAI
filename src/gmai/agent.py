"""Double DQN agent with legal-action masking.

Target (van Hasselt et al. 2016), with the argmax restricted to legal moves:

    a*  = argmax_{a legal} Q_online(s', a)
    y   = r + gamma * (1 - done) * Q_target(s', a*)

Selection uses the online network, evaluation the target network — the
decoupling that removes vanilla DQN's maximisation bias.
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
        was_training = self.online.training
        self.online.eval()
        q = self.online(state)
        if was_training:
            self.online.train()
        q = masked_q_values(q, torch.from_numpy(mask).unsqueeze(0).to(self.device))
        return int(q.argmax(dim=1).item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    # ----------------------------------------------------------- learning
    def train_step(self, batch: Batch) -> tuple[float, np.ndarray]:
        """One gradient step; returns (loss, per-sample TD errors)."""
        device = self.device
        states = torch.from_numpy(batch.states).to(device)
        actions = torch.from_numpy(batch.actions).to(device)
        rewards = torch.from_numpy(batch.rewards).to(device)
        next_states = torch.from_numpy(batch.next_states).to(device)
        next_masks = torch.from_numpy(batch.next_masks).to(device)
        dones = torch.from_numpy(batch.dones).to(device)

        q_sa = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Terminal states have empty masks; give them a dummy legal action
            # (the (1 - done) factor zeroes their contribution anyway).
            safe_masks = next_masks.clone()
            empty = ~safe_masks.any(dim=1)
            safe_masks[empty, 0] = True

            next_q_online = masked_q_values(self.online(next_states), safe_masks)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, best_actions)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_target.squeeze(1)

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
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.sync_target()
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.train_steps = ckpt.get("train_steps", 0)
