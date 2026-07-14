"""Gymnasium chess environment.

Single-agent formulation: the environment owns the *opponent* (any
``Opponent`` policy) and plays its reply inside ``step``, so one call =
one full ply pair. The learning agent's colour alternates every episode
unless fixed explicitly.

``info`` always carries ``action_mask`` (shape (4096,), bool) — the key
ingredient that makes DQN tractable on chess: illegal Q-values are masked
to -inf both when acting and when bootstrapping.
"""

from __future__ import annotations

from typing import Any

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .encoding import (
    N_ACTIONS,
    N_PLANES,
    action_to_move,
    encode_board,
    legal_action_mask,
)
from .opponents import Opponent, RandomOpponent
from .rewards import shaping_reward, terminal_reward


class ChessEnv(gym.Env):
    """Chess vs. a pluggable opponent, with optional reward shaping."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent: Opponent | None = None,
        agent_color: chess.Color | None = None,
        max_moves: int = 200,
        gamma: float = 0.99,
        use_shaping: bool = True,
        draw_reward: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_PLANES, 8, 8), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.opponent = opponent or RandomOpponent(seed=seed)
        self._fixed_color = agent_color
        self.max_moves = max_moves
        self.gamma = gamma
        self.use_shaping = use_shaping
        self.draw_reward = draw_reward

        self.board = chess.Board()
        self.agent_color: chess.Color = chess.WHITE
        self._episode = 0

    # ------------------------------------------------------------------ #
    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": legal_action_mask(self.board),
            "fen": self.board.fen(),
            "agent_color": self.agent_color,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = chess.Board()
        if self._fixed_color is None:
            self.agent_color = chess.WHITE if self._episode % 2 == 0 else chess.BLACK
        else:
            self.agent_color = self._fixed_color
        self._episode += 1

        if self.board.turn != self.agent_color:
            self.board.push(self.opponent.select_move(self.board))

        return encode_board(self.board), self._info()

    def step(self, action: int):
        if self.board.turn != self.agent_color:
            raise RuntimeError("step() called but it is the opponent's turn")

        board_before = self.board.copy(stack=False)
        move = action_to_move(int(action), self.board)  # raises if illegal
        self.board.push(move)

        # Opponent replies if the game is still going.
        if not self.board.is_game_over(claim_draw=True):
            self.board.push(self.opponent.select_move(self.board))

        terminated = self.board.is_game_over(claim_draw=True)
        truncated = not terminated and self.board.fullmove_number > self.max_moves

        reward = terminal_reward(self.board, self.agent_color, self.draw_reward)
        if self.use_shaping and not terminated:
            reward += shaping_reward(
                board_before, self.board, self.agent_color, self.gamma
            )

        return encode_board(self.board), reward, terminated, truncated, self._info()

    def render(self) -> str:
        return str(self.board)
