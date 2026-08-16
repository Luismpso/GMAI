"""Gymnasium chess environment.

Single-agent formulation: the environment owns the *opponent* (any
``Opponent`` policy) and plays its reply inside ``step``, so one call =
one full ply pair.

``info`` carries ``action_mask`` (shape (4096,), bool) — needed both to pick
a legal move and to compute the dueling baseline over legal actions only.

**Scope.** The default configuration samples *forced-mate endgames* rather
than the initial position (see :mod:`gmai.endgames`). A search-free DQN can
actually solve those; full chess from move 1 is out of scope and documented
as such. Pass ``position_sampler=None`` to get standard chess back.
"""

from __future__ import annotations

from typing import Any, Callable

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
from .rewards import PotentialFn, material_potential, shaping_reward, terminal_reward


class ChessEnv(gym.Env):
    """Chess vs. a pluggable opponent, with optional reward shaping."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent: Opponent | None = None,
        agent_color: chess.Color | None = None,
        position_sampler: Callable[[], Any] | None = None,
        max_moves: int = 200,
        gamma: float = 0.99,
        use_shaping: bool = True,
        potential_fn: PotentialFn = material_potential,
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
        self.position_sampler = position_sampler
        self.default_max_moves = max_moves
        self.max_moves = max_moves
        self.gamma = gamma
        self.use_shaping = use_shaping
        self.potential_fn = potential_fn
        self.draw_reward = draw_reward

        self.board = chess.Board()
        self.agent_color: chess.Color = chess.WHITE
        self.position_kind: str = "startpos"
        self._episode = 0
        self._start_ply = 0

    # ------------------------------------------------------------------ #
    @property
    def agent_plies(self) -> int:
        """Plies played since the episode started (excludes the sampled setup)."""
        return len(self.board.move_stack) - self._start_ply

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": legal_action_mask(self.board),
            "fen": self.board.fen(),
            "agent_color": self.agent_color,
            "position_kind": self.position_kind,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.position_sampler is not None:
            position = self.position_sampler()
            self.board = position.board
            self.agent_color = position.strong_color
            self.position_kind = position.kind
            self.max_moves = position.max_moves
        else:
            self.board = chess.Board()
            self.position_kind = "startpos"
            self.max_moves = self.default_max_moves
            if self._fixed_color is None:
                self.agent_color = (
                    chess.WHITE if self._episode % 2 == 0 else chess.BLACK
                )
            else:
                self.agent_color = self._fixed_color

        self._episode += 1
        self._start_ply = len(self.board.move_stack)

        # If it is the opponent's turn in the sampled position, let it move.
        if self.board.turn != self.agent_color and not self.board.is_game_over(
            claim_draw=True
        ):
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
        # Truncation is a training-loop artifact, reported separately
        # so the learner still bootstraps the successor's value.
        truncated = not terminated and self.agent_plies >= 2 * self.max_moves

        reward = terminal_reward(self.board, self.agent_color, self.draw_reward)
        if self.use_shaping:
            # Phi(terminal) = 0 keeps the shaping policy-invariant; a
            # truncated state is NOT terminal, so its potential still counts.
            reward += shaping_reward(
                board_before,
                self.board,
                self.agent_color,
                self.gamma,
                potential_fn=self.potential_fn,
                after_is_terminal=terminated,
            )

        return encode_board(self.board), reward, terminated, truncated, self._info()

    def render(self) -> str:
        return str(self.board)
