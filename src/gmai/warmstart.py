"""Supervised warm start from the exact solver.

Pre-trains the advantage head to imitate provably optimal play before RL
begins. This is the AlphaGo recipe — bootstrap from a teacher, then improve
by self-play — except the teacher here is not a human, it is the exact DTM
table from :mod:`gmai.tablebase`, so the labels are optimal by construction.

Why it matters: from a random start, an untrained agent almost never
stumbles into a mate, so the terminal reward is never observed and the
exploration problem dominates. Warm starting collapses that.

This is *not* cheating, and it is not the whole system either: the warm-started
agent still has to hold up under RL, and the evaluation is run against the same
baselines either way. It is documented in the model card.

Also provides :func:`dtm_quality`: **the fraction of the agent's moves that
worsen the distance to mate**. Unlike Elo, it is directly interpretable — it
says how often the agent throws away progress in a position it was already
winning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .agent import DQNAgent
from .encoding import encode_board, legal_action_mask, move_to_action
from .endgames import sample_endgame
from .tablebase import DRAW, EndgameTable


@dataclass
class WarmStartData:
    states: np.ndarray   # (N, 18, 8, 8) float32
    masks: np.ndarray    # (N, 4096)     bool
    targets: np.ndarray  # (N,)          int64 - an optimal action


def build_dataset(
    kind: str,
    table: EndgameTable,
    n_positions: int,
    seed: int = 0,
    max_dtm: int | None = None,
) -> WarmStartData:
    """Sample won positions and label them with an optimal move.

    ``max_dtm`` optionally restricts to positions close to mate, which is a
    useful curriculum: learn to finish first, then learn to make progress.
    """
    rng = random.Random(seed)
    states, masks, targets = [], [], []

    attempts = 0
    while len(states) < n_positions and attempts < n_positions * 50:
        attempts += 1
        position = sample_endgame(kind, rng=rng)
        board, strong = position.board, position.strong_color
        if board.turn != strong:
            continue
        dtm = table.probe(board, strong)
        if dtm >= DRAW:
            continue  # theoretically drawn: no optimal move to imitate
        if max_dtm is not None and dtm > max_dtm:
            continue

        best = table.best_moves(board, strong)
        if not best:
            continue

        states.append(encode_board(board))
        masks.append(legal_action_mask(board))
        targets.append(move_to_action(rng.choice(best), board))

    return WarmStartData(
        states=np.stack(states),
        masks=np.stack(masks),
        targets=np.asarray(targets, dtype=np.int64),
    )


def pretrain(
    agent: DQNAgent,
    data: WarmStartData,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = True,
) -> list[float]:
    """Cross-entropy pre-training of Q-values towards the optimal move.

    The loss is a masked softmax over legal actions only, so the network is
    never asked to rank the ~4 000 illegal outputs.
    """
    device = agent.device
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=lr)

    states = torch.from_numpy(data.states)
    masks = torch.from_numpy(data.masks)
    targets = torch.from_numpy(data.targets)
    n = len(targets)

    history: list[float] = []
    agent.online.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        losses, correct = [], 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            s = states[idx].to(device)
            m = masks[idx].to(device)
            y = targets[idx].to(device)

            logits = agent.online(s, m)
            logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), agent.grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))
            correct += int((logits.argmax(dim=1) == y).sum().item())

        mean_loss = sum(losses) / len(losses)
        accuracy = correct / n
        history.append(mean_loss)
        if verbose:
            print(
                f"  [warmstart] epoch {epoch + 1}/{epochs} "
                f"loss {mean_loss:.4f} | top-1 agreement {accuracy:.3f}"
            )

    agent.sync_target()
    return history


def dtm_quality(
    agent: DQNAgent,
    kind: str,
    table: EndgameTable,
    n_positions: int = 500,
    seed: int = 0,
) -> dict:
    """Fraction of the agent's moves that are optimal / neutral / worsening.

    For each sampled won position, compare the DTM before and after the
    agent's move (from the strong side's perspective, in plies):

    * ``optimal``   — the move preserves the shortest mate;
    * ``suboptimal``— still winning, but the mate got further away;
    * ``throw_away``— the position is no longer won at all.

    ``throw_away`` is the headline number: it counts moves that turn a forced
    win into a draw, which no amount of Elo can express as clearly.
    """
    rng = random.Random(seed)
    optimal = suboptimal = throw_away = 0
    mean_loss_plies: list[int] = []

    checked = 0
    attempts = 0
    while checked < n_positions and attempts < n_positions * 50:
        attempts += 1
        position = sample_endgame(kind, rng=rng)
        board, strong = position.board, position.strong_color
        if board.turn != strong:
            continue
        before = table.probe(board, strong)
        if before >= DRAW:
            continue

        move = table.best_moves(board, strong)  # ensure the position is playable
        if not move:
            continue

        action = agent.act(board, greedy=True)
        from .encoding import action_to_move

        board.push(action_to_move(action, board))
        if board.is_checkmate():
            after = -1  # mate delivered: strictly better than any DTM
        elif board.is_game_over(claim_draw=True):
            after = DRAW
        else:
            after = table.probe(board, strong)

        checked += 1
        if after >= DRAW:
            throw_away += 1
        elif after <= before - 1:
            optimal += 1
        else:
            suboptimal += 1
            mean_loss_plies.append(int(after) - int(before) + 1)

    total = max(checked, 1)
    return {
        "n": checked,
        "optimal_rate": round(optimal / total, 4),
        "suboptimal_rate": round(suboptimal / total, 4),
        "throw_away_rate": round(throw_away / total, 4),
        "mean_plies_lost": round(
            sum(mean_loss_plies) / len(mean_loss_plies), 2
        ) if mean_loss_plies else 0.0,
    }
