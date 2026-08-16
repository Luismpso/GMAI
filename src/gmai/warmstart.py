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
from .encoding import N_ACTIONS, encode_board, legal_action_mask, move_to_action
from .endgames import sample_endgame
from .tablebase import DRAW, EndgameTable


MAX_LEGAL = 48  # upper bound on legal moves in a 3-4 piece endgame


@dataclass
class QWarmStartData:
    """Regression targets: the exact Q-value of every legal action.

    Stored compactly — actions and their targets are padded to
    :data:`MAX_LEGAL` per position rather than materialising 4096 columns.
    """

    states: np.ndarray    # (N, 18, 8, 8) float32
    actions: np.ndarray   # (N, MAX_LEGAL) int64, -1 padding
    targets: np.ndarray   # (N, MAX_LEGAL) float32
    valid: np.ndarray     # (N, MAX_LEGAL) bool


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


def optimal_q_value(
    dtm_after: int,
    phi_before: float,
    gamma: float,
    draw_reward: float,
    mate: bool,
) -> float:
    """The exact Q-value of an action, on the scale the RL loop uses.

    Under optimal play a position with ``dtm_after`` plies to mate needs
    ``ceil(dtm_after / 2)`` further agent steps (one env step = agent move +
    opponent reply), so the undiscounted return is ``gamma ** steps``.

    Potential-based shaping offsets every Q-value by exactly ``-Phi(s)``::

        Q_shaped(s, a) = Q_unshaped(s, a) - Phi(s)

    so the shaped target is the discounted win minus the current potential.
    Getting this offset right is what makes the pre-trained values *drop in*
    to the RL loop instead of colliding with it.
    """
    if mate:
        unshaped = 1.0
    elif dtm_after >= DRAW:
        unshaped = draw_reward
    else:
        steps = (int(dtm_after) + 1) // 2
        unshaped = gamma**steps
    return unshaped - phi_before


def build_q_dataset(
    kind: str,
    table: EndgameTable,
    n_positions: int,
    seed: int = 0,
    gamma: float = 0.99,
    draw_reward: float = -1.0,
    potential_fn=None,
) -> QWarmStartData:
    """Label every legal action of sampled positions with its exact Q-value.

    Richer than imitation: instead of only naming the best move, this teaches
    how much *each* move is worth, which is precisely the quantity Q-learning
    goes on to refine.
    """
    from .rewards import endgame_potential

    potential_fn = potential_fn or endgame_potential
    rng = random.Random(seed)
    states, actions, targets, valid = [], [], [], []

    attempts = 0
    while len(states) < n_positions and attempts < n_positions * 50:
        attempts += 1
        position = sample_endgame(kind, rng=rng)
        board, strong = position.board, position.strong_color
        if board.turn != strong:
            continue
        if table.probe(board, strong) >= DRAW:
            continue  # theoretically drawn: nothing to teach

        phi = potential_fn(board, strong)
        acts, tgts = [], []
        for move in board.legal_moves:
            board.push(move)
            mate = board.is_checkmate()
            if mate:
                dtm_after = 0
            elif board.is_game_over(claim_draw=True):
                dtm_after = DRAW
            else:
                dtm_after = table.probe(board, strong)
            board.pop()

            acts.append(move_to_action(move, board))
            tgts.append(
                optimal_q_value(dtm_after, phi, gamma, draw_reward, mate)
            )
            if len(acts) == MAX_LEGAL:
                break

        pad = MAX_LEGAL - len(acts)
        valid.append([True] * len(acts) + [False] * pad)
        actions.append(acts + [0] * pad)
        targets.append(tgts + [0.0] * pad)
        states.append(encode_board(board))

    return QWarmStartData(
        states=np.stack(states),
        actions=np.asarray(actions, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
    )


def pretrain_q(
    agent: DQNAgent,
    data: QWarmStartData,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    ce_weight: float = 1.0,
    verbose: bool = True,
) -> list[float]:
    """Pre-train Q-values on exact targets, with an auxiliary ranking loss.

    Two objectives, because neither is sufficient alone:

    ``Huber(Q(s,a), Q*(s,a))`` over legal actions
        Fixes the **scale**. Cross-entropy is invariant to adding a constant
        to every logit, so an imitation-only network can rank moves perfectly
        while sitting hundreds of units from the return scale — and the first
        RL gradient then wipes out everything it learned.

    ``ce_weight * CrossEntropy(Q(s,·), argmax_a Q*(s,a))``
        Fixes the **ordering**. Pure regression is dominated by predicting
        ``Phi(s)``, the per-position offset, because that varies far more
        across positions than the discounted-win term varies across actions
        within one. The ranking term keeps the best move on top.

    The ranking weight is **annealed to zero** over training: early epochs are
    dominated by learning the policy, late epochs by pure regression, which
    pulls the scale back onto the return range without disturbing an ordering
    that is already correct. Without the anneal, cross-entropy keeps inflating
    the logits and the scale problem comes straight back.

    Set ``ce_weight=0`` for pure regression.
    """
    device = agent.device
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=lr)

    states = torch.from_numpy(data.states)
    actions = torch.from_numpy(data.actions)
    targets = torch.from_numpy(data.targets)
    valid = torch.from_numpy(data.valid)
    n = len(states)

    history: list[float] = []
    agent.online.train()
    for epoch in range(epochs):
        # Linear anneal: full ranking weight at the start, pure regression at
        # the end, so the final epochs calibrate the scale.
        w_ce = ce_weight * max(0.0, 1.0 - epoch / max(1, epochs - 1))
        perm = torch.randperm(n)
        losses, top1 = [], 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            s = states[idx].to(device)
            a = actions[idx].to(device)
            y = targets[idx].to(device)
            v = valid[idx].to(device)

            mask = torch.zeros(len(idx), N_ACTIONS, dtype=torch.bool, device=device)
            mask.scatter_(1, a, v)

            q_all = agent.online(s, mask)
            q = q_all.gather(1, a)
            loss = (F.smooth_l1_loss(q, y, reduction="none") * v).sum() / v.sum()

            if w_ce > 0:
                # Rank the teacher's move on top, on the masked action set.
                best_slot = y.masked_fill(~v, -1e9).argmax(dim=1)
                best_action = a.gather(1, best_slot.unsqueeze(1)).squeeze(1)
                logits = q_all.masked_fill(~mask, torch.finfo(q_all.dtype).min)
                loss = loss + w_ce * F.cross_entropy(logits, best_action)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), agent.grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))
            with torch.no_grad():
                pred = q.masked_fill(~v, -1e9).argmax(dim=1)
                best = y.masked_fill(~v, -1e9).argmax(dim=1)
                top1 += int(
                    (y.gather(1, pred.unsqueeze(1))
                     == y.gather(1, best.unsqueeze(1))).sum().item()
                )

        mean_loss = sum(losses) / len(losses)
        history.append(mean_loss)
        if verbose:
            print(
                f"  [warmstart-q] epoch {epoch + 1}/{epochs} "
                f"loss {mean_loss:.5f} | rank-w {w_ce:.2f} | "
                f"top-1 agreement {top1 / n:.3f}"
            )

    agent.sync_target()
    return history


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


class ImitationAnchor:
    """Keeps RL from drifting away from the demonstrated policy.

    A network pre-trained by cross-entropy ranks moves well but its Q-values
    sit far from the discounted-return scale (measured at ~-293 against a
    target range of [-1.9, +0.1]). The first TD gradients pull the whole
    output down by that amount, and the learned structure does not survive the
    trip — win-rate collapsed 0.52 -> 0.04 in one run.

    Rather than trying to pre-train perfectly scaled Q-values — which pits
    learning the ordering against learning the scale, since cross-entropy
    fixes only the former and regression is dominated by the per-position
    ``Phi(s)`` offset — this follows the DQfD idea (Hester et al., 2018): keep
    the demonstrations around *during* RL and apply a small auxiliary
    imitation loss alongside the TD loss. The TD loss is then free to move the
    scale, while the anchor holds the policy in place.

    ``weight`` scales the auxiliary gradient; ``every`` applies it once per N
    TD steps. Both default low — this is a regulariser, not a second training
    objective.
    """

    def __init__(
        self,
        agent: DQNAgent,
        data: "WarmStartData",
        lr: float = 1e-5,
        batch_size: int = 128,
        every: int = 1,
        seed: int | None = None,
    ):
        self.agent = agent
        self.states = torch.from_numpy(data.states)
        self.masks = torch.from_numpy(data.masks)
        self.targets = torch.from_numpy(data.targets)
        self.batch_size = batch_size
        self.every = max(1, every)
        self.optimizer = torch.optim.Adam(agent.online.parameters(), lr=lr)
        self._generator = torch.Generator().manual_seed(seed if seed is not None else 0)
        self._calls = 0

    def step(self) -> float | None:
        """One auxiliary imitation gradient step. Returns the loss, or None."""
        self._calls += 1
        if self._calls % self.every != 0:
            return None

        idx = torch.randint(
            0, len(self.targets), (self.batch_size,), generator=self._generator
        )
        device = self.agent.device
        s = self.states[idx].to(device)
        m = self.masks[idx].to(device)
        y = self.targets[idx].to(device)

        logits = self.agent.online(s, m)
        logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
        loss = F.cross_entropy(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.online.parameters(), self.agent.grad_clip
        )
        self.optimizer.step()
        return float(loss.item())


@torch.no_grad()
def calibrate_scale(
    agent: DQNAgent,
    data: QWarmStartData,
    verbose: bool = True,
) -> tuple[float, float]:
    """Rescale the network's output onto the true Q-value range, order-preserving.

    Cross-entropy pre-training learns the *ordering* of moves very well but is
    invariant to affine shifts of the logits, so the resulting values can sit
    hundreds of units away from the discounted-return scale. Dropping that
    straight into a TD loop produces enormous initial errors whose gradients
    destroy the learned representation — measured here at Q ~= -293 against a
    target range of [-1.9, +0.1].

    Rather than retrain, fit ``Q_true ~= alpha * Q_pred + beta`` and fold the
    two coefficients into the output layers. Because the dueling head is
    ``Q = V + A - mean(A)``, scaling both heads' final weights by ``alpha`` and
    shifting the value bias by ``beta`` reproduces ``alpha * Q + beta``
    exactly. With ``alpha > 0`` the ranking — and therefore the greedy policy —
    is provably unchanged.

    The slope is estimated from **within-position deviations only**. A naive
    pooled regression over all (state, action) pairs returns a non-positive
    slope and the fit has to be abandoned: cross-entropy constrains Q
    differences *inside* a position and leaves the overall level free to drift
    *between* positions, so the pooled covariance is dominated by
    between-position noise that carries no signal. Centring each position
    first isolates exactly the structure the pre-training actually learned.
    The intercept then comes from the global means.
    """
    device = agent.device
    agent.online.eval()

    dev_x, dev_y, flat_x, flat_y = [], [], [], []
    for start in range(0, len(data.states), 256):
        s_ = torch.from_numpy(data.states[start : start + 256]).to(device)
        a_ = torch.from_numpy(data.actions[start : start + 256]).to(device)
        y_ = torch.from_numpy(data.targets[start : start + 256]).to(device)
        v_ = torch.from_numpy(data.valid[start : start + 256]).to(device)

        mask = torch.zeros(len(s_), N_ACTIONS, dtype=torch.bool, device=device)
        mask.scatter_(1, a_, v_)
        q = agent.online(s_, mask).gather(1, a_).double()
        y_ = y_.double()
        vf = v_.double()
        n = vf.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Per-position means, then deviations from them.
        qx = (q * vf).sum(dim=1, keepdim=True) / n
        qy = (y_ * vf).sum(dim=1, keepdim=True) / n
        dev_x.append((q - qx)[v_].flatten().cpu())
        dev_y.append((y_ - qy)[v_].flatten().cpu())
        flat_x.append(q[v_].flatten().cpu())
        flat_y.append(y_[v_].flatten().cpu())

    dx, dy = torch.cat(dev_x), torch.cat(dev_y)
    x, y = torch.cat(flat_x), torch.cat(flat_y)

    var = (dx * dx).mean()
    alpha = float((dx * dy).mean() / var) if var > 0 else 1.0
    if alpha <= 0:  # no usable within-position signal
        if verbose:
            print("  [calibrate] non-positive within-position slope, skipping")
        return 1.0, 0.0
    beta = float(y.mean() - alpha * x.mean())

    value_last = agent.online.value_head[-1]
    adv_last = agent.online.advantage_head[-1]
    value_last.weight.mul_(alpha)
    value_last.bias.mul_(alpha).add_(beta)
    adv_last.weight.mul_(alpha)
    adv_last.bias.mul_(alpha)
    agent.sync_target()

    if verbose:
        print(
            f"  [calibrate] Q_true ~= {alpha:.5f} * Q_pred + {beta:.3f} "
            f"(pred range [{x.min():.1f}, {x.max():.1f}] -> "
            f"[{y.min():.2f}, {y.max():.2f}])"
        )
    return alpha, beta


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
