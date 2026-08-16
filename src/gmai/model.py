"""Dueling Q-network.

    Q(s, a) = V(s) + A(s, a) - mean_{a' in LEGAL(s)} A(s, a')

The mean in the dueling decomposition must be taken over the
*legal* actions only. Averaging over all 4096 outputs, when ~30 are legal in
a typical position (and ~3-10 in an endgame), subtracts the mean of ~4000
never-trained outputs from every Q-value: pure noise injected into every
estimate, and a V(s) that never converges. This was the single most damaging
bug in an earlier version of this network; see docs/POSTMORTEM.md.

BatchNorm is deliberately avoided in favour of GroupNorm. BatchNorm is a poor fit for
DQN: the running statistics are estimated from a replay batch whose
distribution shifts continuously as the policy changes, and acting on a
single state (batch of 1) uses different statistics than learning does.
GroupNorm normalises per-sample, so train and act agree by construction and
there are no running stats to poison.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import N_ACTIONS, N_PLANES


def _norm(channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm with a group count that always divides `channels`."""
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class DuelingChessNet(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 4, hidden: int = 512):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(N_PLANES, channels, kernel_size=3, padding=1),
            _norm(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks - 1):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                _norm(channels),
                nn.ReLU(inplace=True),
            ]
        self.trunk = nn.Sequential(*layers)

        flat = channels * 8 * 8
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Q-values. ``mask`` (B, 4096) restricts the advantage mean to legal moves.

        Passing ``mask=None`` falls back to the mean over all 4096 actions,
        which is only correct when every action is legal — never true in
        chess. Callers should always supply the mask; the fallback exists so
        the module stays usable in isolation (e.g. shape tests).
        """
        z = self.trunk(x)
        value = self.value_head(z)            # (B, 1)
        advantage = self.advantage_head(z)    # (B, 4096)

        if mask is None:
            baseline = advantage.mean(dim=1, keepdim=True)
        else:
            mask_f = mask.to(advantage.dtype)
            n_legal = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            baseline = (advantage * mask_f).sum(dim=1, keepdim=True) / n_legal

        return value + advantage - baseline


def masked_q_values(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set Q-values of illegal actions to -inf (used for argmax/max)."""
    neg_inf = torch.finfo(q.dtype).min
    return q.masked_fill(~mask, neg_inf)
