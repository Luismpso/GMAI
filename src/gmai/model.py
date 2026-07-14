"""Dueling Q-network.

Conv trunk over the 18x8x8 board tensor, then two heads (Wang et al. 2016):

    Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

Chess is a natural fit for the dueling decomposition: in many positions the
*state* value (winning/losing) matters far more than fine-grained action
differences, so learning V(s) separately stabilises training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import N_ACTIONS, N_PLANES


class DuelingChessNet(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 4, hidden: int = 512):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(N_PLANES, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks - 1):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        value = self.value_head(z)                      # (B, 1)
        advantage = self.advantage_head(z)              # (B, 4096)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def masked_q_values(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set Q-values of illegal actions to -inf (used for argmax/max)."""
    neg_inf = torch.finfo(q.dtype).min
    return q.masked_fill(~mask, neg_inf)
