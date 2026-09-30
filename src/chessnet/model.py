"""Policy-value network.

A residual convolutional tower in the AlphaZero/Leela shape, with two heads:

``policy``
    Logits over the 4096 ``from x to`` actions. This is what plays.
``value``
    A scalar in [-1, 1] predicting the game result from the side-to-move's
    point of view.

The value head is not needed to select moves — a policy alone plays at
roughly Maia strength — but it costs almost nothing to train alongside, and it
is the component MCTS needs later. Training it now avoids reprocessing the data
when search gets added.

Normalisation is BatchNorm here, unlike the DQN work: this is plain supervised
learning on a fixed, i.i.d. dataset, which is exactly the setting BatchNorm was
designed for.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoding import N_ACTIONS, N_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ChessNet(nn.Module):
    def __init__(
        self,
        channels: int = 192,
        blocks: int = 10,
        policy_channels: int = 32,
        value_channels: int = 8,
        value_hidden: int = 256,
    ):
        super().__init__()
        self.config = {
            "channels": channels,
            "blocks": blocks,
            "policy_channels": policy_channels,
            "value_channels": value_channels,
            "value_hidden": value_hidden,
        }

        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, 1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_channels * 64, N_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_channels, 1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_channels * 64, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tower(self.stem(x))
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self.config, "state_dict": self.state_dict(), **extra}, path
        )

    @classmethod
    def load(cls, path, device: str | None = None) -> ChessNet:
        """Rebuild with the architecture recorded in the checkpoint."""
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        return model


def masked_policy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal actions to -inf before softmax or argmax."""
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
