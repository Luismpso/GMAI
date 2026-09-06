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
import torch.nn.functional as F

from .encoding import N_ACTIONS, N_PLANES


def _norm(channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm with a group count that always divides `channels`."""
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _trunk_layers(channels: int, n_blocks: int) -> list[nn.Module]:
    """Conv trunk shared by the DQN and policy/value networks.

    Returned as a flat list so callers wrap it in ``nn.Sequential(*layers)``
    identically — the resulting ``state_dict`` keys are the same either way, so
    factoring this out does not change existing checkpoints.
    """
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
    return layers


class DuelingChessNet(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 4, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(*_trunk_layers(channels, n_blocks))

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Q-values. ``mask`` (B, 4096) restricts the advantage mean to legal moves.

        Passing ``mask=None`` falls back to the mean over all 4096 actions,
        which is only correct when every action is legal — never true in
        chess. Callers should always supply the mask; the fallback exists so
        the module stays usable in isolation (e.g. shape tests).
        """
        z = self.trunk(x)
        value = self.value_head(z)  # (B, 1)
        advantage = self.advantage_head(z)  # (B, 4096)

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


class _ResidualBlock(nn.Module):
    """AlphaZero-style residual block: two 3x3 convs with a skip connection.

    The skip connection is what lets the trunk go deep (12+ blocks) without the
    vanishing gradients that a plain convolutional stack — like
    :class:`DuelingChessNet`'s — suffers past a handful of layers.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.norm2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class PolicyValueNet(nn.Module):
    """Residual policy + value network for imitation learning on human games.

    A convolutional stem feeds ``n_blocks`` residual blocks, then two
    AlphaZero-style heads:

    * **policy** — 4096 logits over ``from*64+to`` actions, trained by
      cross-entropy against the move a strong human actually played.
    * **value**  — a single ``tanh`` scalar in ``[-1, 1]``, trained by MSE
      against the game result from the mover's point of view (+1 the side to
      move went on to win, -1 lost, 0 drew).

    Illegal actions are **not** masked during training: the target is always a
    single legal move, so cross-entropy teaches legality implicitly, and
    skipping mask construction keeps the dataloader cheap. Masking is applied at
    inference time instead (see :mod:`gmai.selector`).

    Unlike :class:`DuelingChessNet` (kept as-is for the endgame DQN), this net
    is free to define its own trunk — no checkpoint depends on its layout.
    """

    def __init__(self, channels: int = 128, n_blocks: int = 12, hidden: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            _norm(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(_ResidualBlock(channels) for _ in range(n_blocks))

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            _norm(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, N_ACTIONS)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            _norm(8),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8 * 8, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits (B, 4096), value (B,) in [-1, 1])``."""
        z = self.stem(x)
        for block in self.blocks:
            z = block(z)
        logits = self.policy_fc(torch.flatten(self.policy_conv(z), 1))
        value = torch.tanh(self.value_fc(torch.flatten(self.value_conv(z), 1))).squeeze(
            -1
        )
        return logits, value

    @classmethod
    def from_checkpoint(cls, path, device: str | None = None) -> PolicyValueNet:
        """Rebuild the network with the architecture stored in the checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        net = cls(**ckpt.get("arch", {}))
        net.load_state_dict(ckpt["policy_value"])
        if device is not None:
            net.to(device)
        net.eval()
        return net
