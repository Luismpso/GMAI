"""Shape, range and trainability checks for the policy/value network."""

from __future__ import annotations

import torch

from gmai.encoding import N_ACTIONS, N_PLANES
from gmai.model import PolicyValueNet


def _batch(n: int = 4) -> torch.Tensor:
    return torch.rand(n, N_PLANES, 8, 8)


def test_output_shapes_and_value_range():
    net = PolicyValueNet(channels=16, n_blocks=2, hidden=64)
    logits, value = net(_batch(4))
    assert logits.shape == (4, N_ACTIONS)
    assert value.shape == (4,)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_single_train_step_reduces_loss_on_fixed_batch():
    """One batch, overfit a few steps: loss must go down, proving grads flow."""
    torch.manual_seed(0)
    net = PolicyValueNet(channels=16, n_blocks=2, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    planes = _batch(8)
    actions = torch.randint(0, N_ACTIONS, (8,))
    targets = torch.empty(8).uniform_(-1, 1)

    def loss_fn():
        logits, value = net(planes)
        return torch.nn.functional.cross_entropy(
            logits, actions
        ) + torch.nn.functional.mse_loss(value, targets)

    first = loss_fn().item()
    for _ in range(20):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    assert loss_fn().item() < first


def test_from_checkpoint_roundtrip(tmp_path):
    net = PolicyValueNet(channels=16, n_blocks=2, hidden=64)
    ckpt = tmp_path / "pv.pt"
    torch.save(
        {
            "policy_value": net.state_dict(),
            "arch": {"channels": 16, "n_blocks": 2, "hidden": 64},
        },
        ckpt,
    )
    loaded = PolicyValueNet.from_checkpoint(ckpt)
    x = _batch(2)
    with torch.no_grad():
        a, b = net(x)
        c, d = loaded(x)
    assert torch.allclose(a, c) and torch.allclose(b, d)
