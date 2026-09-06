"""Supervised imitation training on extracted Lichess shards.

Trains :class:`gmai.model.PolicyValueNet` to (a) predict the move a strong
human played (policy, cross-entropy) and (b) predict the eventual game result
from the mover's point of view (value, MSE). This is behavioural cloning: the
network learns to imitate ~2000+ Elo full chess, in contrast to the DQN in
``gmai.train`` which self-plays forced-mate endgames.

Built for long runs (the target is a multi-day session on a full month of
games):

* **step-based**, not epoch-based, so a schedule and resume are well defined;
* **mixed precision** (AMP) on CUDA;
* **warmup + cosine** learning-rate schedule;
* **checkpointing** — ``last.pt`` (full state, for ``--resume``) every
  ``--ckpt-every`` steps, ``best.pt`` (lean, for inference) whenever validation
  improves, and the final weights at ``--out``.

    python scripts/train_supervised.py --data data/train --device cuda
    python scripts/train_supervised.py --resume runs/supervised/last.pt ...

The training core is exposed as :func:`run_training` so tests can drive it on a
tiny synthetic shard without the CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gmai.dataset import ShardIterableDataset, list_shards
from gmai.model import PolicyValueNet


@dataclass
class TrainConfig:
    data: Path
    out: Path = Path("runs/supervised/final.pt")
    epochs: int = 3
    total_steps: int | None = None  # overrides epochs when set
    max_steps: int | None = None  # hard cap (smoke tests)
    batch_size: int = 1024
    lr: float = 1e-3
    warmup_steps: int = 2000
    value_weight: float = 1.0
    channels: int = 128
    n_blocks: int = 12
    hidden: int = 256
    val_shards: int = 1
    ckpt_every: int = 2000
    eval_batches: int = 200  # cap validation cost during periodic evals
    log_every: int = 200
    amp: bool = True
    resume: Path | None = None
    device: str | None = None
    seed: int = 0
    arch: dict = field(init=False)

    def __post_init__(self):
        self.data = Path(self.data)
        self.out = Path(self.out)
        if self.resume is not None:
            self.resume = Path(self.resume)
        self.arch = {
            "channels": self.channels,
            "n_blocks": self.n_blocks,
            "hidden": self.hidden,
        }


def _split_shards(cfg: TrainConfig) -> tuple[list[Path], list[Path]]:
    shards = list_shards(cfg.data)
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz under {cfg.data}")
    n_val = max(0, cfg.val_shards)
    # n_val == 0 must not become shards[:-0] (== shards[:0], i.e. empty!); and
    # too few shards to hold any out means everything trains, nothing validates.
    if n_val == 0 or len(shards) <= n_val:
        return shards, []
    return shards[:-n_val], shards[-n_val:]


def _train_positions(cfg: TrainConfig, train_shards: list[Path], all_shards: int) -> int:
    """Position count for the train split, cheaply from the manifest if present."""
    manifest = cfg.data / "manifest.json"
    if manifest.exists() and all_shards:
        total = json.loads(manifest.read_text()).get("positions")
        if total:
            return int(total * len(train_shards) / all_shards)
    return sum(int(np.load(s)["actions"].shape[0]) for s in train_shards)


def _resolve_total_steps(
    cfg: TrainConfig, train_shards: list[Path], all_shards: int
) -> int:
    if cfg.max_steps is not None:
        return cfg.max_steps
    if cfg.total_steps is not None:
        return cfg.total_steps
    positions = _train_positions(cfg, train_shards, all_shards)
    return cfg.epochs * math.ceil(positions / cfg.batch_size)


def _make_scheduler(opt, total_steps: int, warmup: int):
    warmup = min(warmup, max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _step_metrics(logits, actions, value, targets):
    """Top-1 / top-3 move match and value MAE, as plain floats."""
    with torch.no_grad():
        top3 = logits.topk(3, dim=1).indices
        top1_acc = (top3[:, 0] == actions).float().mean().item()
        top3_acc = (top3 == actions.unsqueeze(1)).any(dim=1).float().mean().item()
        value_mae = (value - targets).abs().mean().item()
    return top1_acc, top3_acc, value_mae


def _loss(model, planes, actions, targets, value_weight):
    logits, value = model(planes)
    policy_loss = F.cross_entropy(logits, actions)
    value_loss = F.mse_loss(value, targets)
    return policy_loss + value_weight * value_loss, logits, value, policy_loss, value_loss


@torch.no_grad()
def evaluate(model, dataset, device, value_weight, max_batches=None) -> dict:
    model.eval()
    tot = dict(loss=0.0, top1=0.0, top3=0.0, mae=0.0, n=0)
    for i, (planes, actions, targets) in enumerate(dataset):
        if max_batches is not None and i >= max_batches:
            break
        planes, actions, targets = (
            planes.to(device),
            actions.to(device),
            targets.to(device),
        )
        loss, logits, value, _, _ = _loss(model, planes, actions, targets, value_weight)
        t1, t3, mae = _step_metrics(logits, actions, value, targets)
        bs = len(actions)
        tot["loss"] += loss.item() * bs
        tot["top1"] += t1 * bs
        tot["top3"] += t3 * bs
        tot["mae"] += mae * bs
        tot["n"] += bs
    model.train()
    n = max(tot["n"], 1)
    return {k: tot[k] / n for k in ("loss", "top1", "top3", "mae")}


def _save_lean(path: Path, model, cfg: TrainConfig, step: int, history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_value": model.state_dict(),
            "arch": cfg.arch,
            "kind": "policy_value",
            "step": step,
            "history": history,
        },
        path,
    )


def _save_full(path, model, opt, sched, scaler, cfg, step, best_val, history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_value": model.state_dict(),
            "arch": cfg.arch,
            "kind": "policy_value",
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val": best_val,
            "history": history,
        },
        path,
    )


def run_training(cfg: TrainConfig) -> dict:
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = cfg.amp and device.type == "cuda"
    if device.type == "cuda":
        # Input shapes are fixed (B, 18, 8, 8), so let cuDNN autotune kernels.
        torch.backends.cudnn.benchmark = True

    all_shards = list_shards(cfg.data)
    train_shards, val_shards = _split_shards(cfg)
    total_steps = _resolve_total_steps(cfg, train_shards, len(all_shards))
    print(f"device : {device} | amp={use_amp}")
    print(f"shards : {len(train_shards)} train, {len(val_shards)} val")
    print(f"steps  : {total_steps:,} (batch {cfg.batch_size})")

    model = PolicyValueNet(**cfg.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = _make_scheduler(opt, total_steps, cfg.warmup_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_ds = ShardIterableDataset(
        train_shards, cfg.batch_size, shuffle=True, seed=cfg.seed
    )
    val_ds = (
        ShardIterableDataset(val_shards, cfg.batch_size, shuffle=False, seed=cfg.seed)
        if val_shards
        else None
    )

    global_step, best_val = 0, math.inf
    history: list[dict] = []
    if cfg.resume is not None and cfg.resume.exists():
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["policy_value"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", math.inf)
        history = ckpt.get("history", [])
        print(f"resumed from {cfg.resume} at step {global_step:,}")

    last_path = cfg.out.parent / "last.pt"
    best_path = cfg.out.parent / "best.pt"

    def snapshot(win: dict, seen: int) -> dict:
        val = (
            evaluate(model, val_ds, device, cfg.value_weight, cfg.eval_batches)
            if val_ds
            else {}
        )
        train = {k: win[k] / max(seen, 1) for k in ("loss", "top1", "top3", "mae")}
        entry = {"step": global_step, "train": train, "val": val}
        history.append(entry)
        return val

    win = dict(loss=0.0, top1=0.0, top3=0.0, mae=0.0)
    seen = 0
    started = time.time()
    data_epoch = 0
    model.train()

    while global_step < total_steps:
        train_ds.set_epoch(data_epoch)
        data_epoch += 1
        for planes, actions, targets in train_ds:
            planes = planes.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, logits, value, p_loss, v_loss = _loss(
                    model, planes, actions, targets, cfg.value_weight
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1

            t1, t3, mae = _step_metrics(logits, actions, value, targets)
            win["loss"] += loss.item()
            win["top1"] += t1
            win["top3"] += t3
            win["mae"] += mae
            seen += 1

            if global_step % cfg.log_every == 0:
                rate = global_step * cfg.batch_size / (time.time() - started)
                print(
                    f"  step {global_step:>8,}/{total_steps:,} | "
                    f"loss {win['loss'] / seen:.3f} (p {p_loss.item():.3f} v {v_loss.item():.3f}) "
                    f"| top1 {win['top1'] / seen:.3f} top3 {win['top3'] / seen:.3f} "
                    f"| vmae {win['mae'] / seen:.3f} | lr {sched.get_last_lr()[0]:.2e} "
                    f"| {rate:,.0f} pos/s",
                    flush=True,
                )

            if global_step % cfg.ckpt_every == 0:
                val = snapshot(win, seen)
                _save_full(
                    last_path,
                    model,
                    opt,
                    sched,
                    scaler,
                    cfg,
                    global_step,
                    best_val,
                    history,
                )
                if val and val["loss"] < best_val:
                    best_val = val["loss"]
                    _save_lean(best_path, model, cfg, global_step, history)
                print(
                    f"  [ckpt] step {global_step:,} val={val} (best {best_val:.3f})",
                    flush=True,
                )
                win = dict(loss=0.0, top1=0.0, top3=0.0, mae=0.0)
                seen = 0

            if global_step >= total_steps:
                break

    # Always leave one final snapshot and the final weights behind.
    final_val = snapshot(win, seen) if seen else (history[-1]["val"] if history else {})
    if final_val and final_val.get("loss", math.inf) < best_val:
        best_val = final_val["loss"]
        _save_lean(best_path, model, cfg, global_step, history)
    _save_lean(cfg.out, model, cfg, global_step, history)
    (cfg.out.parent / "train_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    print(f"saved {cfg.out} ({global_step:,} steps, best val {best_val:.3f})")
    return {
        "history": history,
        "checkpoint": str(cfg.out),
        "best_val": best_val,
        "steps": global_step,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/train"))
    ap.add_argument("--out", type=Path, default=Path("runs/supervised/final.pt"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    run_training(TrainConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
