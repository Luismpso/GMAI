"""Supervised training on human games.

Two losses, summed:

* **policy** — cross-entropy against the move actually played;
* **value**  — MSE against the game result from the side-to-move's view,
  weighted down because a game result is a very noisy label for any single
  position (won games contain plenty of bad positions).

Two ways to decide how long to train:

``--schedule plateau`` (default)
    Train until the model stops improving. Validation runs every
    ``--eval-every`` steps; the learning rate drops when validation accuracy
    stalls, and training ends after ``--patience`` evaluations without a new
    best. Safe to interrupt at any time with Ctrl+C: ``best.pt`` always holds
    the best model seen so far.

``--max-hours N``
    A fixed wall-clock budget. Measures throughput, sizes a one-cycle schedule
    to fit, and completes it. One-cycle usually ends slightly better than
    plateau for the same compute, but only if it runs to the end.

Progress is written to TensorBoard when it is installed::

    tensorboard --logdir runs

The metric to watch is top-1 accuracy on held-out positions. Maia reports
roughly 50% move-matching. Below ~35% means something is wrong; well above
~55% is worth checking for leakage between the splits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import ShardDataset
from .model import ChessNet


def evaluate(
    model,
    data: ShardDataset,
    device,
    batch_size: int,
    max_batches: int = 200,
    memory_format=torch.contiguous_format,
) -> dict:
    model.eval()
    losses, correct, top5, total = [], 0, 0, 0
    value_error = 0.0
    with torch.no_grad():
        for i, (planes, actions, results) in enumerate(
            data.iter_batches(batch_size, train=False, shuffle=False)
        ):
            if i >= max_batches:
                break
            planes = planes.to(device, non_blocking=True, memory_format=memory_format)
            actions = actions.to(device, non_blocking=True)
            results = results.to(device, non_blocking=True)
            logits, value = model(planes)
            losses.append(F.cross_entropy(logits.float(), actions).item())
            top = logits.topk(5, dim=1).indices
            correct += (top[:, 0] == actions).sum().item()
            top5 += (top == actions[:, None]).any(dim=1).sum().item()
            value_error += F.mse_loss(value.float(), results, reduction="sum").item()
            total += len(actions)
    model.train()
    return {
        "val_loss": sum(losses) / max(len(losses), 1),
        "top1": correct / max(total, 1),
        "top5": top5 / max(total, 1),
        "value_mse": value_error / max(total, 1),
    }


def _measure_throughput(model, data, device, args, optimizer, steps: int = 30) -> float:
    """Positions per second over a short warm-up, to size a time-budgeted run."""
    print("measuring throughput...", flush=True)
    use_amp = device.type == "cuda"
    seen, started = 0, None
    for i, (planes, actions, results) in enumerate(
        data.iter_batches(args.batch_size, train=True, seed=args.seed)
    ):
        if i == 5:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started, seen = time.time(), 0
        planes, actions, results = (t.to(device) for t in (planes, actions, results))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, value = model(planes)
            loss = F.cross_entropy(logits, actions) + args.value_weight * F.mse_loss(
                value, results
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if started is not None:
            seen += len(actions)
        if i >= steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - (started or time.time())
    for module in model.modules():  # undo the warm-up so it does not bias training
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    optimizer.state.clear()
    return seen / max(elapsed, 1e-6)


def _block_tensorflow_in_tensorboard() -> None:
    """Stop TensorBoard from importing TensorFlow behind our back.

    ``torch.utils.tensorboard`` does ``from tensorboard.compat import tf``,
    which imports the real TensorFlow whenever it is installed. TensorFlow then
    reserves nearly all GPU memory for itself. In one run that left PyTorch
    starved, the driver paged to system RAM, and training ran at ~2 000
    positions/s against ~17 000 for the identical configuration benchmarked in
    isolation.

    TensorBoard's documented switch: if ``tensorboard.compat.notf`` is
    importable, it always uses its lightweight stub instead of TensorFlow.
    Writing scalar event files needs nothing more.
    """
    import sys
    import types

    sys.modules.setdefault(
        "tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf")
    )


class _Logger:
    """TensorBoard if available, otherwise a no-op. Never required."""

    def __init__(self, log_dir: Path):
        _block_tensorflow_in_tensorboard()
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(log_dir))
            print(f"tensorboard: tensorboard --logdir {log_dir.parent.parent}")
        except ImportError:
            self.writer = None
            print(
                "tensorboard not installed (pip install tensorboard); "
                "progress is still saved to history.json"
            )

    def scalars(self, prefix: str, values: dict, step: int) -> None:
        if self.writer is None:
            return
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def train(args) -> Path:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    data = ShardDataset(
        args.data, val_fraction=args.val_fraction, seed=args.seed, limit=args.limit
    )
    print(data.summary())
    if data.manifest:
        print(
            f"source: {data.manifest.get('source')} | min_elo "
            f"{data.manifest.get('min_elo')} | {data.manifest.get('games_kept', 0):,} games"
        )

    if device.type == "cuda":
        # Input shapes never change, so let cuDNN time its algorithms once and
        # keep the fastest. The default heuristic choice can be far off on
        # small 8x8 spatial inputs.
        torch.backends.cudnn.benchmark = True
    memory_format = (
        torch.channels_last
        if device.type == "cuda" and args.channels_last
        else torch.contiguous_format
    )
    model = ChessNet(channels=args.channels, blocks=args.blocks).to(
        device, memory_format=memory_format
    )
    print(
        f"model : {args.channels}ch x {args.blocks} blocks, "
        f"{model.n_parameters / 1e6:.1f}M parameters on {device}"
    )
    base_model = model
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, data.n_train // args.batch_size)
    use_amp = device.type == "cuda"

    # ---------------------------------------------------------------- schedule
    schedule = "onecycle" if args.max_hours else args.schedule
    epochs = args.epochs
    onecycle = plateau = None
    if schedule == "onecycle":
        if args.max_hours:
            rate = _measure_throughput(model, data, device, args, optimizer)
            per_epoch = data.n_train / max(rate, 1)
            epochs = max(1, int(args.max_hours * 3600 * 0.92 // per_epoch))
            print(
                f"budget: {args.max_hours:.1f} h at {rate:,.0f} pos/s -> "
                f"{per_epoch / 60:.0f} min/epoch -> {epochs} epochs"
            )
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=epochs * steps_per_epoch,
            pct_start=0.05,
        )
    else:
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_decay,
            patience=args.lr_patience,
        )
        print(
            f"schedule: plateau | eval every {args.eval_every:,} steps "
            f"({args.eval_every * args.batch_size / 1e6:.1f}M positions) | "
            f"stop after {args.patience} evals without improvement"
        )

    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    logger = _Logger(run_dir / "tb")
    print(f"run   : {run_dir}\n")

    history: list[dict] = []
    best_top1, evals_since_best = 0.0, 0
    global_step, started_all = 0, time.time()
    deadline = started_all + args.max_hours * 3600 if args.max_hours else None
    stop_reason = "epochs exhausted"

    def run_eval(epoch: int) -> bool:
        """Validate, checkpoint, update schedule. Returns True to stop."""
        nonlocal best_top1, evals_since_best, stop_reason
        metrics = evaluate(
            model, data, device, args.batch_size, memory_format=memory_format
        )
        metrics.update(
            step=global_step,
            epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            hours=round((time.time() - started_all) / 3600, 2),
        )
        history.append(metrics)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        logger.scalars("val", metrics, global_step)

        improved = metrics["top1"] > best_top1
        print(
            f"[eval step {global_step:,}] val_loss {metrics['val_loss']:.4f} | "
            f"top1 {metrics['top1']:.3f} | top5 {metrics['top5']:.3f} | "
            f"value_mse {metrics['value_mse']:.3f} | lr {metrics['lr']:.2e} | "
            f"{metrics['hours']} h" + ("  * new best" if improved else ""),
            flush=True,
        )

        base_model.save(run_dir / "last.pt", step=global_step, metrics=metrics)
        if improved:
            best_top1, evals_since_best = metrics["top1"], 0
            base_model.save(run_dir / "best.pt", step=global_step, metrics=metrics)
        elif global_step >= args.warmup_steps:
            # Stalls during warm-up say nothing: the learning rate is still
            # climbing. Only count them once the schedule is in charge.
            evals_since_best += 1

        if plateau is not None and global_step >= args.warmup_steps:
            plateau.step(metrics["top1"])
            if evals_since_best >= args.patience:
                stop_reason = f"no improvement in {args.patience} evaluations"
                return True
            if optimizer.param_groups[0]["lr"] < args.min_lr:
                stop_reason = "learning rate fell below --min-lr"
                return True
        return False

    try:
        stop = False
        for epoch in range(1, epochs + 1):
            running, seen, correct = [], 0, 0
            t_epoch = time.time()
            for planes, actions, results in data.iter_batches(
                args.batch_size, train=True, seed=args.seed + epoch
            ):
                planes = planes.to(device, non_blocking=True, memory_format=memory_format)
                actions = actions.to(device, non_blocking=True)
                results = results.to(device, non_blocking=True)

                if plateau is not None and global_step < args.warmup_steps:
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr * (global_step + 1) / args.warmup_steps

                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits, value = model(planes)
                    policy_loss = F.cross_entropy(
                        logits, actions, label_smoothing=args.label_smoothing
                    )
                    value_loss = F.mse_loss(value, results)
                    loss = policy_loss + args.value_weight * value_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if (
                    onecycle is not None
                    and onecycle.last_epoch < onecycle.total_steps - 1
                ):
                    onecycle.step()
                global_step += 1

                running.append(policy_loss.item())
                correct += (logits.argmax(dim=1) == actions).sum().item()
                seen += len(actions)

                if global_step % args.log_every == 0:
                    rate = seen / (time.time() - t_epoch)
                    train_metrics = {
                        "policy_loss": float(np.mean(running[-args.log_every :])),
                        "top1": correct / seen,
                        "lr": optimizer.param_groups[0]["lr"],
                        "positions_per_s": rate,
                    }
                    logger.scalars("train", train_metrics, global_step)
                    print(
                        f"  ep {epoch} step {global_step:>8,} | loss "
                        f"{train_metrics['policy_loss']:.4f} | top1 {correct / seen:.3f} | "
                        f"{rate:,.0f} pos/s | lr {train_metrics['lr']:.2e}",
                        flush=True,
                    )

                if (
                    plateau is not None
                    and global_step % args.eval_every == 0
                    and run_eval(epoch)
                ):
                    stop = True
                    break

                if deadline and time.time() > deadline:
                    stop_reason = "time budget reached"
                    stop = True
                    break

            if stop:
                break
            if onecycle is not None:  # one-cycle validates once per epoch
                run_eval(epoch)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\ninterrupted: saving last.pt (best.pt already holds the best model)")
        base_model.save(run_dir / "last.pt", step=global_step)

    logger.close()
    hours = (time.time() - started_all) / 3600
    print(f"\nstopped: {stop_reason}")
    print(
        f"{hours:.1f} h, {global_step:,} steps. best top-1 {best_top1:.3f} "
        f"-> {run_dir / 'best.pt'}"
    )
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)

    ap.add_argument("--schedule", choices=["plateau", "onecycle"], default="plateau")
    ap.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="upper bound; plateau usually stops well before",
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="fixed time budget (switches to one-cycle)",
    )
    ap.add_argument(
        "--eval-every", type=int, default=5000, help="steps between evaluations"
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=6,
        help="evaluations without improvement before stopping",
    )
    ap.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help="evaluations without improvement before lowering the LR",
    )
    ap.add_argument("--lr-decay", type=float, default=0.3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=1000)

    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument(
        "--no-channels-last",
        dest="channels_last",
        action="store_false",
        help="disable the channels-last memory format on CUDA",
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
