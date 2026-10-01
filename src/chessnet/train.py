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

Continuing from a checkpoint:

``--resume runs/<run>/last.pt``
    Picks up exactly where a run stopped (weights, optimizer, learning-rate
    schedule, step and position within the epoch), in the same run directory.
    ``last.pt`` is rewritten atomically at every evaluation, so a crash costs
    at most ``--eval-every`` steps.
``--init-from runs/<run>/best.pt``
    Starts a new run from those weights, with a fresh optimizer, a short
    warm-up and ``--lr``. Use it to continue from a checkpoint that has no
    optimizer state, or to fine-tune at a lower learning rate.

``--min-delta`` sets how much validation top-1 must improve to count as
progress for the learning-rate schedule and early stopping, so tiny gains do
not keep the learning rate high indefinitely.

Progress is written to TensorBoard when it is installed::

    tensorboard --logdir runs

The metric to watch is top-1 accuracy on held-out positions. Maia reports
roughly 50% move-matching. Below ~35% means something is wrong; well above
~55% is worth checking for leakage between the splits.
"""

from __future__ import annotations

import argparse
import json
import os
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
    if args.resume and args.init_from:
        raise SystemExit("use either --resume or --init-from, not both")
    if (args.resume or args.init_from) and args.max_hours:
        raise SystemExit(
            "--max-hours measures speed by training throwaway steps and then resets "
            "the weights, so it cannot be combined with --resume or --init-from"
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    source = args.resume or args.init_from
    blob = torch.load(source, map_location="cpu", weights_only=False) if source else None
    if args.resume and "optimizer" not in blob:
        raise SystemExit(
            f"{args.resume} has no optimizer state (saved by an older version). "
            "Start a new run from its weights with --init-from instead."
        )

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
    config = (
        blob["config"] if blob else {"channels": args.channels, "blocks": args.blocks}
    )
    model = ChessNet(**config)
    if blob:
        model.load_state_dict(blob["state_dict"])
    model = model.to(device, memory_format=memory_format)
    print(
        f"model : {config['channels']}ch x {config['blocks']} blocks, "
        f"{model.n_parameters / 1e6:.1f}M parameters on {device}"
    )
    base_model = model
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.resume:
        optimizer.load_state_dict(blob["optimizer"])
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
        if args.resume and blob.get("scheduler"):
            onecycle.load_state_dict(blob["scheduler"])
    else:
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_decay,
            patience=args.lr_patience,
            threshold=args.min_delta,
            threshold_mode="abs",
        )
        if args.resume and blob.get("scheduler"):
            plateau.load_state_dict(blob["scheduler"])
        print(
            f"schedule: plateau | eval every {args.eval_every:,} steps "
            f"({args.eval_every * args.batch_size / 1e6:.1f}M positions) | "
            f"progress means +{args.min_delta:.3f} top-1 | "
            f"stop after {args.patience} evals without it"
        )

    # ------------------------------------------------------------ run state
    if args.resume:
        run_dir = Path(args.resume).resolve().parent
        history_path = run_dir / "history.json"
        history: list[dict] = (
            json.loads(history_path.read_text()) if history_path.exists() else []
        )
    else:
        run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(vars(args), indent=2, default=str)
        )
        history = []
    logger = _Logger(run_dir / "tb")
    print(f"run   : {run_dir}")

    global_step = blob["step"] if args.resume else 0
    start_epoch = blob.get("epoch", 1) if args.resume else 1
    start_batch = blob.get("epoch_step", 0) if args.resume else 0
    # -1 so the first evaluation always writes best.pt, even at 0% accuracy.
    best_top1 = blob.get("best_top1", -1.0) if args.resume else -1.0
    ref_top1 = blob.get("ref_top1", best_top1) if args.resume else -1.0
    evals_since_best = blob.get("evals_since_best", 0) if args.resume else 0
    if args.resume:
        print(
            f"resumed from {args.resume}: step {global_step:,}, epoch {start_epoch}, "
            f"batch {start_batch:,} | best top-1 {best_top1:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )
    if args.init_from:
        # Also checks the loaded weights: a damaged file shows up here.
        start = evaluate(
            model, data, device, args.batch_size, memory_format=memory_format
        )
        best_top1 = ref_top1 = start["top1"]
        base_model.save(run_dir / "best.pt", step=0, metrics=start)
        print(
            f"weights from {args.init_from}: val top1 {start['top1']:.4f} | "
            f"top5 {start['top5']:.4f} (starting point; best.pt holds them until beaten)"
        )
    print()

    started_all = time.time()
    deadline = started_all + args.max_hours * 3600 if args.max_hours else None
    stop_reason = "epochs exhausted"

    def save_last(epoch: int, epoch_step: int, metrics: dict | None = None) -> None:
        """Everything needed to resume, written atomically."""
        extra = {
            "step": global_step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            "best_top1": best_top1,
            "ref_top1": ref_top1,
            "evals_since_best": evals_since_best,
            "optimizer": optimizer.state_dict(),
            "scheduler": (plateau or onecycle).state_dict(),
        }
        if metrics is not None:
            extra["metrics"] = metrics
        tmp = run_dir / "last.pt.tmp"
        base_model.save(tmp, **extra)
        os.replace(tmp, run_dir / "last.pt")

    def run_eval(epoch: int, epoch_step: int, end_of_epoch: bool = False) -> bool:
        """Validate, update the schedule, checkpoint. Returns True to stop."""
        nonlocal best_top1, ref_top1, evals_since_best, stop_reason
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
        if improved:
            best_top1 = metrics["top1"]
            base_model.save(run_dir / "best.pt", step=global_step, metrics=metrics)

        stop = False
        if global_step >= args.warmup_steps:
            # Stalls during warm-up say nothing: the learning rate is still
            # climbing. Only count them once the schedule is in charge.
            if metrics["top1"] >= ref_top1 + args.min_delta:
                ref_top1, evals_since_best = metrics["top1"], 0
            else:
                evals_since_best += 1
            if plateau is not None:
                plateau.step(metrics["top1"])
                if evals_since_best >= args.patience:
                    stop_reason = (
                        f"no progress of +{args.min_delta} in {args.patience} evaluations"
                    )
                    stop = True
                elif optimizer.param_groups[0]["lr"] < args.min_lr:
                    stop_reason = "learning rate fell below --min-lr"
                    stop = True

        if end_of_epoch:
            save_last(epoch + 1, 0, metrics)
        else:
            save_last(epoch, epoch_step, metrics)
        return stop

    epoch, epoch_step = start_epoch, start_batch
    try:
        stop = False
        for epoch in range(start_epoch, epochs + 1):
            epoch_step = start_batch if epoch == start_epoch else 0
            running, seen, correct = [], 0, 0
            t_epoch = time.time()
            for planes, actions, results in data.iter_batches(
                args.batch_size,
                train=True,
                seed=args.seed + epoch,
                skip_batches=epoch_step,
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
                epoch_step += 1

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
                    gpu = ""
                    if device.type == "cuda":
                        train_metrics["gpu_mem_gb"] = (
                            torch.cuda.max_memory_allocated() / 1e9
                        )
                        gpu = f" | gpu {train_metrics['gpu_mem_gb']:.1f} GB"
                    logger.scalars("train", train_metrics, global_step)
                    print(
                        f"  ep {epoch} step {global_step:>8,} | loss "
                        f"{train_metrics['policy_loss']:.4f} | top1 {correct / seen:.3f} | "
                        f"{rate:,.0f} pos/s | lr {train_metrics['lr']:.2e}{gpu}",
                        flush=True,
                    )

                if (
                    plateau is not None
                    and global_step % args.eval_every == 0
                    and run_eval(epoch, epoch_step)
                ):
                    stop = True
                    break

                if args.max_steps and global_step >= args.max_steps:
                    stop_reason = "reached --max-steps"
                    save_last(epoch, epoch_step)
                    stop = True
                    break

                if deadline and time.time() > deadline:
                    stop_reason = "time budget reached"
                    save_last(epoch, epoch_step)
                    stop = True
                    break

            if stop:
                break
            if onecycle is not None:  # one-cycle validates once per epoch
                run_eval(epoch, epoch_step, end_of_epoch=True)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        save_last(epoch, epoch_step)
        print(
            f"\ninterrupted: state saved. Continue with\n"
            f"  python -m chessnet.train --data {args.data} --resume {run_dir / 'last.pt'}"
        )

    logger.close()
    hours = (time.time() - started_all) / 3600
    print(f"\nstopped: {stop_reason}")
    best = f"best top-1 {best_top1:.4f}" if best_top1 >= 0 else "no evaluation yet"
    print(f"{hours:.1f} h, {global_step:,} steps. {best} -> {run_dir / 'best.pt'}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)

    ap.add_argument("--resume", default=None, help="continue a run from its last.pt")
    ap.add_argument(
        "--init-from", default=None, help="start a new run from these weights"
    )

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
        "--max-steps", type=int, default=None, help="stop after this many steps"
    )
    ap.add_argument(
        "--eval-every", type=int, default=5000, help="steps between evaluations"
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="top-1 gain that counts as progress (0.001 = 0.1 points)",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=6,
        help="evaluations without progress before stopping",
    )
    ap.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help="evaluations without progress before lowering the LR",
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
    return ap


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
