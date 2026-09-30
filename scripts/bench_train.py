"""Measure training throughput on this machine, one setting at a time.

Separates the two possible bottlenecks so the fix follows the evidence:

* **GPU compute** — synthetic batches already on the device, so nothing but
  forward/backward is timed, under each combination of optimisations;
* **data pipeline** — real shards expanded on the CPU, if ``--data`` is given.

If the data pipeline is slower than the best GPU number, the CPU is the
bottleneck; otherwise it is the model.

    python scripts/bench_train.py
    python scripts/bench_train.py --data data/train --channels 192 --blocks 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from chessnet.encoding import N_ACTIONS, N_PLANES
from chessnet.model import ChessNet


def bench_gpu(channels, blocks, batch, channels_last, benchmark, seconds=8.0) -> float:
    torch.backends.cudnn.benchmark = benchmark
    device = torch.device("cuda")
    model = ChessNet(channels=channels, blocks=blocks).to(device)
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    model = model.to(memory_format=fmt)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.rand(batch, N_PLANES, 8, 8, device=device).to(memory_format=fmt)
    y = torch.randint(0, N_ACTIONS, (batch,), device=device)
    z = torch.rand(batch, device=device) * 2 - 1

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, value = model(x)
            loss = F.cross_entropy(logits, y) + 0.3 * F.mse_loss(value, z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(15):  # warm-up, includes cuDNN autotuning
        step()
    torch.cuda.synchronize()
    n, start = 0, time.time()
    while time.time() - start < seconds:
        step()
        n += 1
    torch.cuda.synchronize()
    return n * batch / (time.time() - start)


def bench_data(data_dir, batch, seconds=8.0) -> float:
    from chessnet.dataset import ShardDataset

    data = ShardDataset(data_dir, val_fraction=0.0, limit=4_000_000)
    n, start = 0, time.time()
    for planes, actions, _results in data.iter_batches(batch, seed=0):
        planes.pin_memory()
        n += len(actions)
        if time.time() - start > seconds:
            break
    return n / (time.time() - start)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    print(
        f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
        f"{args.channels}ch x {args.blocks} blocks\n"
    )

    configs = [
        ("baseline (what is running now)", 1024, False, False),
        ("+ cudnn.benchmark", 1024, False, True),
        ("+ channels_last", 1024, True, True),
        ("+ batch 2048", 2048, True, True),
        ("+ batch 4096", 4096, True, True),
    ]
    best = 0.0
    for label, batch, cl, bm in configs:
        try:
            rate = bench_gpu(args.channels, args.blocks, batch, cl, bm)
            mem = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            best = max(best, rate)
            print(
                f"  {label:<32} {rate:>9,.0f} pos/s   "
                f"{mem:4.1f} GB used / {reserved:4.1f} GB reserved"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:<32}    out of memory")
            torch.cuda.empty_cache()

    if args.data:
        rate = bench_data(args.data, 2048)
        print(f"\n  data pipeline (one CPU thread)   {rate:>9,.0f} pos/s")
        verdict = "CPU" if rate < best else "GPU"
        print(f"\n  bottleneck: {verdict}")


if __name__ == "__main__":
    main()
