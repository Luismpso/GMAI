"""Diagnostic runner: exercise each training stage in isolation.

When `gmai.train` dies without a traceback, the usual causes are memory
pressure or a stage that silently produced nothing. This runs the stages one
at a time, reports resident memory after each, and fails loudly.

    python scripts/doctor.py
    python scripts/doctor.py --positions 5000 --channels 32
"""

import argparse
import faulthandler
import platform
import sys
import traceback
from pathlib import Path

faulthandler.enable()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _rss_mb() -> float:
    """Resident memory in MB, without requiring psutil."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        try:  # Linux fallback
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * 4096 / 1024 / 1024
        except OSError:
            return float("nan")


def _total_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        return float("nan")


def step(label: str):
    print(f"  {label:<44} {_rss_mb():>8.0f} MB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose a GMAI training run")
    ap.add_argument("--kind", default="KQvK")
    ap.add_argument("--positions", type=int, default=20000)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=100000)
    args = ap.parse_args()

    print(f"python   {platform.python_version()} on {platform.system()}")
    print(f"total RAM {_total_ram_gb():.1f} GB\n")
    print(f"  {'stage':<44} {'resident':>8}")
    print("  " + "-" * 55)
    step("start")

    try:
        import numpy as np  # noqa: F401
        import torch

        step(f"import torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"      CUDA: {torch.cuda.get_device_name(0)}")

        from gmai.agent import DQNAgent
        from gmai.replay_buffer import PrioritizedReplayBuffer
        from gmai.tablebase import get_table
        from gmai.warmstart import build_dataset, pretrain

        step("import gmai")

        agent = DQNAgent(
            channels=args.channels, n_blocks=args.n_blocks, hidden=args.hidden
        )
        step(f"agent ({args.channels}ch x {args.n_blocks}, hidden {args.hidden})")
        print(f"      device: {agent.device}")

        PrioritizedReplayBuffer(args.buffer)
        step(f"replay buffer (capacity {args.buffer:,}, empty)")
        # ~10.84 KB per transition, measured with bit-packed masks:
        # 2 x 4608 B of board planes plus 2 x 512 B of packed mask, plus
        # Python object overhead.
        full_gb = args.buffer * 10.84 / 1024 / 1024
        print(f"      when full this needs ~{full_gb:.2f} GB more")

        table = get_table(args.kind, verbose=True)
        if table is None:
            print(f"\n  !! no solver available for {args.kind}")
            return 1
        step(f"tablebase {args.kind}")

        data = build_dataset(args.kind, table, args.positions, seed=0)
        step(f"dataset ({len(data.targets):,} positions)")

        pretrain(agent, data, epochs=1, batch_size=256, verbose=False)
        step("one pre-training epoch")

        print("\n  all stages completed.")
        peak_gb = _rss_mb() / 1024 + full_gb
        total = _total_ram_gb()
        print(f"  projected peak with a full buffer: ~{peak_gb:.1f} GB")

        if total != total:  # NaN: psutil unavailable
            print("  (install psutil for a RAM headroom check)")
        elif peak_gb > total * 0.8:
            print("  !! close to this machine's RAM — lower replay.capacity "
                  "or warmstart.positions.")
        else:
            print(f"  that is {peak_gb / total:.0%} of {total:.0f} GB: comfortable.")
            print("  If a run still dies with no traceback, memory is NOT the")
            print("  cause. Re-run with faulthandler to catch a native crash:")
            print("     python -X faulthandler -m gmai.train "
                  "--config configs/endgame.yaml")
        return 0

    except MemoryError:
        print("\n  !! MemoryError — lower replay.capacity or warmstart.positions")
        return 1
    except Exception:
        print("\n  !! failed:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
