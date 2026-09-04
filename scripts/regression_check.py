"""Performance regression gate.

Unit tests catch code that throws. They do not catch code that runs fine and
quietly plays worse — which, on this project, is the failure mode that actually
happened: a dueling baseline averaged over illegal actions produced no errors
for thousands of episodes.

This evaluates a fixed checkpoint on a fixed set of positions with a fixed seed
and fails if the result drops below a recorded baseline. Run in CI on every PR.

    python scripts/regression_check.py --checkpoint models/final.pt
    python scripts/regression_check.py --checkpoint models/final.pt --update

`--update` rewrites the baseline; do that deliberately, in its own commit, when
a change is meant to move the numbers.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASELINE_PATH = Path(__file__).resolve().parents[1] / "tests" / "baseline.json"

# How far below the recorded baseline a run may fall before this fails.
# Evaluation is seeded and deterministic, so this is slack for library and
# hardware differences, not for the model getting worse.
TOLERANCE = 0.05


def evaluate(checkpoint: Path, kind: str, games: int, seed: int) -> dict:
    from gmai.agent import DQNAgent
    from gmai.encoding import action_to_move
    from gmai.evaluate import evaluate_kind
    from gmai.opponents import RandomOpponent

    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")
    agent.epsilon = 0.0

    def policy(board):
        return action_to_move(agent.act(board, greedy=True), board)

    stats = evaluate_kind(policy, RandomOpponent(seed=seed), kind, games, seed)
    result = {
        "win_rate": round(stats.win_rate, 4),
        "draw_rate": round(stats.draw_rate, 4),
        "loss_rate": round(stats.loss_rate, 4),
        "mean_plies": round(stats.mean_plies, 2),
    }

    # DTM quality where the solver covers the endgame: a far more sensitive
    # signal than win-rate, because it scores every move rather than the
    # outcome of a whole game.
    from gmai.tablebase import SOLVABLE, get_table

    if kind in SOLVABLE:
        table = get_table(kind, verbose=False)
        if table is not None:
            from gmai.warmstart import dtm_quality

            q = dtm_quality(agent, kind, table, n_positions=150, seed=seed)
            result["optimal_rate"] = q["optimal_rate"]
            result["throw_away_rate"] = q["throw_away_rate"]
    return result


def random_baseline(kind: str, games: int, seed: int) -> dict:
    """What chance alone scores, so the gate has a floor to sit above."""
    from gmai.evaluate import evaluate_kind
    from gmai.opponents import RandomOpponent

    rng = random.Random(seed + 7)

    def policy(board):
        return rng.choice(list(board.legal_moves))

    stats = evaluate_kind(policy, RandomOpponent(seed=seed), kind, games, seed)
    return {"win_rate": round(stats.win_rate, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Performance regression gate")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--kind", default="KQvK")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--update", action="store_true", help="rewrite the baseline")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    print(
        f"evaluating {args.checkpoint} on {args.games} {args.kind} games "
        f"(seed {args.seed})..."
    )
    current = evaluate(args.checkpoint, args.kind, args.games, args.seed)
    floor = random_baseline(args.kind, args.games, args.seed)

    print(f"\n  {'metric':<20} {'current':>9} {'random':>9}")
    print("  " + "-" * 41)
    for key, value in current.items():
        ref = floor.get(key, "")
        print(f"  {key:<20} {value:>9} {ref if ref == '' else f'{ref:>9}'}")

    if args.update:
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "kind": args.kind,
                    "games": args.games,
                    "seed": args.seed,
                    "metrics": current,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nbaseline written to {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.exists():
        print(f"\nno baseline at {BASELINE_PATH} — create one with --update")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text())
    if (baseline["kind"], baseline["games"], baseline["seed"]) != (
        args.kind,
        args.games,
        args.seed,
    ):
        print("\nbaseline was recorded with different settings; not comparing")
        return 0

    failures = []
    for metric in ("win_rate", "optimal_rate"):
        if metric not in baseline["metrics"] or metric not in current:
            continue
        before, after = baseline["metrics"][metric], current[metric]
        if after < before - TOLERANCE:
            failures.append(f"{metric}: {before} -> {after}")
    # Throwing away won positions must not get worse either.
    if "throw_away_rate" in baseline["metrics"] and "throw_away_rate" in current:
        before, after = baseline["metrics"]["throw_away_rate"], current["throw_away_rate"]
        if after > before + TOLERANCE:
            failures.append(f"throw_away_rate: {before} -> {after}")

    if failures:
        print("\nREGRESSION:")
        for line in failures:
            print(f"  {line}")
        print(
            f"\n(tolerance {TOLERANCE}; if this change is intended, re-run "
            "with --update in its own commit)"
        )
        return 1

    print("\nno regression against the recorded baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
