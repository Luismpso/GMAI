"""Reproducible evaluation report.

Regenerates every published number from a single command, so the README, the
model card and the checkpoint can never quietly disagree with each other.

    python scripts/evaluate_all.py --checkpoint models/final.pt
    python scripts/evaluate_all.py --checkpoint models/final.pt --inject

`--inject` rewrites the block between the RESULTS markers in README.md and
docs/MODEL_CARD.md. Everything is seeded, so two runs on the same checkpoint
produce identical output.
"""

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BEGIN = "<!-- BEGIN:RESULTS -->"
END = "<!-- END:RESULTS -->"


def evaluate(checkpoint: Path, kinds: list[str], games: int, seed: int) -> dict:
    from gmai.agent import DQNAgent
    from gmai.encoding import action_to_move
    from gmai.evaluate import StubbornKingOpponent, evaluate_kind
    from gmai.opponents import RandomOpponent
    from gmai.tablebase import SOLVABLE, get_table

    agent = DQNAgent.from_checkpoint(checkpoint, device="cpu")
    agent.epsilon = 0.0

    def agent_policy(board):
        return action_to_move(agent.act(board, greedy=True), board)

    report: dict = {
        "checkpoint": checkpoint.name,
        "date": date.today().isoformat(),
        "games_per_cell": games,
        "seed": seed,
        "endgames": {},
    }

    for kind in kinds:
        print(f"  {kind}...", flush=True)
        cell: dict = {}
        for opponent in (RandomOpponent(seed=seed), StubbornKingOpponent(seed=seed)):
            rng = random.Random(seed + 7)

            def random_policy(board, _rng=rng):
                return _rng.choice(list(board.legal_moves))

            cell[opponent.name] = {
                "agent": evaluate_kind(
                    agent_policy, opponent, kind, games, seed
                ).as_dict(),
                "random_baseline": evaluate_kind(
                    random_policy, opponent, kind, games, seed
                ).as_dict(),
            }

        if kind in SOLVABLE:
            table = get_table(kind, verbose=False)
            if table is not None:
                from gmai.warmstart import dtm_quality

                cell["dtm_quality"] = dtm_quality(agent, kind, table, 300, seed)

        report["endgames"][kind] = cell
    return report


def to_markdown(report: dict) -> str:
    lines = [
        f"Checkpoint `{report['checkpoint']}`, "
        f"{report['games_per_cell']} games per cell, seed {report['seed']}. "
        f"Generated {report['date']} by `scripts/evaluate_all.py`.",
        "",
        "| Endgame | Defender | Agent | W | D | L | Win rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, defenders in report["endgames"].items():
        for defender, entries in defenders.items():
            if defender == "dtm_quality":
                continue
            for who, label in (("agent", "model"), ("random_baseline", "random")):
                r = entries[who]
                bold = "**" if who == "agent" else ""
                lines.append(
                    f"| {kind} | {defender} | {label} | {r['wins']} | {r['draws']} | "
                    f"{r['losses']} | {bold}{r['win_rate']:.3f}{bold} |"
                )

    dtm_rows = [
        (kind, cell["dtm_quality"])
        for kind, cell in report["endgames"].items()
        if "dtm_quality" in cell
    ]
    if dtm_rows:
        lines += [
            "",
            "Move quality against the solver's ground truth:",
            "",
            "| Endgame | Optimal moves | Suboptimal | Turns a win into a draw |",
            "|---|---|---|---|",
        ]
        for kind, q in dtm_rows:
            lines.append(
                f"| {kind} | {q['optimal_rate']:.1%} | {q['suboptimal_rate']:.1%} "
                f"| {q['throw_away_rate']:.1%} |"
            )
    return "\n".join(lines)


def inject(path: Path, block: str) -> bool:
    """Replace the content between the RESULTS markers. Returns True if changed."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"  {path.name}: no RESULTS markers, skipped")
        return False
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = f"{head}{BEGIN}\n{block}\n{END}{tail}"
    if updated == text:
        print(f"  {path.name}: already up to date")
        return False
    path.write_text(updated, encoding="utf-8")
    print(f"  {path.name}: updated")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate the published results")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "final.pt")
    ap.add_argument("--kinds", nargs="+", default=["KQvK"])
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=ROOT / "docs" / "results.json")
    ap.add_argument(
        "--inject",
        action="store_true",
        help="write into README.md and docs/MODEL_CARD.md",
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    print(
        f"evaluating {args.checkpoint.name} "
        f"({args.games} games per cell, seed {args.seed})"
    )
    report = evaluate(args.checkpoint, args.kinds, args.games, args.seed)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")

    block = to_markdown(report)
    print(f"\n{block}\n")
    print(f"json -> {args.json_out.relative_to(ROOT)}")

    if args.inject:
        print("\ninjecting:")
        for target in (ROOT / "README.md", ROOT / "docs" / "MODEL_CARD.md"):
            inject(target, block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
