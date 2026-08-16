"""Curriculum training loop over forced-mate endgames.

Curriculum (scope decision, see :mod:`gmai.endgames`):

    KQ vs K  ->  KR vs K  ->  KRR vs K

promoted on **pure win-rate** measured greedily, never on the chess score.
A draw counts as a failure here: in a forced win, a stalemate or a 50-move
draw is the agent throwing the game away, and averaging it in as 0.5 is what
made 2000 episodes of noise look like progress.

The gradient-step schedule is a plain step counter:
``env_steps % train_every == 0``.

Usage:
    python -m gmai.train --config configs/endgame.yaml
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import time
from pathlib import Path

# A native crash (bad CUDA kernel, MKL fault, stack overflow) kills the process
# without a Python traceback, which looks identical to a clean exit. This makes
# the interpreter dump C and Python stacks on the way out instead.
faulthandler.enable()

import chess
import yaml

from .agent import DQNAgent
from .endgames import ENDGAME_ORDER, make_sampler
from .environment import ChessEnv
from .metrics import RollingStats, classify_episode
from .opponents import RandomOpponent
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from .rewards import POTENTIALS
from .tablebase import SOLVABLE, get_table
from .warmstart import build_dataset, dtm_quality, pretrain


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_buffer(cfg: dict):
    if cfg["replay"]["prioritized"]:
        return PrioritizedReplayBuffer(
            cfg["replay"]["capacity"],
            alpha=cfg["replay"]["alpha"],
            beta=cfg["replay"]["beta"],
            seed=cfg["seed"],
        )
    return ReplayBuffer(cfg["replay"]["capacity"], seed=cfg["seed"])


def build_agent(cfg: dict) -> DQNAgent:
    return DQNAgent(
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        epsilon_start=cfg["epsilon"]["start"],
        epsilon_end=cfg["epsilon"]["end"],
        epsilon_decay_steps=cfg["epsilon"]["decay_steps"],
        target_sync_every=cfg["target_sync_every"],
        channels=cfg["model"]["channels"],
        n_blocks=cfg["model"]["n_blocks"],
        hidden=cfg["model"]["hidden"],
        seed=cfg["seed"],
    )


def greedy_eval(agent: DQNAgent, cfg: dict, kind: str, n_games: int) -> RollingStats:
    """Pure greedy win-rate — the number the promotion gate reads."""
    env = ChessEnv(
        opponent=RandomOpponent(seed=cfg["seed"] + 999),
        position_sampler=make_sampler(kind, seed=cfg["seed"] + 999),
        gamma=cfg["gamma"],
        use_shaping=False,
        potential_fn=POTENTIALS[cfg["reward"]["potential"]],
        draw_reward=cfg["reward"]["draw"],
        move_limit_is_terminal=cfg["reward"].get("move_limit_is_terminal", True),
    )
    stats = RollingStats(window=n_games)
    for _ in range(n_games):
        _, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(env.board, greedy=True)
            _, _, terminated, truncated, info = env.step(action)
        stats.add(classify_episode(env.board, env.agent_color, truncated))
    return stats


def run_warmstart(agent: DQNAgent, cfg: dict, kind: str) -> dict | None:
    """Supervised pre-training against the exact solver, if enabled."""
    wcfg = cfg.get("warmstart", {})
    if not wcfg.get("enabled") or kind not in SOLVABLE:
        return None

    table = get_table(kind, cache_dir=wcfg.get("cache_dir", "tablebases"), verbose=True)
    if table is None:
        return None

    print(f"[warmstart] building {wcfg['positions']} labelled {kind} positions...")
    data = build_dataset(kind, table, wcfg["positions"], seed=cfg["seed"])
    pretrain(
        agent, data,
        epochs=wcfg["epochs"], batch_size=wcfg["batch_size"], lr=wcfg["lr"],
    )
    quality = dtm_quality(agent, kind, table, n_positions=200, seed=cfg["seed"])
    print(f"[warmstart] DTM quality after pre-training: {quality}")
    return quality


def train(cfg: dict) -> Path:
    run_dir = Path(cfg["run_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    agent = build_agent(cfg)
    buffer = build_buffer(cfg)
    potential_fn = POTENTIALS[cfg["reward"]["potential"]]

    stages: list[str] = cfg["curriculum"]["stages"]
    stage = 0
    env = ChessEnv(
        opponent=RandomOpponent(seed=cfg["seed"]),
        position_sampler=make_sampler(stages[stage], seed=cfg["seed"]),
        gamma=cfg["gamma"],
        use_shaping=cfg["reward"]["shaping"],
        potential_fn=potential_fn,
        draw_reward=cfg["reward"]["draw"],
        move_limit_is_terminal=cfg["reward"].get("move_limit_is_terminal", True),
        seed=cfg["seed"],
    )

    warm = run_warmstart(agent, cfg, stages[stage])
    if warm is not None:
        agent.epsilon = cfg["warmstart"]["epsilon_after"]
        agent.save(run_dir / "warmstart.pt")

    stats = RollingStats(window=cfg["curriculum"]["window"])
    env_steps = 0
    history: list[dict] = []

    log_path = run_dir / "logs.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode", "stage", "reward", "win", "draw", "loss",
                "plies", "termination", "epsilon", "loss_td",
                "roll_win_rate", "roll_draw_rate", "roll_conv_fail",
            ]
        )

        for episode in range(1, cfg["episodes"] + 1):
            state, info = env.reset()
            mask = info["action_mask"]
            episode_reward, losses = 0.0, []
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.act(env.board)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_mask = info["action_mask"]

                # store `terminated`, not `terminated or truncated`
                buffer.push(
                    state, mask, action, reward,
                    next_state, next_mask, float(terminated),
                )
                state, mask = next_state, next_mask
                episode_reward += reward
                env_steps += 1
                agent.decay_epsilon()

                # gradient-step schedule
                if (
                    len(buffer) >= cfg["learn_start"]
                    and env_steps % cfg["train_every"] == 0
                ):
                    batch = buffer.sample(cfg["batch_size"])
                    loss, td = agent.train_step(batch)
                    losses.append(loss)
                    if isinstance(buffer, PrioritizedReplayBuffer):
                        buffer.update_priorities(batch.indices, td)

            result = classify_episode(env.board, env.agent_color, truncated)
            stats.add(result)

            writer.writerow([
                episode, stages[stage], round(episode_reward, 4),
                int(result.win), int(result.draw), int(result.loss),
                result.plies, result.termination, round(agent.epsilon, 4),
                round(sum(losses) / len(losses), 5) if losses else "",
                round(stats.win_rate, 4), round(stats.draw_rate, 4),
                round(stats.conversion_failure_rate, 4),
            ])

            if episode % cfg["log_every"] == 0:
                print(
                    f"ep {episode:>6} | {stages[stage]:<6} | "
                    f"{stats.summary_line()} | eps {agent.epsilon:.3f}"
                )

            # ------------------- promotion gate (greedy) -------------------
            gate_due = (
                stats.is_full
                and episode % cfg["curriculum"]["eval_every"] == 0
                and stats.win_rate >= cfg["curriculum"]["screen_at"]
            )
            if gate_due:
                ev = greedy_eval(agent, cfg, stages[stage], cfg["curriculum"]["eval_games"])
                record = {"episode": episode, "stage": stages[stage], **ev.as_dict()}
                history.append(record)
                print(f"  [eval] {stages[stage]} greedy -> {ev.summary_line()}")

                threshold = cfg["curriculum"]["promote_at"][stages[stage]]
                if ev.win_rate >= threshold:
                    agent.save(run_dir / f"passed_{stages[stage]}.pt")
                    print(
                        f"  [PASS] {stages[stage]} win-rate {ev.win_rate:.3f} "
                        f">= {threshold}"
                    )
                    if stage + 1 < len(stages):
                        stage += 1
                        stats.clear()
                        env.position_sampler = make_sampler(
                            stages[stage], seed=cfg["seed"] + stage
                        )
                        print(f"  [curriculum] -> {stages[stage]}")
                    else:
                        print("  [curriculum] all stages passed")
                        break

            if episode % cfg["checkpoint_every"] == 0:
                agent.save(run_dir / f"checkpoint_ep{episode}.pt")

    agent.save(run_dir / "final.pt")

    # Final report on every stage, pass or fail — honest by construction.
    final = {}
    for kind in stages:
        ev = greedy_eval(agent, cfg, kind, cfg["curriculum"]["eval_games"])
        entry = ev.as_dict()
        table = get_table(kind, verbose=False) if kind in SOLVABLE else None
        if table is not None:
            entry["dtm_quality"] = dtm_quality(agent, kind, table, 300, cfg["seed"])
        final[kind] = entry
        print(f"[final] {kind}: {ev.summary_line()}")
        if table is not None:
            print(f"         dtm: {entry['dtm_quality']}")
    with open(run_dir / "report.json", "w") as f:
        json.dump({"eval_history": history, "final": final}, f, indent=2)

    print(f"Done. Artifacts in {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GMAI")
    parser.add_argument("--config", default="configs/endgame.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.episodes:
        cfg["episodes"] = args.episodes
    train(cfg)


if __name__ == "__main__":
    main()
