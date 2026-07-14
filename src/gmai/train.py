"""Curriculum self-play training loop.

Stage 1  RandomOpponent          — learn the rules of winning.
Stage 2  GreedyMaterialOpponent  — learn not to hang pieces.
Stage 3  OpponentPool (self-play)— learn strategy vs. past selves.

Stages advance on a rolling win-rate threshold, echoing the curriculum +
domain-randomization recipe from my Formula Student racing agent (AR2) —
here the "randomised domain" is the adversary.

Usage:
    python -m gmai.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import yaml

from .agent import DQNAgent
from .environment import ChessEnv
from .opponents import GreedyMaterialOpponent, OpponentPool, RandomOpponent
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

STAGES = ("random", "greedy", "self-play")


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


def train(cfg: dict) -> Path:
    run_dir = Path(cfg["run_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent(
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
    buffer = build_buffer(cfg)
    pool = OpponentPool(capacity=cfg["curriculum"]["pool_size"], seed=cfg["seed"])

    stage = 0
    opponent = RandomOpponent(seed=cfg["seed"])
    env = ChessEnv(
        opponent=opponent,
        max_moves=cfg["max_moves"],
        gamma=cfg["gamma"],
        use_shaping=cfg["reward"]["shaping"],
        draw_reward=cfg["reward"]["draw"],
        seed=cfg["seed"],
    )

    results = deque(maxlen=cfg["curriculum"]["window"])
    log_path = run_dir / "logs.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "stage", "reward", "result", "moves", "epsilon", "loss"]
        )

        for episode in range(1, cfg["episodes"] + 1):
            state, info = env.reset()
            episode_reward, losses, done = 0.0, [], False

            while not done:
                # act() re-derives the mask from env.board (single source of truth)
                action = agent.act(env.board)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.push(
                    state, action, reward, next_state, info["action_mask"], done
                )
                state = next_state
                episode_reward += reward
                agent.decay_epsilon()

                if (
                    len(buffer) >= cfg["learn_start"]
                    and agent.train_steps * cfg["train_every"] <= episode * 400
                ):
                    batch = buffer.sample(cfg["batch_size"])
                    loss, td = agent.train_step(batch)
                    losses.append(loss)
                    if isinstance(buffer, PrioritizedReplayBuffer):
                        buffer.update_priorities(batch.indices, td)

            outcome = env.board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                result = 0.5
            else:
                result = 1.0 if outcome.winner == env.agent_color else 0.0
            results.append(result)

            writer.writerow(
                [
                    episode,
                    STAGES[stage],
                    round(episode_reward, 4),
                    result,
                    env.board.fullmove_number,
                    round(agent.epsilon, 4),
                    round(sum(losses) / len(losses), 5) if losses else "",
                ]
            )

            # ---------------- curriculum logic ----------------
            win_rate = sum(results) / len(results) if results else 0.0
            ready = (
                len(results) == results.maxlen
                and win_rate >= cfg["curriculum"]["promote_at"]
            )
            if ready and stage < 2:
                stage += 1
                results.clear()
                if stage == 1:
                    env.opponent = GreedyMaterialOpponent(seed=cfg["seed"])
                else:
                    pool.add_snapshot(agent)
                    env.opponent = pool.sample()
                print(f"[curriculum] episode {episode}: promoted to '{STAGES[stage]}'")
            elif stage == 2 and episode % cfg["curriculum"]["snapshot_every"] == 0:
                pool.add_snapshot(agent)
                env.opponent = pool.sample()

            if episode % cfg["checkpoint_every"] == 0:
                agent.save(run_dir / f"checkpoint_ep{episode}.pt")
            if episode % cfg["log_every"] == 0:
                print(
                    f"ep {episode:>6} | stage {STAGES[stage]:<9} | "
                    f"win-rate {win_rate:5.2f} | eps {agent.epsilon:.3f}"
                )

    agent.save(run_dir / "final.pt")
    print(f"Done. Artifacts in {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GMAI")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
