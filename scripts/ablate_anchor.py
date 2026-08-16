"""Ablation: does the imitation anchor stop RL destroying the warm start?

Both arms start from the same warm-started checkpoint and differ only in
whether the auxiliary imitation loss is applied.

    python scripts/ablate_anchor.py --anchor on  --episodes 600
    python scripts/ablate_anchor.py --anchor off --episodes 600
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gmai.agent import DQNAgent  # noqa: E402
from gmai.endgames import make_sampler  # noqa: E402
from gmai.environment import ChessEnv  # noqa: E402
from gmai.metrics import RollingStats, classify_episode  # noqa: E402
from gmai.opponents import RandomOpponent  # noqa: E402
from gmai.replay_buffer import PrioritizedReplayBuffer  # noqa: E402
from gmai.rewards import POTENTIALS  # noqa: E402
from gmai.warmstart import ImitationAnchor, WarmStartData  # noqa: E402

OUT = Path("runs/pipeline")
GAMMA = 0.90


def greedy_win_rate(agent, n=120, seed=999):
    env = ChessEnv(
        opponent=RandomOpponent(seed=seed),
        position_sampler=make_sampler("KQvK", seed=seed),
        use_shaping=False,
    )
    stats = RollingStats(window=n)
    for _ in range(n):
        env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(agent.act(env.board, greedy=True))
        stats.add(classify_episode(env.board, env.agent_color, truncated))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=["on", "off"], required=True)
    ap.add_argument("--limit-terminal", choices=["on", "off"], default="on")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--epsilon", type=float, default=0.10)
    args = ap.parse_args()

    agent = DQNAgent.from_checkpoint(OUT / "base_warm.pt", device="cpu")
    agent.gamma = GAMMA
    agent.epsilon = args.epsilon
    agent.epsilon_end = 0.05
    agent.epsilon_decay = (args.epsilon - 0.05) / max(1, args.episodes * 8)

    anchor = None
    if args.anchor == "on":
        blob = np.load(OUT / "demo.npz")
        anchor = ImitationAnchor(
            agent,
            WarmStartData(blob["states"], blob["masks"], blob["targets"]),
            lr=1e-5, batch_size=128, every=1, seed=0,
        )

    env = ChessEnv(
        opponent=RandomOpponent(seed=1),
        position_sampler=make_sampler("KQvK", seed=1),
        use_shaping=True,
        potential_fn=POTENTIALS["endgame"],
        draw_reward=-1.0,
        gamma=GAMMA,
        move_limit_is_terminal=args.limit_terminal == "on",
    )
    buffer = PrioritizedReplayBuffer(30_000, seed=1)

    start = greedy_win_rate(agent)
    tag = f"anchor={args.anchor} limit={args.limit_terminal}"
    print(f"[{tag}] start: {start.summary_line()}")

    steps = 0
    curve = [{"episode": 0, **start.as_dict()}]
    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        state, info = env.reset()
        mask = info["action_mask"]
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(env.board)
            next_state, reward, terminated, truncated, info = env.step(action)
            buffer.push(state, mask, action, reward,
                        next_state, info["action_mask"], float(terminated))
            state, mask = next_state, info["action_mask"]
            steps += 1
            agent.decay_epsilon()
            if len(buffer) >= 1000 and steps % 4 == 0:
                batch = buffer.sample(128)
                _, td = agent.train_step(batch)
                buffer.update_priorities(batch.indices, td)
                if anchor is not None:
                    anchor.step()
        if ep % 200 == 0:
            stats = greedy_win_rate(agent)
            curve.append({"episode": ep, **stats.as_dict()})
            print(f"[{tag}] ep {ep} ({time.time()-t0:.0f}s): "
                  f"{stats.summary_line()}")

    name = f"anchor_{args.anchor}_limit_{args.limit_terminal}"
    agent.save(OUT / f"{name}.pt")
    (OUT / f"{name}.json").write_text(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
