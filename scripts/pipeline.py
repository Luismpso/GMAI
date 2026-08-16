"""Training pipeline, split into resumable stages.

Each stage writes a checkpoint, so a long run can be stopped and picked up
later without repeating the expensive parts (the solver and the supervised
warm start are cached):

    python scripts/pipeline.py warmstart --kind KQvK
    python scripts/pipeline.py rl        --kind KQvK
    python scripts/pipeline.py report    --kind KQvK

`gmai.train` runs the same thing end to end; this exists for when you want to
iterate on one stage at a time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gmai.agent import DQNAgent  # noqa: E402
from gmai.endgames import make_sampler  # noqa: E402
from gmai.environment import ChessEnv  # noqa: E402
from gmai.metrics import RollingStats, classify_episode  # noqa: E402
from gmai.opponents import RandomOpponent  # noqa: E402
from gmai.replay_buffer import PrioritizedReplayBuffer  # noqa: E402
from gmai.rewards import POTENTIALS  # noqa: E402
from gmai.tablebase import get_table  # noqa: E402
from gmai.warmstart import build_dataset, dtm_quality, pretrain  # noqa: E402

OUT = Path("runs/pipeline")
ARCH = {"channels": 48, "n_blocks": 3, "hidden": 256}


def greedy_eval(agent, kind, n_games=200, seed=999):
    env = ChessEnv(
        opponent=RandomOpponent(seed=seed),
        position_sampler=make_sampler(kind, seed=seed),
        use_shaping=False,
    )
    stats = RollingStats(window=n_games)
    for _ in range(n_games):
        env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(agent.act(env.board, greedy=True))
        stats.add(classify_episode(env.board, env.agent_color, truncated))
    return stats


def phase_warmstart(kind, positions, epochs):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    agent = DQNAgent(**ARCH, device="cpu", seed=0, lr=3e-4)
    table = get_table(kind, verbose=True)

    t0 = time.time()
    data = build_dataset(kind, table, positions, seed=0)
    print(f"[dataset] {len(data.targets)} positions in {time.time() - t0:.0f}s")

    pretrain(agent, data, epochs=epochs, batch_size=256, lr=1e-3)
    agent.save(OUT / f"{kind}_warm.pt")

    stats = greedy_eval(agent, kind)
    print(f"[warmstart] {kind}: {stats.summary_line()}")
    print(f"[warmstart] dtm: {dtm_quality(agent, kind, table, 300, seed=1)}")
    return stats


def phase_rl(kind, episodes, epsilon):
    agent = DQNAgent.from_checkpoint(OUT / f"{kind}_warm.pt", device="cpu")
    agent.epsilon = epsilon
    agent.epsilon_end = 0.05
    agent.epsilon_decay = (epsilon - 0.05) / max(1, episodes * 8)

    env = ChessEnv(
        opponent=RandomOpponent(seed=1),
        position_sampler=make_sampler(kind, seed=1),
        use_shaping=True,
        potential_fn=POTENTIALS["endgame"],
        draw_reward=-1.0,
        gamma=0.99,
    )
    buffer = PrioritizedReplayBuffer(50_000, seed=1)
    stats = RollingStats(window=200)
    env_steps = 0

    for ep in range(1, episodes + 1):
        state, info = env.reset()
        mask = info["action_mask"]
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(env.board)
            next_state, reward, terminated, truncated, info = env.step(action)
            buffer.push(state, mask, action, reward,
                        next_state, info["action_mask"], float(terminated))
            state, mask = next_state, info["action_mask"]
            env_steps += 1
            agent.decay_epsilon()
            if len(buffer) >= 1000 and env_steps % 4 == 0:
                batch = buffer.sample(128)
                _, td = agent.train_step(batch)
                buffer.update_priorities(batch.indices, td)
        stats.add(classify_episode(env.board, env.agent_color, truncated))
        if ep % 200 == 0:
            print(f"ep {ep:>5} | {stats.summary_line()} | eps {agent.epsilon:.3f}")

    agent.save(OUT / f"{kind}_rl.pt")
    ev = greedy_eval(agent, kind)
    print(f"[rl] {kind}: {ev.summary_line()}")
    return ev


def phase_report(kind):
    report = {}
    table = get_table(kind, verbose=False)
    for tag in ("warm", "rl"):
        path = OUT / f"{kind}_{tag}.pt"
        if not path.exists():
            continue
        agent = DQNAgent.from_checkpoint(path, device="cpu")
        agent.epsilon = 0.0
        stats = greedy_eval(agent, kind, n_games=300)
        report[tag] = stats.as_dict()
        report[tag]["dtm_quality"] = dtm_quality(agent, kind, table, 300, seed=1)
        print(f"[{tag}] {stats.summary_line()}")
    (OUT / f"{kind}_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["warmstart", "rl", "report"])
    ap.add_argument("--kind", default="KQvK")
    ap.add_argument("--positions", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--epsilon", type=float, default=0.25)
    args = ap.parse_args()

    if args.phase == "warmstart":
        phase_warmstart(args.kind, args.positions, args.epochs)
    elif args.phase == "rl":
        phase_rl(args.kind, args.episodes, args.epsilon)
    else:
        phase_report(args.kind)
