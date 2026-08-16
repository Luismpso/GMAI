# ♟️ GMAI — Grand Master AI

**A search-free Deep RL chess agent for forced-mate endgames**, with an exact
solver for its own domain, honest W/D/L evaluation, and a UCI adapter.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-env-2e7d32?style=flat-square)
![Tests](https://img.shields.io/badge/pytest-161%20passed-1b5e20?style=flat-square)
![UCI](https://img.shields.io/badge/UCI-compatible-2e7d32?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-66bb6a?style=flat-square)

---

## 🎯 Scope

GMAI plays **forced-mate endgames**: KQ vs K, KR vs K, KRR vs K.

**Full chess from the opening is out of scope.** That is a deliberate narrowing,
not an unfinished feature. A DQN without search cannot solve chess from move 1 —
the reward is too sparse and the horizon too long, which is why AlphaZero pairs
its network with tree search. Endgames are the regime where the method works on
its own terms: horizon under 20 moves, mate reachable by exploration, and a
random-play win-rate of exactly **0.000**, so every point of progress is real
signal.

Reasoning in [`docs/DESIGN.md`](docs/DESIGN.md).

---

## 📊 Current results

Supervised warm start against the exact solver (48ch × 3 blocks, 20k labelled
positions, ~9 min CPU).

| KQ vs K | | W | D | L | win-rate |
|---|---|---|---|---|---|
| vs. random defender | **agent** | 84 | 66 | 0 | **0.560** |
| | random baseline | 3 | 147 | 0 | 0.020 |
| vs. stubborn defender | **agent** | 74 | 76 | 0 | **0.493** |
| | random baseline | 0 | 150 | 0 | 0.000 |

Top-1 agreement with provably optimal play: **95.2%** ·
optimal moves **69.3%** · moves that throw the win away **2.7%**

**Zero losses in 300 games.** The agent does not blunder into defeat — it fails
to *convert*: 60 threefold repetitions, 45 lost queens, 31 stalemates. That is
the expected gap between imitating individual moves and executing a ten-move
plan: a 2.7% per-move error rate compounds over ~9 agent moves into roughly a
22% chance of throwing away any given game.

Target is 90% on KQ vs K. Not there yet.

### ⚠️ Open problem: RL does not preserve the warm start

Running RL on top of the warm start drives the win-rate **down**, from 0.52 to
0.04. Instrumenting the first thousand gradient steps shows the policy dying
between step 30 and step 300:

| gradient steps | Q scale | win-rate |
|---|---|---|
| 0 | 6.41 | 0.212 |
| 30 | 4.89 | 0.275 |
| 300 | 2.75 | 0.062 |

Three contributing causes are identified and measured — a scale mismatch
between cross-entropy logits and the return range, an un-penalised stalling
exit, and replay overfitting on a small buffer. Two are fixed; the collapse is
not yet closed. Five approaches were tried and are tabulated with their
results, including the negative ones, in
**[`docs/POSTMORTEM.md`](docs/POSTMORTEM.md)**.

---

## 🧠 What's in here

| | |
|---|---|
| ♟️ **Exact endgame solver** | Retrograde analysis over ~368 000 states. Reproduces textbook theory: KQ vs K mates in **10 moves**, KR vs K in **16**. No tablebase download needed |
| 🎓 **Supervised warm start** | Provably optimal (position, move) pairs from the solver, cross-entropy pre-training before RL — the AlphaGo recipe with a perfect teacher |
| 📏 **DTM-quality metric** | Fraction of moves that *worsen* the distance to mate. Interpretable in a way Elo never is |
| 🧩 **Legal-action masking** | Illegal Q-values masked to −∞ when acting, when bootstrapping, **and** in the dueling baseline |
| 🧠 **Dueling Double DQN** | Conv trunk → V(s) + A(s,a), GroupNorm, Double DQN target |
| ⚖️ **Prioritized replay** | Proportional PER on a SumTree with importance-sampling correction |
| 🎯 **Policy-invariant shaping** | F(s,s′) = γΦ(s′) − Φ(s) with **Φ(terminal) = 0** — verified to 1e-9, so the Ng et al. (1999) guarantee actually holds |
| 📊 **Separated W/D/L** | Every result reported against a baseline computed on the same positions |
| ⚓ **Imitation anchor** | DQfD-style auxiliary loss to keep RL from drifting off the demonstrated policy (ablated against a matched control) |
| 🧪 **161 pytest tests** | Including regression tests for five bugs that silently broke learning |

---

## 🔍 The metric that hid four bugs

An earlier version reported `score = (W + 0.5·D)/n` and sat at 0.55 for 2 100
episodes. Measuring the baseline explains it:

| | `score` | **pure win-rate** | draws |
|---|---|---|---|
| Random vs. random, full chess | **0.499** | **0.065** | 86.8% |

Since 87% of random games are drawn, the score *starts* at 0.499 and saturates.
A learning agent and a coin flip look identical. Underneath were five real bugs
— a dueling baseline averaged over 4 000 illegal actions, BatchNorm in a DQN,
truncation treated as terminal, shaping that voided its own policy-invariance
guarantee, and a gradient schedule that never bound.

All five are fixed, each with a regression test.
Full write-up: **[`docs/POSTMORTEM.md`](docs/POSTMORTEM.md)**.

---

## 📂 Structure

```
GMAI/
├── configs/endgame.yaml        # scope, curriculum, warm start
├── scripts/
│   ├── pipeline.py             # resumable stages: warmstart / rl / report
│   └── ablate_anchor.py        # A/B the imitation anchor against a control
├── src/gmai/
│   ├── tablebase.py            # exact solver (retrograde analysis)
│   ├── warmstart.py            # supervised pre-training + DTM metric
│   ├── endgames.py             # position generator
│   ├── metrics.py              # separated W/D/L instrumentation
│   ├── encoding.py             # board → 18×8×8 · move ↔ action id
│   ├── environment.py          # Gymnasium env, terminated ≠ truncated
│   ├── model.py                # dueling net, masked baseline, GroupNorm
│   ├── replay_buffer.py        # uniform + PER, stores both masks
│   ├── agent.py                # Double DQN
│   ├── rewards.py              # terminal + potential-based shaping
│   ├── train.py · evaluate.py · play.py · uci.py
├── docs/
│   ├── DESIGN.md               # scope, solver, shaping, evaluation
│   ├── POSTMORTEM.md           # the metric, the five bugs, what generalises
│   └── PLAYING_ONLINE.md       # UCI + Lichess deployment
└── tests/                      # 161 tests
```

---

## 🚀 Quickstart

```bash
pip install -e ".[dev]"

# solve the endgame exactly (~85 s, cached to tablebases/)
python -m gmai.tablebase --kind KQvK

# train: supervised warm start, then curriculum RL
python -m gmai.train --config configs/endgame.yaml

# evaluate — reports the agent AND the random baseline, side by side
python -m gmai.evaluate --checkpoint runs/<run>/final.pt --games 200

# play, or expose as a UCI engine
python -m gmai.play --checkpoint runs/<run>/final.pt --color white
python -m gmai.uci  --checkpoint runs/<run>/final.pt
```

To iterate on one stage at a time without repeating the expensive parts:

```bash
python scripts/pipeline.py warmstart --kind KQvK --positions 20000 --epochs 16
python scripts/pipeline.py rl        --kind KQvK --episodes 3000 --epsilon 0.25
python scripts/pipeline.py report    --kind KQvK
```

```bash
pytest -q          # 161 passed
```

---

## 🌐 Playing online

UCI adapter included, so the agent plugs into any chess GUI and into **Lichess**
via the official [`lichess-bot`](https://github.com/lichess-bot-devs/lichess-bot)
bridge. Walkthrough in [`docs/PLAYING_ONLINE.md`](docs/PLAYING_ONLINE.md).

> Given the scope, a Lichess bot should advertise that it is only competent in
> endgames. chess.com's fair-play policy prohibits engine assistance in human
> games, so there is no legitimate way to deploy there.

---

## 🗺️ Roadmap

- [ ] Close the RL-collapse problem: larger `learn_start`, stronger anchor, frozen trunk
- [ ] Reach 90% on KQ vs K, then KR vs K and KRR vs K
- [ ] Stockfish ladder via `cutechess-cli` with **Elo ± error bars**
- [ ] FastAPI service with an honest `in_scope` field, Docker, CI/CD
- [ ] Live Lichess Elo dashboard
- [ ] Full 8×8×73 action space (under-promotions) if scope ever widens

## 🔗 Related

- 🎯 [AR1 — Reinforcement Learning portfolio](https://github.com/Luismpso/AR1)
- 🏎️ [AR2 — Autonomous FS Racing Agent](https://github.com/pedroreis2468/AR2)

## 📄 License

MIT © 2026 Luis Miguel Pereira Silva
