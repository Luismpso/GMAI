# ♟️ GMAI — Grand Master AI

**Deep Reinforcement Learning chess engine** · Dueling Double DQN · Prioritized Experience Replay · legal-action masking · 3-stage curriculum self-play · 70 pytest tests

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-env-2e7d32?style=flat-square)
![python-chess](https://img.shields.io/badge/python--chess-rules-2e7d32?style=flat-square)
![Tests](https://img.shields.io/badge/pytest-70%20passed-1b5e20?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-66bb6a?style=flat-square)

A chess agent that learns **from zero knowledge** — no opening books, no handcrafted evaluation — purely by playing, first against scripted opponents and then against frozen copies of itself.

Builds directly on my RL coursework: the **DQN → Double DQN → Dueling → PER** ladder from my [RL portfolio (AR1)](https://github.com/Luismpso/AR1), and the **curriculum + opponent randomization** recipe from my [autonomous racing agent (AR2)](https://github.com/pedroreis2468/AR2) — here the "randomised domain" is the adversary itself.

---

## ✨ Highlights

| | |
|---|---|
| 🧩 **Custom Gymnasium env** | Single-agent formulation: the env owns the opponent and replies inside `step()` · colours alternate every episode · truncation at 200 moves |
| 🎭 **Legal-action masking** | The make-or-break trick for DQN on chess: illegal Q-values are masked to −∞ **both** when acting *and* in the bootstrap argmax |
| 🧠 **Dueling Double DQN** | Conv trunk over 18×8×8 planes → V(s) + A(s,a) heads · Double DQN target removes maximisation bias (van Hasselt 2016) |
| ⚖️ **Prioritized replay** | Proportional PER on a SumTree (O(log n) sampling) with importance-sampling correction · uniform buffer also available |
| 🎯 **Policy-invariant shaping** | Potential-based material shaping F(s,s′) = γφ(s′) − φ(s) (Ng et al. 1999) — denser signal, provably unchanged optimal policy |
| 🏎️ **Curriculum self-play** | random → greedy-material → pool of frozen past selves, promoted on rolling win-rate — the AR2 domain-randomization idea applied to the opponent |
| 🧪 **70 pytest tests** | Encoding round-trips, mask correctness, dueling identifiability, SumTree proportions, shaping telescoping, checkpoint round-trips |

---

## 🧠 How it works

### State — 18 binary planes (8×8), always from the mover's POV

The board is vertically mirrored when Black moves, so the network learns a single colour-agnostic representation.

```
 0–5   own pieces  (P N B R Q K)      13–16  castling rights (own K/Q, opp K/Q)
 6–11  opp pieces  (P N B R Q K)      17     en-passant square
 12    side to move
```

### Action — `from_square × 64 + to_square` → 4096 discrete actions

Promotions default to a queen (under-promotion is <0.1% of practical moves — documented simplification, full 8×8×73 AlphaZero action space is on the roadmap).

### Learning — masked Double DQN

```
a*  =  argmax over LEGAL a of  Q_online(s′, a)
y   =  r + γ · (1 − done) · Q_target(s′, a*)
```

Huber loss, gradient clipping, target network sync every 1 000 steps, ε-greedy over **legal moves only**.

### Curriculum — three stages, promoted at 85% rolling win-rate

| Stage | Opponent | What the agent learns |
|---|---|---|
| 1️⃣ | `RandomOpponent` | the mechanics of winning (deliver mate, don't stalemate) |
| 2️⃣ | `GreedyMaterialOpponent` (1-ply, mate-aware) | don't hang pieces · punish blunders |
| 3️⃣ | `OpponentPool` — up to 5 frozen snapshots | strategy vs. a *distribution* of past selves, avoiding the classic self-play overfitting trap |

---

## 📂 Project structure

```
GMAI/
├── configs/default.yaml        # all hyperparameters in one place
├── notebooks/
│   └── 01-training-analysis.ipynb   # learning curves + arena report
├── src/gmai/
│   ├── encoding.py             # board → 18×8×8 planes · move ↔ action id
│   ├── environment.py          # Gymnasium ChessEnv with action_mask in info
│   ├── model.py                # DuelingChessNet (conv trunk + V/A heads)
│   ├── replay_buffer.py        # uniform + PER (SumTree)
│   ├── agent.py                # Double DQN agent · masked ε-greedy
│   ├── rewards.py              # terminal + potential-based material shaping
│   ├── opponents.py            # random · greedy · frozen-snapshot pool
│   ├── train.py                # curriculum training loop (CSV logging)
│   ├── evaluate.py             # arena: W/D/L, score, Elo-diff estimate
│   └── play.py                 # play vs. the agent in the terminal
└── tests/                      # 70 tests · pytest
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/Luismpso/GMAI.git && cd GMAI
pip install -e ".[dev]"

# train (CPU works; a GPU is strongly recommended for the full run)
python -m gmai.train --config configs/default.yaml

# evaluate a checkpoint in the arena (100 games vs. each baseline)
python -m gmai.evaluate --checkpoint runs/<run>/final.pt --games 100

# play against it
python -m gmai.play --checkpoint runs/<run>/final.pt --color white
```

Training artifacts (checkpoints + `logs.csv`) land in `runs/<timestamp>/`; open `notebooks/01-training-analysis.ipynb` to inspect the learning curves.

## 🧪 Tests

```bash
pytest -q          # 70 passed
```

The suite covers the parts that silently break RL projects: move↔action round-trips on real positions, mask/legal-move equivalence, POV mirroring, dueling mean-centering, SumTree proportional sampling, the telescoping identity of the shaping term, and save/load round-trips.

---

## 🗺️ Roadmap

- [ ] Full **8×8×73 AlphaZero action space** (under-promotions, no from-to ambiguity)
- [ ] **MCTS + learned prior/value** — the AlphaZero configuration from AR1, scaled up
- [ ] **PPO baseline** for an on-policy vs. off-policy comparison (closing the loop with AR2)
- [ ] Elo anchoring vs. **Stockfish at fixed skill levels**
- [ ] n-step returns + noisy nets (completing the Rainbow checklist)

## 🔗 Related work

- 🎯 [AR1 — Reinforcement Learning portfolio](https://github.com/Luismpso/AR1) · 5 envs · 20+ algorithms · 71 tests
- 🏎️ [AR2 — Autonomous FS Racing Agent](https://github.com/pedroreis2468/AR2) · SAC/PPO · procedural tracks · domain randomization

## 📄 License

MIT © 2026 Luis Miguel Pereira Silva
