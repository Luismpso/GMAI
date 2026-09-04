# GMAI

A search-free deep reinforcement learning agent for forced-mate chess endgames,
served as a containerised HTTP API.

[![CI](https://github.com/Luismpso/GMAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Luismpso/GMAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

GMAI selects moves in three-and-four-piece chess endgames using a single
forward pass through a dueling Double DQN — no tree search. It ships with an
exact solver for its own domain, which provides both provably optimal training
labels and a ground-truth evaluation metric.

The project covers the full path from research to deployment: a Gymnasium
environment, a training pipeline, reproducible evaluation, an inference service
with Prometheus instrumentation, container images, and CI that gates both code
quality and model performance.

| | |
|---|---|
| Domain | KQ vs K, KR vs K, KRR vs K |
| Model | Dueling Double DQN, convolutional trunk over 18×8×8 planes |
| Inference | ~7 ms per move on CPU |
| Interfaces | HTTP (FastAPI), UCI, CLI |
| Tests | 203, including a performance regression gate |

## Scope and limitations

**GMAI does not play full chess.** The domain is limited to forced-mate
endgames by design.

A search-free DQN selects moves from one evaluation of the current position.
From the opening, that requires assessing a 40-move horizon against a reward
signal that arrives only at the end — the credit assignment problem AlphaZero
addresses by pairing its network with Monte Carlo tree search. Endgames are the
regime where the method is applicable on its own: horizons under 20 moves,
terminal rewards reachable through exploration, and a random-play win rate of
exactly 0.000, giving evaluation full dynamic range.

The service reports scope per request. Positions outside the trained endgames
receive a legal move together with `in_scope: false` and a reason.

Two limitations are material and are documented rather than worked around:

- The model draws roughly 45% of won positions rather than converting them.
  It does not lose — it fails to finish.
- Reinforcement learning on top of the supervised warm start currently degrades
  performance. The published checkpoint is warm-start only.

Both are quantified in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), with the
investigation in [`docs/POSTMORTEM.md`](docs/POSTMORTEM.md).

---

## Quick start

### Run the service

```bash
docker compose up --build
```

This starts the inference API on port 8000, Prometheus on 9090, and Grafana on
3000 with a provisioned dashboard.

```bash
curl -X POST localhost:8000/move \
  -H 'Content-Type: application/json' \
  -d '{"fen":"4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"}'
```

```json
{
  "uci": "a2c4",
  "san": "Qc4",
  "q_value": 0.75,
  "inference_ms": 6.155,
  "in_scope": true,
  "scope_detail": "KQvK",
  "legal_moves": 26
}
```

### Local development

```bash
pip install -e ".[dev,api]"
python -m gmai.tablebase --kind KQvK    # solve the endgame (~85 s, cached)
pytest -q                                # 203 tests
```

---

## API reference

Interactive documentation is served at `/docs`.

### `POST /move`

Select a move for a position.

**Request**

| Field | Type | Description |
|---|---|---|
| `fen` | string | Position in Forsyth–Edwards Notation |

**Response**

| Field | Type | Description |
|---|---|---|
| `uci` | string | Selected move in UCI notation |
| `san` | string | The same move in algebraic notation |
| `q_value` | float | Value the model assigns to the move |
| `inference_ms` | float | Model inference time |
| `in_scope` | bool | Whether the position is a trained endgame |
| `scope_detail` | string | Endgame identifier, or the reason it is out of scope |
| `legal_moves` | int | Number of legal moves in the position |

**Status codes**

| Code | Condition |
|---|---|
| 200 | Move returned |
| 409 | The game is already over in this position |
| 422 | Malformed FEN, or a position that is not legally reachable |
| 503 | No checkpoint loaded |

### `GET /health`

Returns service status, package version, the loaded checkpoint path, and the
list of supported endgames. Reports `model_loaded: false` rather than refusing
to start when no checkpoint is available, so the condition is visible to
monitoring.

### `GET /metrics`

Prometheus exposition. Request counts by outcome, inference latency histogram,
Q-value distribution of selected moves, and the out-of-scope request rate.

---

## Architecture

### State and action representation

Positions are encoded as 18 binary 8×8 planes, always from the perspective of
the side to move, so the network learns a single colour-agnostic
representation.

| Planes | Contents |
|---|---|
| 0–5 | Own pieces (P, N, B, R, Q, K) |
| 6–11 | Opponent pieces |
| 12 | Side to move |
| 13–16 | Castling rights |
| 17 | En-passant target square |

Actions are `from_square × 64 + to_square`, giving 4096 discrete actions.
Promotions resolve to a queen; under-promotion is unreachable and is documented
as a constraint on widening scope.

### Learning

Double DQN with the target's argmax restricted to legal moves:

```
a* = argmax over LEGAL a of Q_online(s', a)
y  = r + γ · (1 − terminated) · Q_target(s', a*)
```

The dueling baseline is the mean advantage over **legal** actions only.
Averaging over all 4096 outputs injects the mean of roughly 4000 untrained
values into every Q-value; this was one of five defects that silently prevented
learning, each now covered by a regression test.

Reward shaping is potential-based, `F(s,s') = γΦ(s') − Φ(s)` with
`Φ(terminal) = 0`, which preserves the optimal policy. The telescoping identity
is verified numerically to 1e-9.

### Exact endgame solver

Three-piece endgames have 524 288 encodable states, of which ~368 000 are
legal — small enough to solve outright. `gmai/tablebase.py` computes
distance-to-mate by retrograde analysis rather than depending on a tablebase
download.

Results reproduce established chess theory:

| Endgame | Legal states | Won | Maximum DTM under optimal play | Known result |
|---|---|---|---|---|
| KQ vs K | 368 452 | 93.7% | 20 plies (10 moves) | mate in ≤ 10 |
| KR vs K | 368 452 | — | 32 plies (16 moves) | mate in ≤ 16 |

This provides two capabilities that self-play alone cannot: provably optimal
supervised labels, and an interpretable evaluation metric — the proportion of
moves that worsen the distance to mate.

---

## Results

All figures are reported as separated wins, draws and losses against a
random-play baseline computed on the same positions with the same seed. The
conventional chess score `(W + 0.5·D)/n` is not used as a headline metric:
because most random games are drawn, it begins at approximately 0.5 and carries
almost no signal.

<!-- BEGIN:RESULTS -->
Checkpoint `final.pt`, 150 games per cell, seed 0. Generated 2026-09-04 by `scripts/evaluate_all.py`.

| Endgame | Defender | Agent | W | D | L | Win rate |
|---|---|---|---|---|---|---|
| KQvK | random | model | 79 | 71 | 0 | **0.527** |
| KQvK | random | random | 1 | 149 | 0 | 0.007 |
| KQvK | stubborn | model | 52 | 98 | 0 | **0.347** |
| KQvK | stubborn | random | 1 | 149 | 0 | 0.007 |

Move quality against the solver's ground truth:

| Endgame | Optimal moves | Suboptimal | Turns a win into a draw |
|---|---|---|---|
| KQvK | 70.0% | 28.0% | 2.0% |
<!-- END:RESULTS -->

The failure mode is conversion rather than defeat: the model does not lose, it
draws. A per-move error rate of a few percent compounds over the nine or so
moves a mate requires, and the numbers above are regenerated from the published
checkpoint by `scripts/evaluate_all.py`.

---

## Deployment

Two container images are maintained separately.

| | `Dockerfile` | `Dockerfile.train` |
|---|---|---|
| Purpose | Inference | Training |
| PyTorch | CPU wheel | CUDA runtime |
| User | Non-root (uid 10001) | root |
| Additional | FastAPI, Prometheus client | Gymnasium, matplotlib, solver cache |

Single-position inference is dominated by interpreter overhead rather than
matrix multiplication, so the CPU wheel costs nothing and removes approximately
2.5 GB from the deployed image. `tests/test_serving_boundary.py` enforces the
separation by importing the API with training-only packages blocked, and
verifies that the blocking mechanism itself is effective.

```bash
docker build -t gmai-api .
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" gmai-api
```

Tagged releases publish multi-architecture images to GHCR. Cloud Run
instructions are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Configuration

| Variable | Default | Description |
|---|---|---|
| `GMAI_CHECKPOINT` | `/app/models/final.pt` | Path to the model checkpoint |
| `GMAI_DEVICE` | `cpu` | Inference device |

---

## Testing and CI

```bash
pytest -q                              # 203 tests
ruff check src tests scripts           # lint
ruff format --check src tests scripts  # formatting
```

Every pull request runs lint, type checks, the test suite on Python 3.10–3.12,
a container build with an HTTP smoke test, and a performance regression gate.

The regression gate evaluates a fixed checkpoint on fixed positions with a
fixed seed and fails if win rate or move quality drop below a recorded
baseline. Unit tests catch code that raises; this catches code that runs
correctly and plays worse, which is the failure mode this project has
demonstrably had.

```bash
python scripts/regression_check.py --checkpoint models/final.pt
python scripts/regression_check.py --checkpoint models/final.pt --update
```

Baseline updates belong in their own commit.

### Regenerating published results

```bash
python scripts/evaluate_all.py --checkpoint models/final.pt --inject
```

Evaluates the checkpoint against both defenders with a matched random baseline,
computes move quality against the solver, writes `docs/results.json`, and
rewrites the results tables in this file and the model card. Everything is
seeded, so the same checkpoint always produces the same report.

---

## Project structure

```
.
├── src/gmai/
│   ├── api.py              HTTP inference service
│   ├── tablebase.py        Exact solver (retrograde analysis)
│   ├── warmstart.py        Supervised pre-training and DTM metrics
│   ├── endgames.py         Position generator
│   ├── encoding.py         Board and move representation
│   ├── environment.py      Gymnasium environment
│   ├── model.py            Dueling network
│   ├── agent.py            Double DQN agent
│   ├── replay_buffer.py    Uniform and prioritized replay
│   ├── rewards.py          Terminal rewards and potential shaping
│   ├── metrics.py          Evaluation instrumentation
│   ├── train.py            Training loop
│   ├── evaluate.py         Arena evaluation
│   ├── uci.py              UCI protocol adapter
│   └── play.py             Terminal interface
├── tests/                  203 tests
├── scripts/
│   ├── pipeline.py         Resumable training stages
│   ├── evaluate_all.py     Reproducible evaluation report
│   ├── regression_check.py Performance gate
│   ├── ablate_anchor.py    Controlled ablation
│   └── doctor.py           Environment diagnostics
├── deploy/                 Prometheus config, Grafana dashboard
├── docs/                   Design, model card, post-mortem, deployment
├── configs/endgame.yaml    Training configuration
├── Dockerfile              Inference image
├── Dockerfile.train        Training image
└── docker-compose.yml      Full local stack
```

---

## Documentation

| Document | Contents |
|---|---|
| [Model card](docs/MODEL_CARD.md) | Intended use, training data, measured performance, limitations |
| [Design notes](docs/DESIGN.md) | Scope rationale, solver design, reward shaping, evaluation methodology |
| [Post-mortem](docs/POSTMORTEM.md) | Five defects that prevented learning, and the investigations that found them |
| [Deployment](docs/DEPLOYMENT.md) | Docker, Compose, GHCR, Cloud Run |
| [Playing online](docs/PLAYING_ONLINE.md) | UCI integration and Lichess bot setup |

## Roadmap

- Resolve the RL degradation described in the post-mortem
- Extend coverage to KR vs K and KRR vs K at the target win rate
- Stockfish evaluation ladder with Elo confidence intervals
- Cloud Run deployment with a public endpoint

## Related work

- [AR1](https://github.com/Luismpso/AR1) — Reinforcement learning portfolio
- [AR2](https://github.com/pedroreis2468/AR2) — Autonomous racing agent

## License

MIT. See [LICENSE](LICENSE).
