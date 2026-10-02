# GMAI

Two chess engines built from scratch, from data to deployment:

- **ChessNet** plays full chess. A residual policy-value network learns from
  80 million positions of strong human games and chooses moves with Monte Carlo
  tree search. It plays on Lichess as a bot.
- The **endgame agent** solves forced-mate endgames with a search-free dueling
  Double DQN, served as a containerised HTTP API, with an exact solver for
  ground truth.

[![CI](https://github.com/Luismpso/GMAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Luismpso/GMAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

| | ChessNet | Endgame agent |
|---|---|---|
| Plays | Full chess | KQ vs K, KR vs K, KRR vs K |
| Learns from | 80M positions from Lichess games between players rated 2000+ | An exact endgame solver, then reinforcement learning |
| Model | Residual network, 10 blocks × 192 channels, policy and value heads (15.2M parameters) | Dueling Double DQN over 18×8×8 planes |
| Chooses moves by | Monte Carlo tree search (PUCT), batched on the GPU | One forward pass, no search |
| Strength | ≈2490 Elo on Stockfish's scale at 800 simulations per move; 2347 blitz on Lichess | Wins 53% of KQ vs K games against a random defender, never loses |
| Interfaces | UCI, Lichess bot, CLI | HTTP (FastAPI), UCI, CLI |
| Code | [`src/chessnet`](src/chessnet) | [`src/gmai`](src/gmai) |

The two answer the same question from opposite ends. The endgame agent is
deliberately confined to positions where a network can play well on its own,
without search ([design notes](docs/DESIGN.md)). ChessNet takes the route
AlphaZero takes for full chess and pairs the network with tree search.

---

## ChessNet

### Results

Playing strength from matches against Stockfish 19 at fixed `UCI_Elo` levels
(0.5 s per move): 48 games per configuration, six common openings played with
both colours, one rating fitted to all games by maximum likelihood.

| Search per move | Estimated Elo | 95% interval |
|---|---|---|
| None (policy only) | 1650 | 1508 – 1787 |
| 200 simulations | 2063 | 1927 – 2196 |
| 800 simulations | 2493 | 2358 – 2631 |
| 3200 simulations | 2830 | 2665 – 3026 |

Search is worth over a thousand points, with diminishing returns as it grows.
The ratings are on Stockfish's own scale, which is anchored to engine rating
lists rather than to human ratings, and limited-strength Stockfish makes
deliberate random errors that a tactically careful engine exploits; the gaps
between rows are the robust part. The 3200-simulation estimate is loosely
bracketed, since it won every game against the two weaker levels.

On Lichess, ChessNet plays as the bot
[**Luismpso**](https://lichess.org/@/Luismpso), where the clock sets the search
budget (typically 15 000 – 50 000 simulations per move in blitz). Its blitz
rating after the first 32 games, almost all against other bots, is **2347**
(rating deviation 68).

The network predicts the move a 2000+ player chose in **55.6%** of held-out
positions (top-5: 91.9%). Validation positions are sampled from the same games
as training positions, so these figures are somewhat optimistic.

Method, training history and limitations: [`docs/CHESSNET.md`](docs/CHESSNET.md).

### Quick start

```powershell
pip install -e ".[dev,chessnet]"

# Data and training (RTX 5070 Ti: ~40 min extraction, ~5 h training)
python scripts/download_lichess.py --month 2025-01
python scripts/extract_lichess.py --file data/raw/lichess_db_standard_rated_2025-01.pgn.zst --positions 80000000
python -m chessnet.train --data data/train

# Analyse a position, with and without search
python -m chessnet.play --checkpoint runs/<run>/best.pt --fen "<FEN>" --nodes 800

# Measure strength against Stockfish
python -m chessnet.arena --checkpoint runs/<run>/best.pt --stockfish <path-to-stockfish> --nodes 800

# Play on Lichess (setup in bot/README.md)
.\bot\run_bot.ps1
```

Checkpoints are written to `runs/`, which is not versioned. `scripts/overnight.ps1`
runs download, extraction and training unattended, resuming whichever stage is
unfinished.

---

## Endgame agent

The endgame agent selects moves in three-and-four-piece endgames using a single
forward pass through a dueling Double DQN. It ships with an exact solver for its
own domain, which provides both provably optimal training labels and a
ground-truth evaluation metric.

The project around it covers the full path from research to deployment: a
Gymnasium environment, a training pipeline, reproducible evaluation, an
inference service with Prometheus instrumentation, container images, and CI that
gates both code quality and model performance.

| | |
|---|---|
| Domain | KQ vs K, KR vs K, KRR vs K |
| Model | Dueling Double DQN, convolutional trunk over 18×8×8 planes |
| Inference | ~7 ms per move on CPU |
| Interfaces | HTTP (FastAPI), UCI, CLI |

### Scope and limitations

**The endgame agent does not play full chess** (ChessNet does). Its domain is
limited to forced-mate endgames by design.

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

### Quick start

#### Run the service

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

#### Local development

```bash
pip install -e ".[dev,api]"
python -m gmai.tablebase --kind KQvK    # solve the endgame (~85 s, cached)
pytest -q
```

### API reference

Interactive documentation is served at `/docs`.

#### `POST /move`

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

#### `GET /health`

Returns service status, package version, the loaded checkpoint path, and the
list of supported endgames. Reports `model_loaded: false` rather than refusing
to start when no checkpoint is available, so the condition is visible to
monitoring.

#### `GET /metrics`

Prometheus exposition. Request counts by outcome, inference latency histogram,
Q-value distribution of selected moves, and the out-of-scope request rate.

### Architecture

#### State and action representation

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

#### Learning

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

#### Exact endgame solver

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

### Results

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

### Deployment

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

#### Configuration

| Variable | Default | Description |
|---|---|---|
| `GMAI_CHECKPOINT` | `/app/models/final.pt` | Path to the model checkpoint |
| `GMAI_DEVICE` | `cpu` | Inference device |

---

## Testing and CI

```bash
pytest -q                              # 262 tests
ruff check src tests scripts           # lint
ruff format --check src tests scripts  # formatting
```

Every pull request runs lint, type checks, the test suite on Python 3.10–3.12,
a container build with an HTTP smoke test, and a performance regression gate.

Besides unit tests, the suite pins down the failure modes that matter for each
engine. For ChessNet: the training encoding and the play-time encoding must
agree exactly (a Black-to-move orientation bug once made every Black label
wrong); the search must find mates for both colours even with a random network
(mutation-tested against sign errors); an interrupted and resumed training run
must end with bit-for-bit the same weights as an uninterrupted one; and the
Lichess bot configuration must stay compatible with the engine's command line.

The regression gate evaluates a fixed endgame checkpoint on fixed positions
with a fixed seed and fails if win rate or move quality drop below a recorded
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

Evaluates the endgame checkpoint against both defenders with a matched random
baseline, computes move quality against the solver, writes `docs/results.json`,
and rewrites the results tables in this file and the model card. Everything is
seeded, so the same checkpoint always produces the same report.

---

## Project structure

```
.
├── src/
│   ├── chessnet/               Full-chess engine
│   │   ├── encoding.py         Board planes and move indexing
│   │   ├── dataset.py          Training shards, expanded to planes per batch
│   │   ├── model.py            Residual policy-value network
│   │   ├── train.py            Training: plateau schedule, resumable
│   │   ├── search.py           Monte Carlo tree search (PUCT)
│   │   ├── play.py             Move selection, CLI and UCI engine
│   │   └── arena.py            Matches against Stockfish, Elo fitting
│   └── gmai/                   Endgame agent
│       ├── api.py              HTTP inference service
│       ├── tablebase.py        Exact solver (retrograde analysis)
│       ├── warmstart.py        Supervised pre-training and DTM metrics
│       ├── endgames.py         Position generator
│       ├── encoding.py         Board and move representation
│       ├── environment.py      Gymnasium environment
│       ├── model.py            Dueling network
│       ├── agent.py            Double DQN agent
│       ├── replay_buffer.py    Uniform and prioritized replay
│       ├── rewards.py          Terminal rewards and potential shaping
│       ├── metrics.py          Evaluation instrumentation
│       ├── train.py            Training loop
│       ├── evaluate.py         Arena evaluation
│       ├── uci.py              UCI protocol adapter
│       ├── play.py             Terminal interface
│       └── dataset.py, selector.py
│                               First full-chess prototype (imitation with a
│                               one-ply selector), superseded by ChessNet
├── bot/                        Lichess bot: launcher, configuration, run script
├── scripts/
│   ├── download_lichess.py     Resumable, checksum-verified database download
│   ├── extract_lichess.py      PGN to training shards, in parallel
│   ├── overnight.ps1           Download, extract and train, unattended
│   ├── bench_train.py          Training throughput benchmark
│   ├── train_supervised.py     Trainer for the earlier prototype
│   ├── pipeline.py             Resumable training stages (endgame agent)
│   ├── evaluate_all.py         Reproducible evaluation report
│   ├── regression_check.py     Performance gate
│   ├── ablate_anchor.py        Controlled ablation
│   └── doctor.py               Environment diagnostics
├── tests/                      262 tests
├── deploy/                     Prometheus config, Grafana dashboard
├── docs/                       Engine notes, model card, design, post-mortem, deployment
├── configs/endgame.yaml        Endgame training configuration
├── models/final.pt             Endgame checkpoint
├── Dockerfile                  Inference image
├── Dockerfile.train            Training image
└── docker-compose.yml          Full local stack
```

---

## Documentation

| Document | Contents |
|---|---|
| [ChessNet](docs/CHESSNET.md) | Full-chess engine: data, model, training, search, evaluation, limitations |
| [Playing online](docs/PLAYING_ONLINE.md) | UCI engine, chess GUIs, matches against Stockfish, Lichess |
| [Lichess bot](bot/README.md) | Running ChessNet as a Lichess bot |
| [Model card](docs/MODEL_CARD.md) | Endgame agent: intended use, training data, measured performance, limitations |
| [Design notes](docs/DESIGN.md) | Endgame agent: scope rationale, solver design, reward shaping, evaluation methodology |
| [Post-mortem](docs/POSTMORTEM.md) | Five defects that prevented learning, and the investigations that found them |
| [Deployment](docs/DEPLOYMENT.md) | Docker, Compose, GHCR, Cloud Run |

## Roadmap

ChessNet:

- Time management that spends less in the opening and more in the endgame, and
  reuses the search tree between moves
- An opening book for the first moves, which the training data leaves out
- More data, a larger network, and a validation split by whole games
- The trained checkpoint published as a release asset

Endgame agent:

- Resolve the reinforcement learning degradation described in the post-mortem
- Extend coverage to KR vs K and KRR vs K at the target win rate
- Cloud Run deployment with a public endpoint

## License

MIT. See [LICENSE](LICENSE).
