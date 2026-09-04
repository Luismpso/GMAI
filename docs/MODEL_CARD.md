# Model card — GMAI

## Overview

| | |
|---|---|
| **Name** | GMAI (Grand Master AI) |
| **Task** | Move selection in forced-mate chess endgames |
| **Architecture** | Dueling Double DQN, convolutional trunk over 18×8×8 board planes |
| **Training** | Supervised warm start against an exact solver, then RL against a random defender |
| **Inference** | Single forward pass. **No search.** |
| **License** | MIT |

---

## Intended use

Selecting moves in **KQ vs K, KR vs K, and KRR vs K**, given a legal,
non-terminal position with the strong side to move.

Built as a portfolio and teaching artefact: a small problem solved with the
full engineering path around it — exact evaluation, an HTTP service, container
images, CI, and a written record of what went wrong. It is not a competitive
chess engine and is not trying to be one.

## Out of scope

**Full chess.** The model will return a legal move for any position, but
outside the three trained endgames that move carries no support from training.
The API reports this per request in the `in_scope` field rather than leaving
the caller to guess.

Also out of scope: other endgames (K+P vs K, K+B+N vs K), positions where both
sides have material, and any use where move quality matters more than the
demonstration.

## Why the scope is this narrow

A search-free DQN picks moves from one forward pass over the current position.
From the opening that means evaluating a 40-move horizon with a reward that
arrives only at the end — the credit assignment problem AlphaZero avoids by
pairing its network with tree search.

Forced-mate endgames are where the method works on its own terms: the horizon
is under 20 moves, a random agent stumbles into mate often enough for the
terminal signal to be observed, and random play wins **0.000** of KQ vs K games,
so any win at all is unambiguous signal.

---

## Training data

No human games. Positions are sampled uniformly from legal, non-terminal
configurations of each endgame, and labelled by an **exact solver** built into
the repo (`gmai/tablebase.py`), which computes distance-to-mate by retrograde
analysis over ~368 000 states.

The solver reproduces textbook results — KQ vs K mates in at most 10 moves,
KR vs K in at most 16 — and 200/200 sampled won positions reach mate in exactly
the predicted number of plies against optimal defence.

The supervised labels are therefore **provably optimal**, not merely strong.

---

## Evaluation

All figures are W/D/L separated, reported against a random-play baseline
computed on the same positions with the same seed. The chess score
`(W + 0.5·D)/n` is not used as a headline: because most random games are drawn,
it starts at ~0.5 and has almost no dynamic range — a mistake that hid four
bugs on this project and is documented in
[`POSTMORTEM.md`](POSTMORTEM.md).

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

---

## Known limitations

**It fails to convert.** Zero losses in 300 games — the model does not blunder
into defeat. It draws: 60 threefold repetitions, 45 lost queens, 31 stalemates.
A 2.7% per-move error rate compounds over ~9 moves into roughly a 22% chance of
throwing away any given game.

**RL currently makes it worse.** Running reinforcement learning on top of the
warm start drives the win-rate from 0.52 down to 0.04. The cause is understood
in part — cross-entropy pre-training fixes the ranking of moves but not their
magnitude, so the Q-values are hundreds of units from the return scale and the
first TD gradients destroy the policy — and several remedies have been tried
and measured, none successful. Full account in
[`POSTMORTEM.md`](POSTMORTEM.md).

**The published checkpoint is warm-start only.** No RL, because RL has not yet
been shown to help.

**Under-promotion is unreachable.** The action space is `from × to` (4096
actions) with promotions resolved to a queen. Irrelevant in these endgames,
which contain no pawns, but it blocks any widening of scope without an action
space change.

**No Elo figure is published.** An Elo without an opponent pool and error bars
is not a measurement. A Stockfish ladder via `cutechess-cli` is on the roadmap;
until it exists, the tables above are what there is.

---

## Ethical and safety considerations

A chess endgame model has no meaningful misuse surface. The one honesty
requirement is not overstating scope, which is why `in_scope` is a first-class
response field and why the failure modes above are stated in numbers.

If deployed as a Lichess bot, the profile should say it is only competent in
endgames. chess.com's fair-play policy prohibits engine assistance in human
games, and there is no legitimate way to deploy there.

---

## Reproducing

```bash
pip install -e ".[dev,api]"
python -m gmai.tablebase --kind KQvK          # ~85 s, cached
python -m gmai.train --config configs/endgame.yaml
python -m gmai.evaluate --checkpoint runs/<run>/final.pt --games 200
```

Every reported number comes from `gmai.evaluate` or `gmai.warmstart.dtm_quality`
with fixed seeds. `scripts/regression_check.py` re-runs the headline metrics
against a recorded baseline and fails on regression; it runs on every pull
request.
