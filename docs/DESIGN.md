# Design notes

The decisions behind GMAI, and the reasoning that produced them.

---

## Scope: forced-mate endgames, not chess

GMAI plays **KQ vs K, KR vs K and KRR vs K**. Full chess from the initial
position is out of scope.

This is a deliberate narrowing, not an unfinished feature. A DQN without search
selects moves by a single forward pass over the current position. From the
opening, that asks a network to evaluate a 40-move horizon with a reward signal
that arrives only at the end — the credit assignment problem is severe enough
that AlphaZero pairs its network with Monte Carlo tree search precisely to avoid
it.

Endgames are the regime where the method works on its own terms:

* **short horizon** — under 20 moves, often under 10;
* **reachable reward** — a random agent stumbles into mate occasionally, so the
  terminal signal is actually observed;
* **a baseline of exactly zero** — random play wins 0.000 of KQ vs K games, so
  any win at all is unambiguous signal.

The alternative was to spend months tuning hyperparameters on full chess and
end with an agent that plays badly and cannot be evaluated cleanly. A system
that solves a small problem well is more defensible than one that solves a large
problem badly, and it is honest about which it is doing.

---

## Solving the domain exactly instead of downloading tablebases

A three-piece endgame has 64³ × 2 = 524 288 encodable states, of which ~368 000
are legal. That is small enough to solve outright, so `gmai/tablebase.py`
computes the ground truth by retrograde analysis rather than depending on a
Syzygy download.

Standard backward induction over the game graph:

* a mate is a resolved state at distance 0;
* a **strong-side** state is won as soon as *one* successor is won;
* a **weak-side** state is lost only when *every* legal move leads to a won
  state — tracked with a per-state countdown of unresolved moves;
* anything never resolved is a draw.

Move generation is delegated to `python-chess` rather than hand-rolled. A first
attempt at generating king and slider moves directly was ~15× faster and
*wrong*: it mishandled king retreats along the checking ray, and the propagation
died at depth 1. The version in the repo runs once in ~85 seconds and caches to
disk, which is a good trade for removing an entire class of legality bugs.

### Validation

The results reproduce textbook chess theory — numbers an implementation could
not accidentally agree with:

| Endgame | Legal states | Won | Max DTM under optimal play | Known result |
|---|---|---|---|---|
| KQ vs K | 368 452 | 93.7% | **20 plies = 10 moves** | mate in ≤ 10 |
| KR vs K | 368 452 | — | **32 plies = 16 moves** | mate in ≤ 16 |

Self-validation: 200/200 sampled won positions reach mate in exactly the
predicted number of plies against optimal defence.

### What it unlocks

1. **Supervised warm start.** (position, optimal move) pairs, free and provably
   optimal. Cross-entropy pre-training of the advantage head before RL begins —
   the AlphaGo recipe, except the teacher is perfect rather than human.
2. **An interpretable metric.** The fraction of moves that *worsen* the distance
   to mate. Elo says a model is bad; `throw_away_rate` says how often it turns a
   forced win into a draw, which is a claim you can act on.

Warm starting is documented rather than hidden: the agent still has to hold up
under RL, and evaluation runs against the same baselines either way.

---

## Reward shaping

Terminal rewards are ±1, with draws configurable. In a forced-mate endgame the
default is `draw: -1.0` — a draw *is* a loss when the position was winning, and
scoring it as 0.5 is precisely the averaging mistake that
[`POSTMORTEM.md`](POSTMORTEM.md) is about.

On top of that, potential-based shaping (Ng, Harada & Russell 1999):

```
F(s, s') = γ·Φ(s') − Φ(s),    Φ(terminal) = 0
```

Two potentials are provided. `material_potential` is normalised material
balance, appropriate for full chess. `endgame_potential` combines three terms:

| Term | Weight | Why |
|---|---|---|
| Material | 2.00 | Losing the queen turns a forced win into a dead draw, so it must dominate |
| Enemy king towards the rim | 0.30 | A precondition for mate with a queen or rook |
| Own king closer | 0.20 | A queen or rook cannot mate alone |

These are the classic mating heuristics, expressed as a potential so that
tuning the weights changes only how fast the optimal policy is found, never
what it is.

---

## Environment

Single-agent formulation: the environment owns the opponent and plays its reply
inside `step`, so one call is a full ply pair. The alternative — a two-agent
environment — complicates the replay buffer without buying anything here, since
the defender is fixed during any given evaluation.

`info` always carries `action_mask`, needed both to pick a legal move and to
compute the dueling baseline.

`terminated` and `truncated` are strictly separated, per the Gymnasium API.
Conflating them was one of the five bugs.

---

## Evaluation

Every number is reported against a baseline computed on the same positions with
the same seed. Two defenders:

* **random** — what the agent trains against;
* **stubborn** — runs for the centre, maximises distance from the attacking
  king, and grabs any hanging piece. Not tablebase-optimal, but a much stiffer
  test, and it specifically punishes an agent that has only learned to beat a
  defender that wanders.

Results are always W/D/L separated plus the termination breakdown. A single
scalar collapses three distinct failure modes into one number, which is how the
original bugs stayed hidden.
