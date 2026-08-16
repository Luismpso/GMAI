# Post-mortem: how one metric hid four bugs

For roughly 2 100 episodes, GMAI trained on full chess and reported a win-rate
hovering around 0.55. It looked like slow learning. It was not learning at all
— and the reason it took so long to notice is worth writing down.

---

## 1. The metric

The training loop reported the standard chess score:

```
score = (wins + 0.5 × draws) / games
```

That is the right way to score a *tournament*. It is a terrible way to measure
a *learning agent*, and measuring the baseline shows why:

| Setting | `score` | **pure win-rate** | draws |
|---|---|---|---|
| Random vs. random, full chess | **0.499** | **0.065** | 86.8% |
| Random vs. random, KQ vs K | 0.500 | **0.000** | 100% |

Because ~87% of random full-chess games end in a draw — mostly insufficient
material after both sides shuffle pieces off the board — the score *starts* at
0.499 and stays there. A perfectly random agent and a genuinely learning one
produce the same number.

Reporting 0.55 against a baseline of 0.499 means the entire observed "progress"
was 0.05 of a metric with no dynamic range. Underneath it, four separate bugs
were preventing any learning at all, and a fifth was making the training
schedule a no-op.

**The fix is not a better threshold, it is a better observable.** Wins, draws
and losses are now tracked separately, along with episode length and *why* each
game ended. The pure win-rate has a random baseline of 0.065 (full chess) or
exactly 0.000 (endgames), so a single win is unambiguous signal.

The termination breakdown turned out to be just as important: it distinguishes
"lost the queen" from "stalemated the opponent" from "shuffled until the
fifty-move rule", which are three completely different failures that a single
scalar collapses into one.

---

## 2. The bugs

Each has a regression test in `tests/test_regressions.py` that fails against the
pre-fix code.

### Dueling advantage normalised over illegal actions

```python
# before
return value + advantage - advantage.mean(dim=1, keepdim=True)   # over 4096
```

The dueling decomposition needs a baseline to make `V` and `A` identifiable,
and the usual choice is the mean advantage over actions. With a 4096-wide
action head and roughly 30 legal moves in a typical position — 3 to 10 in an
endgame — that mean is dominated by ~4 000 outputs that are never selected,
never trained, and therefore pure noise. Every Q-value had that noise
subtracted from it, and `V(s)` could not converge.

The fix restricts the baseline to the legal-action mask. That has a
consequence: the mask now has to be available wherever a forward pass happens,
so it is threaded through the network signature, the replay buffer and the
agent.

### BatchNorm in a DQN

Two independent problems. First, the running statistics are estimated from
replay batches whose distribution shifts continuously as the policy changes.
Second, acting happens on a batch of one, so the statistics used to choose a
move differed from those used to learn its value.

Replaced with **GroupNorm**, which normalises per sample. Train and act agree
by construction, and there are no running statistics to poison.

### Truncation treated as terminal

```python
targets = rewards + gamma * (1.0 - dones) * next_q   # `dones` included truncation
```

Hitting the move limit is an artifact of the training loop, not a property of
the MDP. The successor state still has value and must be bootstrapped; zeroing
it teaches the agent that the position is worth exactly nothing.

With 86% of episodes ending that way, that is close to the entire dataset. The
buffer now stores `terminated`, and the environment reports `truncated`
separately — which is exactly what the Gymnasium API separates them for.

### Potential-based shaping broken at the terminal state

The shaping term was skipped on the final transition. Ng, Harada and Russell
(1999) guarantee that `F(s, s') = γΦ(s') − Φ(s)` leaves the optimal policy
unchanged, but the proof relies on the terms telescoping along the whole
trajectory, which requires `Φ(terminal) = 0` — not the term's absence.

Skipping it meant the shaping was no longer policy-invariant, so the agent was
being nudged towards whatever the material heuristic liked rather than towards
winning. The README claimed a guarantee the code did not have.

With `Φ(terminal) = 0` included, the discounted shaping along any episode
collapses to `−Φ(s₀)`, a constant independent of the policy. Verified
numerically to 1e-9 in
`test_discounted_shaping_telescopes_to_minus_phi_of_start`.

### The gradient-step schedule never bound

```python
if agent.train_steps * cfg["train_every"] <= episode * 400:   # always true
```

Intended to do one gradient step per four environment steps. It simplifies to
`train_steps <= 100 × episode`, which never binds in practice. Replaced with a
plain counter: `env_steps % train_every == 0`.

---

## 3. The warm start does not survive RL (open)

Supervised pre-training against the exact solver reaches a 0.52 win-rate on
KQ vs K. Running RL on top of it drives that to 0.04. The episode length in
that run fell from 19.2 to 6.6 plies — the agent was ending games faster, by
drawing.

This is measured, not inferred. Instrumenting the first thousand gradient
steps from a warm-started checkpoint:

| gradient steps | Q-value scale | win-rate |
|---|---|---|
| 0 | 6.41 | 0.212 |
| 10 | 5.12 | **0.312** |
| 30 | 4.89 | 0.275 |
| 100 | 4.30 | 0.125 |
| 300 | 2.75 | 0.062 |
| 1000 | 3.21 | 0.062 |

The policy briefly *improves*, then is gone within a few hundred steps — about
100 episodes. Three contributing causes have been identified, and none of them
fully accounts for it on its own.

### Scale mismatch

Cross-entropy is invariant to adding a constant to every logit: it fixes the
*ordering* of moves and says nothing about their magnitude. Measured on a
warm-started network, Q-values averaged **−293** against a target return range
of roughly [−1.9, +0.1]. Every TD update then starts with an error near 300,
and the gradients that close that gap do not preserve what the network learned.

Attempts, all measured on KQ vs K:

| Approach | Q scale | win-rate |
|---|---|---|
| Cross-entropy imitation | −293 | **0.52** |
| Pure Q-regression on exact DTM targets | 0.14 | 0.005 |
| Regression + ranking term (fixed weight) | 5.17 | 0.29 |
| Regression + ranking term, annealed | 0.076 | 0.065 |
| Cross-entropy, then affine calibration | *fit failed* | — |

Regression fixes the scale but not the policy: its loss is dominated by
predicting `Φ(s)`, the per-position offset, which varies far more across
positions than the discounted-win term varies across actions within one.

The affine calibration is the most informative failure. Fitting
`Q_true ≈ α·Q_pred + β` and folding α, β into the output layer would fix the
scale while provably preserving the ranking — but the least-squares slope came
out non-positive. Cross-entropy constrains differences *within* a position and
leaves the level free to drift *between* them, so no single global affine map
can recover the true values. `calibrate_scale` detects this and declines.

### An un-penalised exit

Correcting the truncation bug (§2) introduced an incentive it did not have
before. With `draw_reward = -1.0`, the agent's options at the end of an episode
paid: mate `+1`, stalemate or repetition `−1`, and hitting the move limit `≈0`,
because a truncated state bootstraps instead of terminating. Stalling was
strictly the best available outcome, and the agent found it — episode length
rose from 23.6 to 35.2 plies while the win-rate fell.

The general rule (truncation is not termination) is right; the mistake was
applying it here. "Mate within N moves" *is* the task in a forced-mate endgame,
so running out of moves is failing it, exactly like a stalemate. The
environment now takes `move_limit_is_terminal`, default `True` for endgames.

This removed the stalling behaviour but did not stop the collapse.

### Replay overfitting

With `learn_start: 2000` and one gradient step per four environment steps,
training begins on a buffer of ~2 900 transitions. Measured over 1 000 gradient
steps at batch 128, each transition is sampled roughly 44 times — and
prioritized replay concentrates that further on whichever transitions currently
have the largest TD error. That is a small-buffer overfitting regime, and it
lines up with the timescale on which the policy dies.

### Where this stands

An auxiliary imitation loss during RL (DQfD, Hester et al. 2018) is implemented
as `ImitationAnchor` and was ablated against a matched control from the same
checkpoint. At `lr=1e-5` it did not help: both arms reached 0.058. That does
not rule the approach out — the anchor may simply be too weak relative to the
TD gradient — but it is not evidence for it either, and it is recorded here as
a negative result rather than presented as a fix.

The next things to try, in order of how much they are expected to matter:

1. **Much larger `learn_start`** (~50 000) and a lower gradient-to-environment
   step ratio, to leave the small-buffer regime entirely.
2. **A stronger anchor** — `lr` at 1e-4 rather than 1e-5, and the large-margin
   loss from DQfD rather than plain cross-entropy.
3. **Freeze the trunk** for the first few thousand TD steps, letting only the
   output layers absorb the scale change.

---

## 4. What generalises

**Always measure the baseline before measuring the agent.** One run of
random-vs-random would have shown that 0.499 was the floor, and the whole
episode would have been a day rather than weeks.

**Prefer observables with dynamic range.** Between `score` (floor 0.499) and
pure win-rate (floor 0.000 in endgames) the arithmetic is the same; the
diagnostic value is not.

**A metric that averages distinct failures together will hide all of them.**
Stalemating, hanging the queen and shuffling until the fifty-move rule are
three different bugs. As "0.5 each" they are indistinguishable.

**Suspicious stability is a symptom.** A win-rate that sits at 0.55 ± 0.02 for
2 000 episodes while ε decays from 0.96 to 0.05 is not slow learning. Exploration
collapsing by an order of magnitude with no change in behaviour means behaviour
is not driven by what is being learned.

**Measure the transition, not just the endpoints.** Knowing that a warm start
scores 0.52 and post-RL scores 0.04 says nothing about why. Sampling the
win-rate every few gradient steps showed the policy dying between step 30 and
step 300 — which immediately rules out slow drift and points at the first
updates. That one table was worth more than several full training runs.

**Fixing a bug can install an incentive.** Treating truncation as
non-terminal is correct in general and made stalling the highest-value action
in this specific task. A correction that is right in the abstract still has to
be checked against what the agent can now exploit.
