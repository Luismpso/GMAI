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

A later run made the boundary unmistakable. With `learn_start: 50000` and
~8.65 environment steps per episode, the buffer does not fill until episode
**5780** — so every episode before that is the warm start running untouched,
with zero gradient steps:

```
ep 5000 | win-rate 0.560     <- no training yet, buffer still filling
ep 5600 | win-rate 0.500
ep 5800 | win-rate 0.415     <- TD learning starts at ~5780
ep 6000 | win-rate 0.205
ep 6400 | win-rate 0.020
```

The policy is flat for 5 800 episodes and gone within 600 of the first
gradient step. That also **rules out replay overfitting**: the buffer held
50 000 transitions, and it collapsed anyway.

Instrumenting the first thousand gradient steps from a warm-started checkpoint
gives the same picture at finer resolution:

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

### Replay overfitting — investigated and ruled out

With `learn_start: 2000`, training begins on a buffer of ~2 900 transitions;
over 1 000 gradient steps at batch 128 each transition is sampled roughly 44
times, and prioritized replay concentrates that further. That looked like a
plausible contributor.

It is not the cause. Raising `learn_start` to 50 000 delayed the onset of
training to episode 5 780 and changed nothing about what happened afterwards —
the collapse was, if anything, faster. A negative result, but a clean one: it
removes buffer size from the list.

### Why the scale cannot simply be calibrated away

The affine calibration was retried with the slope estimated from
**within-position deviations only**, on the reasoning that cross-entropy
constrains Q differences inside a position even if the overall level drifts
between positions. The slope came out non-positive again.

That is not a bug — it is what cross-entropy does. The loss maximises the
logit of the correct move and pushes every other logit down through the
log-sum-exp, with nothing distinguishing a good alternative from a terrible
one. So a pre-trained network reliably knows *which* move is best and carries
almost no information about *how much* each move is worth. There is no linear
relationship to recover, at any level of centring.

Removing the `Phi(s)` offset from the regression targets was also tried, on
the reasoning that shaping only existed to solve exploration and the warm
start already solves it. That fixed the scale precisely — Q settled at 0.970
against a target of ~0.9 — but the policy stayed weak: 50% optimal moves
against 80% from cross-entropy, and a 0.025 win-rate. Converting KQ vs K in
under 15 moves seems to need roughly 80% optimal play; 50% is not enough.

### Where this stands

An auxiliary imitation loss during RL (DQfD, Hester et al. 2018) is implemented
as `ImitationAnchor` and was ablated against a matched control from the same
checkpoint. At `lr=1e-5` it did not help: both arms reached 0.058. That does
not rule the approach out — the anchor may simply be too weak relative to the
TD gradient — but it is not evidence for it either, and it is recorded here as
a negative result rather than presented as a fix.

The next things to try, in order of how much they are expected to matter:

1. **A much lower RL learning rate.** The runs above used `lr: 3e-4`, which is
   a fine-tuning rate for a network that is already close to its target and far
   too high for one that has to travel several units in output space. 1e-6 to
   1e-5 would give the policy orders of magnitude more updates to adapt in.
2. **A less punitive `draw_reward`.** At -1.0, with the agent drawing ~48% of
   games, the expected value of almost every state is strongly negative and the
   difference between a good move and a bad one is small against that. Setting
   it to 0 makes wins the only signal that moves the value, which may be the
   easier learning problem even though it is the less faithful reward.
3. **A stronger anchor** — `lr` at 1e-4 rather than 1e-5, and the large-margin
   loss from DQfD rather than plain cross-entropy.
4. **Freeze the trunk** for the first few thousand TD steps, letting only the
   output layers absorb the scale change.

`scripts/ablate_anchor.py` runs any of these against a matched control from the
same checkpoint in a few minutes per arm.

---

## 4. Silent death, no traceback

A run died right after `[warmstart] building 20000 labelled KQvK positions...`,
returning to the prompt with no error at all. No traceback usually means the
process was killed rather than raising — most often memory.

Measuring each stage separately:

| stage | resident |
|---|---|
| torch imported | 220 MB |
| agent (64ch × 4, hidden 512) | 340 MB |
| tablebase loaded | 341 MB |
| dataset, 20 000 positions | 679 MB |
| **replay buffer when full (100 000)** | **+1.74 GB** |

The buffer dominates everything else, and it is almost entirely waste: each
transition stored two 4 096-element boolean masks at one byte per bool.
`np.packbits` stores them as bits instead, taking a transition from 17.0 KB to
10.84 KB — a full 100 000-capacity buffer drops from 1.74 GB to 1.03 GB. The
default capacity is now 60 000 (~0.62 GB).

`scripts/doctor.py` runs each stage in isolation, reports resident memory after
each, and projects the peak with a full buffer, so the next silent death is a
number rather than a guess. `build_dataset` also reports progress now, which
distinguishes "died during generation" from "died while stacking".

---

## 5. What generalises

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

**A crash with no traceback is a resource problem, not a logic problem.**
Python reports its own exceptions. Silence means something outside Python
ended the process, and the first move is to measure the resource rather than
reread the code.

**Know what a loss function actually constrains.** Cross-entropy fixes the
ranking of outputs and says nothing about their magnitude — obvious in
hindsight, and the reason a warm-started network can be simultaneously a good
policy and unusable Q-values. Two separate attempts to recover the scale
failed for that one reason, and a moment's thought about what the loss
constrains would have predicted both.
