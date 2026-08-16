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

## 3. What generalises

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
