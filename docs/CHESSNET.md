# ChessNet

A full-chess engine: a residual policy-value network trained by supervised
learning on human games from the Lichess open database, choosing its moves with
Monte Carlo tree search.

```
Lichess archive ─▶ download_lichess.py ─▶ extract_lichess.py ─▶ shards ─▶ chessnet.train ─▶ best.pt
 (.pgn.zst, 33 GB)   resumable, SHA-256      filter, parallel     bitboards                     │
                                                                                                ▼
                         lichess-bot ◀── UCI ◀── chessnet.play (MCTS) ──▶ chessnet.arena (vs Stockfish)
```

---

## Data

**Source.** Standard rated games from the Lichess open database, January 2025:
one 32.9 GB compressed archive, verified against the SHA-256 Lichess publishes.
The database is released under CC0.

**Filters.** A position is kept only if:

- both players are rated 2000 or more, and within 300 points of each other;
- the game is blitz, rapid or classical — bullet is played on reflex rather than
  judgement, and correspondence games admit outside analysis;
- the game ended normally (abandoned games are dropped);
- it comes after the first eight plies. Opening theory is memorised rather than
  judged, and so many games share it that it would dominate the data.

**Result.** 80,001,952 positions from 1,147,326 games, in 40 shards totalling
554 MB. Each position is stored as twelve piece bitboards plus four metadata
bytes (side to move, castling rights, en-passant square, halfmove clock), the
move that was played, and the final result from the mover's point of view.
Extraction parses games in parallel at about 34,000 positions per second:
40 minutes for the 80 million.

One percent of positions is held out for validation, sampled by position.

```powershell
python scripts/download_lichess.py --month 2025-01      # resumable; --list shows the months
python scripts/extract_lichess.py --file data/raw/lichess_db_standard_rated_2025-01.pgn.zst --positions 80000000
```

`extract_lichess.py --month 2025-01` can instead stream straight from Lichess
and stop once it has enough positions, at the cost of starting over if the
connection drops. `data/` is git-ignored.

---

## Representation

Positions are expanded from bitboards to planes per batch, on the fly: storing
planes would take 4.6 KB per position, against a few bytes as bitboards.

Planes are always from the point of view of the side to move. For Black the
board is mirrored vertically and the colours swapped, so the network learns one
colour-agnostic problem instead of two mirrored ones.

| Planes | Contents |
|---|---|
| 0–5 | Own pieces (P, N, B, R, Q, K) |
| 6–11 | Opponent pieces |
| 12 | Side to move |
| 13–16 | Castling rights (own king side, own queen side, opponent's) |
| 17 | En-passant target square |
| 18 | Halfmove clock, scaled to [0, 1] |

Moves are `from_square × 64 + to_square`, in the same mirrored frame: 4096
actions, masked to the legal ones. All promotions of a pawn share one action;
search gives the under-promotions a small share of its prior.

The stored move must be mirrored exactly like the board. A test pins the
training encoding to the play-time encoding: before it existed, every label for
Black to move pointed at the wrong square.

---

## Model

| Component | Layers |
|---|---|
| Stem | 3×3 convolution to 192 channels, batch norm, ReLU |
| Tower | 10 residual blocks, each two 3×3 convolutions of 192 channels |
| Policy head | 1×1 convolution to 32 channels, linear to 4096 logits |
| Value head | 1×1 convolution to 8 channels, linear to 256, linear to 1, tanh |

15.2 million parameters. The value is the expected result for the side to move,
in [−1, 1].

---

## Training

| Setting | Value |
|---|---|
| Loss | cross-entropy on the move played (label smoothing 0.05) + 0.3 × MSE on the result |
| Optimiser | AdamW, learning rate 2e-3, weight decay 1e-4, gradient clipping at 2.0 |
| Batch | 1024 positions |
| Precision | bfloat16 autocast, channels-last memory format, cuDNN autotuning |
| Throughput | 18,500–19,500 positions/s on an RTX 5070 Ti |
| Schedule | 1000-step warm-up, then learning rate × 0.3 after 3 evaluations in a row without a 0.1-point gain in validation top-1 |
| Stopping | after 6 evaluations without a 0.1-point gain, or when the learning rate falls below 1e-6 |
| Evaluation | every 5000 steps (5.1 million positions) |

`best.pt` holds the best model so far. `last.pt` is rewritten atomically at
every evaluation with everything needed to continue: optimiser and scheduler
state, step, and position within the epoch.

```powershell
python -m chessnet.train --data data/train                                  # train until progress stops
python -m chessnet.train --data data/train --resume runs/<run>/last.pt      # continue where it stopped
python -m chessnet.train --data data/train --init-from runs/<run>/best.pt --lr 6e-4   # new run from weights
tensorboard --logdir runs
```

A resumed run ends with bit-for-bit the same weights as an uninterrupted one;
the test suite checks exactly that, across an epoch boundary.
`scripts/overnight.ps1` chains download, extraction and training, resuming
whichever stage is unfinished.

### Training history

| Run | Duration | Learning rate | Best step | Validation top-1 | Top-5 |
|---|---|---|---|---|---|
| 1 | 3.4 h, 2.3 epochs | 2e-3 → 6e-4 | 175,000 | 55.16% | 91.68% |
| 2, from run 1 with `--init-from` | 1.3 h | 6e-4 → 1.6e-5 | 85,000 | 55.58% | 91.9% |

Run 2 started from run 1's best checkpoint, at the learning rate run 1 had
reached. Each drop of the learning rate brought a step up in accuracy. By the
end of run 2, training accuracy (56.8%) exceeded validation (55.6%) by 1.2
points: the network had extracted most of what these 80 million positions hold.

The value head learns more slowly. Its error fell from 0.81 to 0.76 MSE: a single
position is a noisy predictor of how a whole game will end.

**Validation caveat.** Validation positions are sampled individually, so they
come from games that also contribute training positions. Neighbouring positions
of one game are similar, which makes the figures above somewhat optimistic. A
split by whole games would be cleaner.

An earlier prototype in the `gmai` package (`scripts/train_supervised.py`,
`gmai.selector`) trained a similar network and chose moves by sampling among
the policy's top three, checked one ply ahead by the value head. ChessNet
replaces it with full tree search.

---

## Search

Monte Carlo tree search in the AlphaZero style. Each simulation descends the
tree choosing the child that maximises

```
Q(s, a) + c_puct · P(s, a) · √N(s) / (1 + N(s, a)),    c_puct = 1.8
```

where `P` is the policy prior, `Q` the mean value of the simulations below that
move and `N` the visit counts. The leaf it reaches is evaluated by the network,
and the value is backed up the path, changing sign at each ply.

- **Exact rules at the leaves.** Checkmate, stalemate, insufficient material,
  the fifty-move rule and threefold repetition are detected by the rules, not
  estimated, so the search finds short tactics even where the value head is
  unsure.
- **First-play urgency.** Unvisited moves start slightly below the parent's own
  value (by 0.2), so likely moves are tried first.
- **Batched evaluation.** Thirty-two simulations descend before one forward pass
  on the GPU; a virtual loss steers each away from the paths the others took.
- **Move choice.** The most visited move, or sampling by visit count with a
  temperature for variety.

Python sets the pace: about 4,000–6,000 simulations per second on an RTX 5070 Ti
with a Ryzen 5 9600X.

In UCI mode the clock decides how long to think: the budget is the remaining
time divided by 30, plus 80% of the increment, never more than a quarter of what
is left. `go nodes` and `go movetime` are also honoured.

The search is tested with a network that knows nothing: it must still find a
back-rank mate for either colour. Flipping the sign of the backed-up value
makes both tests fail.

---

## Evaluation

`chessnet.arena` plays matches against Stockfish limited to a fixed strength
(`UCI_LimitStrength`, `UCI_Elo`) and fits one rating to all games by maximum
likelihood, with a 95% profile-likelihood interval. Games start from common
openings, eight plies deep — where the network's training data begins — each
played once with each colour; with 12 games per level, that is the first six of
the sixteen built in.

```powershell
python -m chessnet.arena --checkpoint runs/<run>/best.pt --stockfish <path> --nodes 0
python -m chessnet.arena --checkpoint runs/<run>/best.pt --stockfish <path> --nodes 800 --elo 1800 2100 2400 2700
```

Each run writes every game to `runs/arena-<time>/games.pgn` and the results to
`results.json`.

### Results

Stockfish 19, 0.5 s per move, 12 games per level at four levels:

| Search per move | Levels | Estimated Elo | 95% interval |
|---|---|---|---|
| None (policy only) | 1500–2400 | 1650 | 1508 – 1787 |
| 200 simulations | 1800–2700 | 2063 | 1927 – 2196 |
| 800 simulations | 1800–2700 | 2493 | 2358 – 2631 |
| 3200 simulations | 1800–2700 | 2830 | 2665 – 3026 |

Each fourfold increase in simulations added 340–430 points: still large, but
shrinking. Without search, almost every loss was a checkmate the network did
not see coming; a few hundred simulations remove most of those.

Read these numbers with three caveats:

- The scale is Stockfish's, anchored to engine rating lists. It is not a human
  rating.
- Limited-strength Stockfish weakens itself with deliberate random errors,
  which an engine that rarely misses a short tactic exploits well. Absolute
  values likely overstate strength against people.
- The 3200-simulation point is loosely bracketed: it won every game against the
  two weaker levels, which carry little information.

Comparisons between rows, measured under identical conditions, are the reliable
part.

### On Lichess

ChessNet plays as the bot [**Luismpso**](https://lichess.org/@/Luismpso)
through lichess-bot (setup in [`bot/README.md`](../bot/README.md)). In blitz,
the clock allows 15,000–50,000 simulations per move. After its first 32 games,
almost all against other bots, its blitz rating is **2347** with a rating
deviation of 68.

Lichess gave the new bot account a provisional rating of 3000, so its first
games were against much stronger engines; the rating settled as games accrued.

---

## Intended use and limitations

**Intended use.** Research and demonstration of supervised learning plus
search in chess; playing on Lichess as a publicly labelled bot; analysing
positions.

**Not intended for** assisting a person during games against other people.
Every major chess site forbids it.

**Known weaknesses**, visible in its games:

- **Openings.** The network never saw the first eight plies of a game in
  training. Search compensates, but an opening book would serve it better.
- **Endgames.** Long technical endgames are rarer in human games and decided by
  calculation, and the value head is least reliable there. Slow losses in
  endgames are its typical defeat against strong engines.
- **Time management.** Dividing the remaining time evenly front-loads thinking
  into the opening; the search tree is also discarded after every move.
- **Draws and resignations.** The value head is not calibrated well enough to
  decide them, so the bot never resigns or offers a draw.
- **Speed.** Search runs in Python, two to three orders of magnitude slower than
  engines written in C++.

---

## Reference hardware

RTX 5070 Ti (16 GB), Ryzen 5 9600X, 32 GB RAM. Training holds the full dataset in
memory, about 10 GB for 80 million positions; use `--limit` to train on fewer.
