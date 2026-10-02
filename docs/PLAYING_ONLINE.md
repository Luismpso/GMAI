# Playing online

ChessNet speaks UCI, so the same entry point works with any chess GUI (Arena,
Cute Chess, En Croissant, BanksiaGUI), with automated matches, and with the
Lichess bot bridge.

---

## chess.com is not a supported target

chess.com's fair-play policy prohibits engine assistance in human games, and
accounts found doing so are closed. The Computer Chess Championship is
invitation-only and limited to established engines. There is no legitimate
deployment path.

Lichess provides an official Bot API and labels bot accounts publicly, so
opponents know what they are playing. It is the supported venue.

---

## Running as a UCI engine

```bash
python -m chessnet.play --checkpoint runs/<run>/best.pt --uci --nodes 1000000
```

| Option | Effect |
|---|---|
| `--nodes 0` | Policy only: one forward pass per move, instant |
| `--nodes N` | Monte Carlo tree search, at most N simulations per move |
| `--device cpu` | Run without a GPU (search is then much slower) |

With search enabled, `go wtime/btime/winc/binc` gives the search a time budget
from the clock (the remaining time divided by 30, plus 80% of the increment,
never more than a quarter of what is left), and `go nodes N` and
`go movetime MS` are honoured. A large `--nodes` therefore means "let the clock
decide".

Manual smoke test (type these, one per line):

```
uci
isready
position startpos moves e2e4
go movetime 2000
```

Expected: `id name ChessNet` / `uciok`, then `readyok`, then an `info` line
with nodes, speed, score and principal variation, and a legal `bestmove`.

### Wrapper script

GUIs expect a single executable. On Windows, `chessnet.bat`:

```bat
@echo off
"C:\path\to\venv\Scripts\python.exe" -u -m chessnet.play --checkpoint "C:\path\to\GMAI\runs\<run>\best.pt" --uci --nodes 1000000
```

On Linux or macOS, `chessnet.sh` (`chmod +x`):

```bash
#!/usr/bin/env bash
exec /path/to/venv/bin/python -u -m chessnet.play --checkpoint /path/to/GMAI/runs/<run>/best.pt --uci --nodes 1000000
```

`-u` keeps the output unbuffered, so every UCI reply reaches the GUI at once.

---

## Local GUIs

Register the wrapper as a UCI engine in **Arena**, **Cute Chess**,
**En Croissant** or **BanksiaGUI**. ChessNet declares no UCI options, so leave
threads, hash and tablebase settings unset.

---

## Matches against Stockfish

`chessnet.arena` plays Stockfish limited to fixed Elo levels and fits a rating
with a confidence interval:

```bash
python -m chessnet.arena --checkpoint runs/<run>/best.pt --stockfish /path/to/stockfish --nodes 800
```

The method and the measured results are in
[`CHESSNET.md`](CHESSNET.md#evaluation). Run one match at a time: Stockfish plays
on a fixed time per move, so a busy processor would make it weaker than its
nominal level.

`cutechess-cli` works as well, with the wrapper above:

```bash
cutechess-cli \
  -engine name=ChessNet cmd=./chessnet.sh proto=uci \
  -engine name=SF2100 cmd=stockfish proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2100 \
  -each tc=60+1 -games 40 -repeat -pgnout chessnet_vs_sf.pgn
```

---

## Lichess bot

ChessNet plays on Lichess as the bot
[**Luismpso**](https://lichess.org/@/Luismpso), through
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot), the official
bridge between the Lichess Bot API and UCI engines.

Everything needed is in [`bot/`](../bot): the configuration, the launcher
lichess-bot runs, and `run_bot.ps1`, which starts it with one command and keeps
the API token out of the repository. Step-by-step setup is in
[`bot/README.md`](../bot/README.md). In short:

1. Create a new Lichess account for the bot. It must never have played a game:
   only such accounts can become bots, and the change is irreversible.
2. Create a token for it with only the `bot:play` scope.
3. Clone lichess-bot next to this repository and install its requirements.
4. Run `.\bot\run_bot.ps1 -Upgrade` once, then `.\bot\run_bot.ps1`.

The bot accepts blitz and rapid challenges with increment, challenges other
bots when idle, and plays one game at a time. Ctrl+C lets the current game
finish before quitting.

---

## Measured performance

| Measure | Result |
|---|---|
| Stockfish 19 matches, policy only | 1650 Elo (95%: 1508–1787) |
| Stockfish 19 matches, 800 simulations per move | 2493 Elo (95%: 2358–2631) |
| Lichess blitz, after 32 games | 2347 (rating deviation 68) |

Stockfish's scale is not a human rating, and Lichess bots play mostly other
bots. Details and caveats in [`CHESSNET.md`](CHESSNET.md#results).

---

## The endgame agent

The DQN endgame agent also speaks UCI:

```bash
python -m gmai.uci --checkpoint models/final.pt
```

It answers in a single forward pass and ignores the clock. It is only competent
in KQ vs K, KR vs K and KRR vs K; anywhere else it returns a legal move with no
training behind it. See the [model card](MODEL_CARD.md).
