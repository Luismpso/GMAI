# Playing online

GMAI implements UCI, so the same entry point works with any chess GUI (Arena,
Cute Chess, En Croissant, BanksiaGUI) and with the Lichess bot bridge.

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
python -m gmai.uci --checkpoint runs/<run>/final.pt
```

Manual smoke test (type these, one per line):

```
uci
isready
position startpos moves e2e4
go
```

Expected: `id name` / `uciok`, then `readyok`, then a legal `bestmove`.

> **Note on time controls.** GMAI is a *policy*, not a search — it answers in one forward pass, so `go` ignores `wtime`/`btime`/`movetime` and replies immediately. That is valid UCI: the engine is simply very fast. It also means GMAI never loses on time.

### Wrapper script

GUIs and bridges expect a single executable. Create `gmai-engine.sh` (`chmod +x`):

```bash
#!/usr/bin/env bash
cd /absolute/path/to/GMAI
exec python -m gmai.uci --checkpoint runs/<run>/final.pt
```

On Windows, the equivalent `gmai-engine.bat`:

```bat
@echo off
cd /d C:\path\to\GMAI
python -m gmai.uci --checkpoint runs\<run>\final.pt
```

---

## Local GUIs

Register `gmai-engine.sh` as a UCI engine in **Arena**, **Cute Chess**, **En Croissant** or **BanksiaGUI**. Cute Chess is the best choice for benchmarking: `cutechess-cli` runs automated matches and computes Elo with error bars.

```bash
cutechess-cli \
  -engine name=GMAI cmd=./gmai-engine.sh proto=uci \
  -engine name=SF-lvl1 cmd=stockfish proto=uci option.UCI_LimitStrength=true option.UCI_Elo=1320 \
  -each tc=10+0.1 -games 100 -pgnout gmai_vs_sf.pgn
```

This is how you get a **real Elo anchor** for the README, rather than only the internal random/greedy baselines.

---

## Lichess bot

`lichess-bot` is the official free bridge between the Lichess Bot API and chess engines. Your bot plays humans and other bots, and the games are viewable live on Lichess.

### 1. Create a new account

Create a **brand-new** Lichess account and **play zero games on it**. An account with played games can never be upgraded to a BOT account.

### 2. Generate an API token

At `lichess.org/account/oauth/token`, create a personal token with the **"Play bot moves"** scope. Store it — it is shown only once.

### 3. Install the bridge

```bash
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
pip install -r requirements.txt
```

### 4. Configure the engine path

In `config.yml`:

```yaml
token: "your_token_here"
url: "https://lichess.org/"

engine:
  dir: "/absolute/path/to/GMAI"
  name: "gmai-engine.sh"
  protocol: "uci"

challenge:
  concurrency: 1
  variants: ["standard"]
  time_controls: ["rapid", "classical"]   # avoid bullet: model inference latency
  modes: ["casual"]                       # go rated once it stops blundering
```

### 5. Upgrade the account and run

```bash
python lichess-bot.py -u      # -u upgrades the account, then starts playing
```

> **The upgrade is irreversible.** The account can only ever be a bot
> afterwards, which is why step 1 specifies a new account.

Subsequent runs need no `-u`:

```bash
python lichess-bot.py
```

---

## Expected performance

A search-free DQN with a 4096-action head will not be strong, and will hang
pieces well into training. This is the expected result rather than a defect: it
is the reason AlphaZero pairs a network with MCTS instead of acting greedily on
Q-values.

Milestones worth recording:

| Milestone | What it demonstrates |
|---|---|
| Beats `RandomOpponent` > 95% | learned the mechanics of winning |
| Beats `GreedyMaterialOpponent` > 60% | learned not to hang pieces |
| Positive score vs. Stockfish `UCI_Elo=1320` | genuinely competitive play |
| A rated Lichess Elo of any kind | end-to-end deployment works |
