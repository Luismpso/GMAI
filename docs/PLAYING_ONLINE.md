# 🌐 Playing online with GMAI

GMAI speaks **UCI**, the standard protocol every chess GUI and bot bridge understands. That means the same engine binary works locally (Arena, Cute Chess, En Croissant) and on Lichess.

---

## ⚠️ chess.com: don't

There is **no legitimate way** to run your own engine in your own chess.com games. Their fair-play policy prohibits engine assistance, and accounts caught doing it get closed. Their Computer Chess Championship exists, but it is invitation-only for established top engines.

**Lichess is the right venue** — it has an official, supported Bot API, and bots are clearly labelled so opponents know what they are playing.

---

## 1. Run GMAI as a UCI engine

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

## 2. Local GUIs

Register `gmai-engine.sh` as a UCI engine in **Arena**, **Cute Chess**, **En Croissant** or **BanksiaGUI**. Cute Chess is the best choice for benchmarking: `cutechess-cli` runs automated matches and computes Elo with error bars.

```bash
cutechess-cli \
  -engine name=GMAI cmd=./gmai-engine.sh proto=uci \
  -engine name=SF-lvl1 cmd=stockfish proto=uci option.UCI_LimitStrength=true option.UCI_Elo=1320 \
  -each tc=10+0.1 -games 100 -pgnout gmai_vs_sf.pgn
```

This is how you get a **real Elo anchor** for the README, rather than only the internal random/greedy baselines.

---

## 3. Lichess bot

`lichess-bot` is the official free bridge between the Lichess Bot API and chess engines. Your bot plays humans and other bots, and the games are viewable live on Lichess.

### Step 1 — a fresh account

Create a **brand-new** Lichess account and **play zero games on it**. An account with played games can never be upgraded to a BOT account.

### Step 2 — API token

At `lichess.org/account/oauth/token`, create a personal token with the **"Play bot moves"** scope. Store it — it is shown only once.

### Step 3 — install the bridge

```bash
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
pip install -r requirements.txt
```

### Step 4 — point it at GMAI

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

### Step 5 — upgrade and run

```bash
python lichess-bot.py -u      # -u upgrades the account, then starts playing
```

> ⚠️ **The upgrade is irreversible.** That account can only ever be a bot afterwards. This is exactly why step 1 says to use a throwaway account, not your main one.

Subsequent runs need no `-u`:

```bash
python lichess-bot.py
```

---

## 4. Expectation management

A DQN chess agent with a 4096-action head and no search will **not** be strong — expect it to hang pieces well into training. That is the honest, interesting result, not a failure: it is exactly why AlphaZero pairs a network with MCTS rather than acting greedily on Q-values.

Useful milestones to record in the README:

| Milestone | What it demonstrates |
|---|---|
| Beats `RandomOpponent` > 95% | learned the mechanics of winning |
| Beats `GreedyMaterialOpponent` > 60% | learned not to hang pieces |
| Positive score vs. Stockfish `UCI_Elo=1320` | genuinely competitive play |
| A rated Lichess Elo of any kind | end-to-end deployment works |
