# ChessNet on Lichess

Plays ChessNet (policy-value network with Monte Carlo tree search) on Lichess
as a BOT account, through [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot).

| File | Purpose |
|---|---|
| `lichess-bot.template.yml` | lichess-bot settings for ChessNet; `run_bot.ps1` fills in the paths |
| `chessnet_uci.py` | What lichess-bot runs: ChessNet's UCI loop (`chessnet.play --uci`) |
| `run_bot.ps1` | Builds the configuration and starts lichess-bot |

The generated `lichess-bot.yml` and the `logs/` folder are git-ignored.

## One-time setup

1. Clone lichess-bot next to this repository and install its requirements in
   the venv that has ChessNet:

   ```powershell
   git clone https://github.com/lichess-bot-devs/lichess-bot.git ..\lichess-bot
   pip install -r ..\lichess-bot\requirements.txt
   ```

2. Create a Lichess account for the bot. It must never have played a game.
3. Logged in as the bot, create a token with **only** the `bot:play` scope:
   <https://lichess.org/account/oauth/token/create?scopes[]=bot:play&description=lichess-bot>
4. Upgrade the account to BOT (irreversible) and start playing:

   ```powershell
   .\bot\run_bot.ps1 -Upgrade
   ```

   The script asks for the token once and keeps it encrypted for your Windows
   user (DPAPI) in `%USERPROFILE%\.lichess-bot-token`. It never goes into the
   repository. Setting `LICHESS_BOT_TOKEN` yourself also works.

## Everyday use

```powershell
.\bot\run_bot.ps1                                   # default model
.\bot\run_bot.ps1 -Checkpoint runs\<run>\best.pt    # another model
.\bot\run_bot.ps1 -ForgetToken                      # delete the saved token
```

Ctrl+C stops it after the current game ends (no game is lost on time);
press it twice to stop at once. Each run is logged to `bot/logs/`.

## Behaviour

- Accepts blitz and rapid challenges from people and bots: 3 to 30 minutes,
  with increment. No bullet: Python search plus network lag would lose on time.
- When idle for 2 minutes, challenges bots rated within 300 points to rated
  3+2 or 5+3 games.
- One game at a time. Thinks only on its own time. Never resigns or offers
  draws: the value head is not yet reliable enough to decide that.
- Time per move comes from the clock (UCI time management in `chessnet.play`),
  with 2 s held back for network lag. The first move is always searched for
  10 s, because Lichess allows 30 s for it.
- Lichess allows a bot 100 games a day against other bots.

## Linux

lichess-bot refuses engines without the execute bit, which Git on Windows does
not record: run `chmod +x bot/chessnet_uci.py` after cloning. `run_bot.ps1`
also runs under PowerShell 7 (`pwsh`) if `LICHESS_BOT_TOKEN` is set.
