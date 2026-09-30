# Adding chessnet to the GMAI repository

Two packages side by side under `src/`: `gmai` (forced-mate endgames, DQN) stays
exactly as it is; `chessnet` is the full-chess policy-value network trained on
human games from the Lichess database.

## Files

Extract this archive at the repository root. It adds or replaces:

```
src/chessnet/                  the package
scripts/download_lichess.py    resumable, checksum-verified archive download
scripts/extract_lichess.py     PGN -> training shards (bitboards)
scripts/bench_train.py         GPU vs data-pipeline throughput
scripts/overnight.ps1          download -> extract -> train, unattended
tests/test_chessnet_consistency.py
tests/test_tensorboard_isolation.py
tests/test_download_lichess.py
```

`pyproject.toml` already finds packages under `src/`, so nothing changes there.
Everything is formatted and linted with the repository's ruff settings.

## Setup (Windows, venv)

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
pip install -e ".[dev]" zstandard tensorboard
pytest tests/test_chessnet_consistency.py tests/test_download_lichess.py -q
```

Keep `data/` out of git: the archive alone is ~33 GB.

## Data

```powershell
python scripts/download_lichess.py            # 2025-01 into data/raw, resumable
python scripts/download_lichess.py --list     # available months
```

Interrupted? Run the same command again. The file is verified against the
SHA-256 published by Lichess before it gets its final name.

Alternatively, `extract_lichess.py --month 2025-01` streams straight from
Lichess and stops once it has enough positions (roughly a quarter of the file
for 80M positions), at the cost of starting over if the connection drops.

## Training

```powershell
.\scripts\overnight.ps1          # download (if needed) -> extract (if needed) -> train
tensorboard --logdir runs        # live graph, http://localhost:6006
```

Training stops by itself when validation accuracy stops improving; Ctrl+C is
safe at any point, and `best.pt` always holds the best model so far.
