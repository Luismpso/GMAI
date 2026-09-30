# Download (if needed), extract (if needed), then train - unattended.
#
#   .\scripts\overnight.ps1                   # train until it stops improving
#   .\scripts\overnight.ps1 -Hours 6          # fixed time budget instead
#   .\scripts\overnight.ps1 -Month 2026-08    # use another Lichess month
#   .\scripts\overnight.ps1 -SkipExtract      # shards already in data/train
#
# Each stage is skipped when its output already exists, so running this again
# after an interruption carries on from where it stopped. The download itself
# resumes mid-file.
#
# Watch progress live in another terminal:
#   tensorboard --logdir runs        then open http://localhost:6006
#
# Ctrl+C is safe at any point: best.pt always holds the best model so far.

param(
    [double]$Hours = 0,
    [string]$Month = "2025-01",
    [int]$Positions = 80000000,
    [int]$MinElo = 2000,
    [int]$Workers = 10,
    [int]$Channels = 192,
    [int]$Blocks = 10,
    [int]$BatchSize = 1024,
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"
$started = Get-Date

function Find-Archive {
    (Get-ChildItem data -Recurse -Filter "lichess*.zst" -ErrorAction SilentlyContinue |
        Select-Object -First 1).FullName
}

$haveShards = (Test-Path data/train) -and (Get-ChildItem data/train -Filter "shard_*.npz" -ErrorAction SilentlyContinue)
if (-not $SkipExtract -and -not $haveShards) {
    $pgn = Find-Archive
    if (-not $pgn) {
        Write-Host "=== downloading Lichess $Month (resumable) ===" -ForegroundColor Cyan
        python scripts/download_lichess.py --month $Month --out-dir data/raw
        if ($LASTEXITCODE -ne 0) { throw "download failed; run this script again to resume" }
        $pgn = Find-Archive
    }
    Write-Host "=== extracting $Positions positions, both players >= $MinElo ===" -ForegroundColor Cyan
    python scripts/extract_lichess.py --file $pgn --positions $Positions --min-elo $MinElo --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "extraction failed" }
} else {
    Write-Host "=== using existing shards in data/train ===" -ForegroundColor Cyan
}

$trainArgs = @("--data", "data/train", "--channels", $Channels, "--blocks", $Blocks,
               "--batch-size", $BatchSize)
if ($Hours -gt 0) {
    $left = [math]::Round($Hours - ((Get-Date) - $started).TotalHours, 2)
    if ($left -lt 0.5) { throw "not enough time left to train" }
    Write-Host "=== training for $left h (fixed budget) ===" -ForegroundColor Cyan
    $trainArgs += @("--max-hours", $left)
} else {
    Write-Host "=== training until it stops improving (Ctrl+C to stop early) ===" -ForegroundColor Cyan
}
python -m chessnet.train @trainArgs

Write-Host "`ntotal: $([math]::Round(((Get-Date) - $started).TotalHours, 2)) h" -ForegroundColor Green
