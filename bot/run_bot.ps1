<#
.SYNOPSIS
    Play ChessNet on Lichess through lichess-bot.

.DESCRIPTION
    Fills in the paths in bot/lichess-bot.template.yml, writes bot/lichess-bot.yml
    (git-ignored) and starts lichess-bot with it. Run it from the rl venv.

    The Lichess token never goes into a file of this repository. It is read from
    the LICHESS_BOT_TOKEN environment variable or, failing that, from a copy
    encrypted for your Windows user (DPAPI) that this script saves the first time.

.EXAMPLE
    .\bot\run_bot.ps1
.EXAMPLE
    .\bot\run_bot.ps1 -Checkpoint runs\<run>\best.pt
.EXAMPLE
    .\bot\run_bot.ps1 -ForgetToken
#>
param(
    [string]$Checkpoint = "runs\20261001-124754\best.pt",
    [string]$LichessBot = "",
    [switch]$Upgrade,
    [switch]$ForgetToken
)

$ErrorActionPreference = "Stop"
$gmai = Split-Path -Parent $PSScriptRoot
if (-not $LichessBot) { $LichessBot = Join-Path (Split-Path -Parent $gmai) "lichess-bot" }
$tokenFile = Join-Path $env:USERPROFILE ".lichess-bot-token"

if ($ForgetToken) {
    Remove-Item $tokenFile -ErrorAction SilentlyContinue
    Write-Host "Saved token removed. The next run will ask for it again."
    return
}

# --- checks -----------------------------------------------------------------
if (-not (Test-Path (Join-Path $LichessBot "lichess-bot.py"))) {
    throw "lichess-bot not found in $LichessBot (clone it there, or pass -LichessBot <folder>)"
}
$checkpointPath = $Checkpoint
if (-not [System.IO.Path]::IsPathRooted($checkpointPath)) { $checkpointPath = Join-Path $gmai $Checkpoint }
if (-not (Test-Path $checkpointPath)) { throw "checkpoint not found: $checkpointPath" }
$checkpointPath = (Resolve-Path $checkpointPath).Path

$python = (Get-Command python -ErrorAction Stop).Source
& $python -c "import chessnet.play, torch" 2>$null
if ($LASTEXITCODE -ne 0) { throw "$python cannot import chessnet and torch: activate the rl venv first" }

# --- token ------------------------------------------------------------------
if (-not $env:LICHESS_BOT_TOKEN) {
    if (-not (Test-Path $tokenFile)) {
        $secure = Read-Host "Lichess token (scope bot:play only)" -AsSecureString
        $secure | ConvertFrom-SecureString | Set-Content -Path $tokenFile -Encoding ASCII
        Write-Host "Token saved, encrypted for this Windows user: $tokenFile"
    }
    $secure = Get-Content $tokenFile | ConvertTo-SecureString
    $env:LICHESS_BOT_TOKEN = [System.Net.NetworkCredential]::new("", $secure).Password
}

# --- configuration ----------------------------------------------------------
$config = Get-Content (Join-Path $PSScriptRoot "lichess-bot.template.yml") -Raw
$config = $config.Replace("{{PYTHON}}", ($python -replace '\\', '/'))
$config = $config.Replace("{{ENGINE_DIR}}", ($PSScriptRoot -replace '\\', '/'))
$config = $config.Replace("{{CHECKPOINT}}", ($checkpointPath -replace '\\', '/'))
$configPath = Join-Path $PSScriptRoot "lichess-bot.yml"
Set-Content -Path $configPath -Value $config -Encoding ASCII

# --- run --------------------------------------------------------------------
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("lichess-bot-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
Write-Host "lichess-bot : $LichessBot"
Write-Host "model       : $checkpointPath"
Write-Host "log         : $log"
Write-Host "Ctrl+C to stop."

Push-Location $LichessBot
try {
    $botArgs = @("lichess-bot.py", "--config", $configPath, "--logfile", $log)
    if ($Upgrade) { $botArgs += "-u" }
    & $python @botArgs
}
finally {
    Pop-Location
}
