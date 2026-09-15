# Start the demo and expose it on a temporary public URL.
#
#   .\demo.ps1
#
# For showing the app to someone who is not at this machine — an interview over
# a call, say. Everything runs here: your Ollama, your model, your database. No
# API key, no deploy, nothing hosted.
#
# The tunnel is a Cloudflare quick tunnel: a throwaway HTTPS address that lives
# only while this window is open. Close it and the URL is dead.
#
# Ctrl+C stops everything.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Port = 8501

function Find-Cloudflared {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "C:\Program Files (x86)\cloudflared\cloudflared.exe",
        "C:\Program Files\cloudflared\cloudflared.exe"
    )) { if (Test-Path $p) { return $p } }
    return $null
}

Write-Host ""
Write-Host "  Aqueduct demo" -ForegroundColor Cyan
Write-Host "  =============" -ForegroundColor Cyan
Write-Host ""

# ── 1. Ollama ────────────────────────────────────────────────────────
# It stops itself after an auto-update, which is the most common reason the
# app loads but every question fails.
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/version" -TimeoutSec 3 -UseBasicParsing | Out-Null
    Write-Host "  [ok]   ollama already running"
} catch {
    Write-Host "  [..]   starting ollama"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    $up = $false
    foreach ($i in 1..30) {
        Start-Sleep -Seconds 2
        try {
            Invoke-WebRequest -Uri "http://localhost:11434/api/version" -TimeoutSec 2 -UseBasicParsing | Out-Null
            $up = $true; break
        } catch { }
    }
    if (-not $up) { Write-Host "  [!!]   ollama did not start - run 'ollama serve' yourself" -ForegroundColor Red; exit 1 }
    Write-Host "  [ok]   ollama running"
}

# ── 2. Streamlit ─────────────────────────────────────────────────────
Write-Host "  [..]   starting the app on port $Port"
$app = Start-Process -FilePath "python" `
    -ArgumentList "-m","streamlit","run","$Root\ui\app.py","--server.port","$Port","--server.headless","true" `
    -PassThru -WindowStyle Hidden

$ready = $false
foreach ($i in 1..40) {
    Start-Sleep -Seconds 1
    try {
        Invoke-WebRequest -Uri "http://localhost:$Port" -TimeoutSec 2 -UseBasicParsing | Out-Null
        $ready = $true; break
    } catch { }
}
if (-not $ready) { Write-Host "  [!!]   the app did not start" -ForegroundColor Red; exit 1 }
Write-Host "  [ok]   app running at http://localhost:$Port"

# ── 3. The tunnel ────────────────────────────────────────────────────
$cf = Find-Cloudflared
if (-not $cf) {
    Write-Host ""
    Write-Host "  cloudflared is not installed, so there is no public URL." -ForegroundColor Yellow
    Write-Host "  Install it with:  winget install Cloudflare.cloudflared"
    Write-Host "  The app is still running locally at http://localhost:$Port"
    Write-Host ""
    Write-Host "  Ctrl+C to stop." -ForegroundColor DarkGray
    try { Wait-Process -Id $app.Id } finally { Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue }
    exit 0
}

Write-Host "  [..]   opening the public tunnel"
$log = Join-Path $env:TEMP "aqueduct-tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue

$tunnel = Start-Process -FilePath $cf `
    -ArgumentList "tunnel","--url","http://localhost:$Port","--no-autoupdate" `
    -PassThru -WindowStyle Hidden -RedirectStandardError $log

# cloudflared prints the address a second or two after it connects.
$url = $null
foreach ($i in 1..40) {
    Start-Sleep -Seconds 1
    if (Test-Path $log) {
        $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue
        if ($m) { $url = $m.Matches[0].Value; break }
    }
}

Write-Host ""
if ($url) {
    Write-Host "  ---------------------------------------------------------------"
    Write-Host "   SHARE THIS:  $url" -ForegroundColor Green
    Write-Host "  ---------------------------------------------------------------"
    Write-Host ""
    Write-Host "   Live while this window stays open. Closing it kills the link."
    Write-Host "   First question takes ~30s - that is the model, not a hang."
} else {
    Write-Host "  [!!]   no tunnel URL. Log: $log" -ForegroundColor Yellow
    Write-Host "         The app is still at http://localhost:$Port"
}

Write-Host ""
Write-Host "  Ctrl+C to stop everything." -ForegroundColor DarkGray
Write-Host ""

try {
    Wait-Process -Id $app.Id
} finally {
    Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
    Write-Host "  stopped." -ForegroundColor DarkGray
}
