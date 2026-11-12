# BoardMind AI — one-command local development launcher
# Usage (from the repository root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at $python. Create it first: python -m venv .venv"
}

Write-Host 'Starting BoardMind AI backend on http://127.0.0.1:8001 ...' -ForegroundColor Cyan
$backend = Start-Process -FilePath $python `
    -ArgumentList '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8001', '--reload' `
    -WorkingDirectory $root -PassThru -NoNewWindow

$frontendDir = Join-Path $root 'frontend'
if (-not (Test-Path (Join-Path $frontendDir 'node_modules'))) {
    Write-Host 'Installing frontend dependencies (first run) ...' -ForegroundColor Yellow
    Push-Location $frontendDir
    try { npm install } finally { Pop-Location }
}

Write-Host 'Starting BoardMind AI frontend on http://127.0.0.1:5174 ...' -ForegroundColor Cyan
$frontend = Start-Process -FilePath 'npm.cmd' -ArgumentList 'run', 'dev' `
    -WorkingDirectory $frontendDir -PassThru -NoNewWindow

Write-Host "Backend PID:  $($backend.Id)" -ForegroundColor DarkGray
Write-Host "Frontend PID: $($frontend.Id)" -ForegroundColor DarkGray
Write-Host 'Press Ctrl+C to stop both processes.' -ForegroundColor DarkGray

try {
    Wait-Process -Id $backend.Id, $frontend.Id
} finally {
    foreach ($proc in @($backend, $frontend)) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
