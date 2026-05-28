# Full benchmark: all frameworks enabled in benchmark_config.json (no --frameworks filter).
# Logs to the repo root with a timestamp. Uses UTF-8 and unbuffered Python.
param(
    [switch]$Detached
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$repoRoot = Split-Path $PSScriptRoot -Parent
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $repoRoot "benchmark_full_weekend_$stamp.log"

if ($Detached) {
    # Avoid PYTHONUTF8 in cmd.exe — some shells pass a value Python rejects as invalid.
    $cmd = "cd /d `"$PSScriptRoot`" && python -u run_benchmark.py -v > `"$log`" 2>&1"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Minimized
    Write-Host "Detached benchmark started. Log: $log"
    exit 0
}

Write-Host "Logging to: $log"
Write-Host "Started: $(Get-Date)"
python -u run_benchmark.py -v *>&1 | Tee-Object -FilePath $log
Write-Host "Finished: $(Get-Date)"
