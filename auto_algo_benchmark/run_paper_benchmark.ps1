# Full paper-oriented benchmark: LLaMEA + LLM4AD, 3 tasks x 5 seeds, with live status page + exports.
# Usage (from auto_algo_benchmark):
#   .\run_paper_benchmark.ps1                 # skip seeds already completed successfully
#   .\run_paper_benchmark.ps1 -Full           # re-run all 30 experiments (--no-skip), for a clean paper table
#   .\run_paper_benchmark.ps1 -Port 9000
param(
    [int]$Port = 8765,
    [switch]$Full
)
# Do not use "Stop" here: Python logs INFO to stderr; PowerShell would treat stderr as errors when piping.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$repoRoot = Split-Path $PSScriptRoot -Parent
$results = Join-Path $repoRoot "benchmark_results"
$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "paper_benchmark_$stamp.log"

Write-Host "Results dir: $results"
Write-Host "Log file:    $logFile"
Write-Host "Starting status server on http://127.0.0.1:$Port/  (Refresh in browser to check progress)"
Write-Host ""

$statusProc = Start-Process -FilePath "python" -ArgumentList @(
    "-u", "check_benchmark_status.py",
    "--serve", "--output-dir", $results,
    "--port", "$Port"
) -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden

if ($Full) {
    Write-Host "Mode: FULL re-run (all 30 experiments)."
} else {
    Write-Host "Mode: resume (skip completed valid seeds). Use -Full for a clean-slate paper run."
}

$code = 0
try {
    # Run via cmd so Python's stderr (logging) is merged inside cmd and does not become PowerShell ErrorRecords.
    if ($Full) {
        $inner = "cd /d `"$PSScriptRoot`" && python -u run_benchmark.py --config configs\benchmark_paper.json --no-skip -v 2>&1"
    } else {
        $inner = "cd /d `"$PSScriptRoot`" && python -u run_benchmark.py --config configs\benchmark_paper.json -v 2>&1"
    }
    cmd /c $inner | Tee-Object -FilePath $logFile
    $code = $LASTEXITCODE
} finally {
    Stop-Process -Id $statusProc.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Exporting paper tables (dedupe-last + valid-only)..."
python analyze_benchmark.py --results-dir $results --dedupe-last --valid-only --export-paper (Join-Path $results "paper_summary.csv")
python analyze_benchmark.py --results-dir $results --dedupe-last --valid-only --export (Join-Path $results "paper_all_runs.csv") --error-max-chars 800

if ($code -ne 0) {
    exit $code
}
Write-Host "Done. See paper_summary.csv and paper_all_runs.csv in benchmark_results."
