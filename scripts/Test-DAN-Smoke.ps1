[CmdletBinding()]
param(
    [string]$ModelPath = 'build-provider-owned-cuda\gateway-cache\qwen2.5-0.5b-instruct-q4-k-m-0-24.gguf',
    [string]$Manifest = 'config\provider-owned-qwen2.5-0.5b-q4km.json',
    [string]$WorkerExe = 'build-provider-owned-cuda\Release\dan-stage-worker.exe',
    [string]$CoordinatorExe = 'build-provider-owned-cuda\Release\dan-provider-owned-coordinator.exe',
    [string]$Port = '50101',
    [int]$LoadWaitSeconds = 8,
    [string]$ReportPath = 'build-provider-owned-cuda\smoke-report.json',
    [string]$DllDir = 'build-provider-owned-cuda\bin\Release'
)

# One-command end-to-end check: load the real provider-owned engine on this
# GPU, run one request through the coordinator, and verify a real completion
# came back. Not a substitute for the ring/soak/stranger-machine gates in
# docs/PROGRESS.md -- this only proves the local single-provider
# path still works after a change.

$ErrorActionPreference = 'Stop'

foreach ($path in @($ModelPath, $Manifest, $WorkerExe, $CoordinatorExe)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}

$reportDir = Split-Path -Parent $ReportPath
if ($reportDir -and -not (Test-Path -LiteralPath $reportDir)) {
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
}
$workerLog = Join-Path $reportDir 'smoke-worker.out.log'
$workerErr = Join-Path $reportDir 'smoke-worker.err.log'

if (Test-Path -LiteralPath $DllDir) {
    $env:PATH = "$((Resolve-Path -LiteralPath $DllDir).Path);$env:PATH"
}

$worker = Start-Process -FilePath (Resolve-Path -LiteralPath $WorkerExe) -ArgumentList @(
    '--model', (Resolve-Path -LiteralPath $ModelPath),
    '--stage-start', '0', '--stage-end', '24',
    '--host', '127.0.0.1', '--port', $Port,
    '--ctx', '512', '--gpu-layers', '999', '--max-sessions', '4'
) -NoNewWindow -PassThru -RedirectStandardOutput $workerLog -RedirectStandardError $workerErr

try {
    Start-Sleep -Seconds $LoadWaitSeconds
    if ($worker.HasExited) {
        throw "Worker exited early (code $($worker.ExitCode)); see $workerErr"
    }

    & $CoordinatorExe --manifest $Manifest --provider-a "127.0.0.1:$Port" `
        --prompt 'The capital of France is' --tokens 16 --requests 1 `
        --report $ReportPath --shutdown-workers
    $coordinatorExit = $LASTEXITCODE

    if ($coordinatorExit -ne 0) {
        throw "Coordinator exited with code $coordinatorExit"
    }
    if (-not (Test-Path -LiteralPath $ReportPath)) {
        throw "No report written to $ReportPath"
    }

    $report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
    if ($report.tokens_generated -le 0) {
        throw 'No tokens were generated'
    }
    if (-not $report.outputs -or -not $report.outputs[0]) {
        throw 'Coordinator returned an empty completion'
    }

    Write-Host "DAN smoke test passed: $($report.tokens_generated) tokens generated, output: '$($report.outputs[0])'"
} finally {
    if (-not $worker.HasExited) {
        Start-Sleep -Seconds 1
        if (-not $worker.HasExited) { Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue }
    }
}
