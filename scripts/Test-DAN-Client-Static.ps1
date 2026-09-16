# Runs 3 local stage workers and checks that every client gives identical output.
# Clients: the static provider-owned coordinator and dan-client (whichever exist in
# -BuildDir). -BaselineDir adds reports saved earlier, e.g. from a pre-refactor build.
# Exit code 0 only when every report agrees.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$Model,
    [string]$Manifest = (Join-Path $PSScriptRoot '..\config\provider-owned-qwen2.5-0.5b-q4km.json'),
    [Parameter(Mandatory)][string]$OutDir,
    [string]$BaselineDir,
    [int[]]$Boundaries = @(0, 8, 16, 24),
    [int]$BasePort = 7101,
    [int]$Tokens = 32
)

$ErrorActionPreference = 'Stop'
$bin = Join-Path $BuildDir 'Release'
# llama.cpp/ggml DLLs are built into bin\Release.
$dlls = Join-Path $BuildDir 'bin\Release'
if (Test-Path -LiteralPath $dlls) { $env:PATH = "$((Resolve-Path $dlls).Path);$env:PATH" }
$worker = Join-Path $bin 'dan-stage-worker.exe'
$clients = [ordered]@{
    coordinator = Join-Path $bin 'dan-provider-owned-coordinator.exe'
    'dan-client' = Join-Path $bin 'dan-client.exe'
}
foreach ($file in @($worker, $Model, $Manifest)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$prompts = @('The capital of France is', 'Name three colors of the rainbow.')
$providerArguments = @()
$workers = @()
try {
    for ($index = 0; $index -lt $Boundaries.Count - 1; ++$index) {
        $port = $BasePort + $index
        $log = Join-Path $OutDir "stage-$index.log"
        $workers += Start-Process -FilePath $worker -PassThru -NoNewWindow `
            -RedirectStandardError $log -RedirectStandardOutput "$log.out" -ArgumentList @(
                '--model', "`"$Model`"", '--stage-start', $Boundaries[$index],
                '--stage-end', $Boundaries[$index + 1], '--host', '127.0.0.1', '--port', $port)
        $providerArguments += @('--provider', "127.0.0.1:$port")
    }
    for ($index = 0; $index -lt $workers.Count; ++$index) {
        $log = Join-Path $OutDir "stage-$index.log"
        $deadline = (Get-Date).AddMinutes(3)
        while (-not ((Test-Path $log) -and (Select-String -Path $log -Pattern 'listening on' -Quiet))) {
            if ($workers[$index].HasExited) { throw "stage $index exited; see $log" }
            if ((Get-Date) -gt $deadline) { throw "stage $index did not start; see $log" }
            Start-Sleep -Milliseconds 250
        }
    }
    $promptArguments = @()
    foreach ($prompt in $prompts) { $promptArguments += @('--prompt', $prompt) }
    foreach ($name in $clients.Keys) {
        if (-not (Test-Path -LiteralPath $clients[$name] -PathType Leaf)) {
            Write-Host "skip $name (not built)"
            continue
        }
        foreach ($mode in @('once', 'persistent')) {
            $report = Join-Path $OutDir "$name-$mode.json"
            # A plain array, not splatting: PowerShell 5.1 splatting mangles '--persistent'.
            $arguments = @('--manifest', $Manifest) + $providerArguments + $promptArguments +
                @('--requests', '2', '--tokens', "$Tokens", '--report', $report)
            if ($mode -eq 'persistent') { $arguments += '--persistent' }
            & $clients[$name] $arguments | Out-File -Encoding utf8 (Join-Path $OutDir "$name-$mode.out")
            if ($LASTEXITCODE -ne 0) { throw "$name ($mode) failed" }
            Write-Host "ran $name ($mode)"
        }
    }
} finally {
    foreach ($process in $workers) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    }
}

$failed = $false
foreach ($mode in @('once', 'persistent')) {
    $reports = @(Get-ChildItem -Path $OutDir -Filter "*-$mode.json")
    if ($BaselineDir) { $reports += @(Get-ChildItem -Path $BaselineDir -Filter "*-$mode.json") }
    if ($reports.Count -lt 2) { Write-Host "FAIL $mode`: fewer than two reports"; $failed = $true; continue }
    $reference = (Get-Content -LiteralPath $reports[0].FullName -Raw | ConvertFrom-Json).outputs
    foreach ($report in $reports) {
        $outputs = (Get-Content -LiteralPath $report.FullName -Raw | ConvertFrom-Json).outputs
        $same = $outputs.Count -eq $reference.Count
        for ($index = 0; $same -and $index -lt $outputs.Count; ++$index) {
            $same = [string]::Equals($outputs[$index], $reference[$index], [StringComparison]::Ordinal)
        }
        if ($same) { Write-Host "PASS $mode $($report.FullName)" }
        else { Write-Host "FAIL $mode $($report.FullName) differs from $($reports[0].FullName)"; $failed = $true }
    }
}
if ($failed) { exit 1 }
Write-Host 'All clients produced identical output.'
