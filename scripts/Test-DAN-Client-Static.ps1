# Runs 3 local stage workers on an explicit route and checks that every client gives
# identical output. -BaselineDir adds reports saved earlier (e.g. a pre-refactor build).
# Exit code 0 only when every report agrees.
#   -Transport hub     client sends to each stage (static coordinator and dan-client)
#   -Transport ring    dan-client sets up a direct ring A -> B -> C -> client over plain TCP
#   -Transport libp2p  same ring, with every hop through dan-sidecar (-Sidecar)
# In ring modes dan-client runs with --require-direct: it fails if any intermediate
# activation reaches it.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$Model,
    [string]$Manifest = (Join-Path $PSScriptRoot '..\config\provider-owned-qwen2.5-0.5b-q4km.json'),
    [Parameter(Mandatory)][string]$OutDir,
    [string]$BaselineDir,
    [ValidateSet('hub', 'ring', 'libp2p')][string]$Transport = 'hub',
    [string]$Sidecar = (Join-Path $BuildDir 'sidecar\dan-sidecar.exe'),
    [int[]]$Boundaries = @(0, 8, 16, 24),
    [int]$BasePort = 7101,
    [int]$Tokens = 32,
    # libp2p only: tell stage B to expect the client's PeerID instead of stage A's. Passes only
    # if B rejects A's authenticated connection and dan-client fails the route cleanly.
    [switch]$WrongPredecessor
)
if ($WrongPredecessor -and $Transport -ne 'libp2p') { throw '-WrongPredecessor needs -Transport libp2p' }

$ErrorActionPreference = 'Stop'
$bin = Join-Path $BuildDir 'Release'
# llama.cpp/ggml DLLs are built into bin\Release.
$dlls = Join-Path $BuildDir 'bin\Release'
if (Test-Path -LiteralPath $dlls) { $env:PATH = "$((Resolve-Path $dlls).Path);$env:PATH" }
$worker = Join-Path $bin 'dan-stage-worker.exe'
$clients = [ordered]@{ 'dan-client' = Join-Path $bin 'dan-client.exe' }
if ($Transport -eq 'hub') {
    $clients = [ordered]@{
        coordinator = Join-Path $bin 'dan-provider-owned-coordinator.exe'
        'dan-client' = Join-Path $bin 'dan-client.exe'
    }
}
$required = @($worker, $Model, $Manifest)
if ($Transport -eq 'libp2p') { $required += $Sidecar }
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$prompts = @('The capital of France is', 'Name three colors of the rainbow.')
$stageCount = $Boundaries.Count - 1
$controlPort = { param($i) $BasePort + $i }
$ringPort = { param($i) $BasePort + 10 + $i }
$returnPort = $BasePort + 20
$p2pPort = { param($i) $BasePort + 30 + $i }   # stage sidecars; client sidecar uses index $stageCount
$proxyPort = { param($i) $BasePort + 40 + $i }
$forwardPort = { param($i) $BasePort + 50 + $i }

$processes = @()
function Wait-ForFile([string]$Path, [string]$Pattern, $Process, [string]$What) {
    $deadline = (Get-Date).AddMinutes(3)
    while (-not ((Test-Path $Path) -and (Select-String -Path $Path -Pattern $Pattern -Quiet))) {
        if ($Process.HasExited) { throw "$What exited; see $Path" }
        if ((Get-Date) -gt $deadline) { throw "$What did not start; see $Path" }
        Start-Sleep -Milliseconds 250
    }
}

try {
    $peerIds = @()
    $clientPeer = ''
    if ($Transport -eq 'libp2p') {
        $keys = Join-Path $OutDir 'keys'
        New-Item -ItemType Directory -Force -Path $keys | Out-Null
        for ($index = 0; $index -le $stageCount; ++$index) {
            $name = if ($index -eq $stageCount) { 'client' } else { "stage-$index" }
            $id = (& $Sidecar -key (Join-Path $keys "$name.key") -id).Trim()
            if ($LASTEXITCODE -ne 0 -or -not $id) { throw "could not create $name identity" }
            if ($index -eq $stageCount) { $clientPeer = $id } else { $peerIds += $id }
        }
        for ($index = 0; $index -lt $stageCount; ++$index) {
            $ready = Join-Path $OutDir "sidecar-$index.ready"
            $sidecarProcess = Start-Process -FilePath $Sidecar -PassThru -NoNewWindow `
                -RedirectStandardOutput (Join-Path $OutDir "sidecar-$index.out") `
                -RedirectStandardError (Join-Path $OutDir "sidecar-$index.err") -ArgumentList @(
                    '-key', (Join-Path $keys "stage-$index.key"),
                    '-listen', "/ip4/127.0.0.1/tcp/$(& $p2pPort $index)",
                    '-inbound', "127.0.0.1:$(& $controlPort $index)", '-allow-any',
                    '-ring-inbound', "127.0.0.1:$(& $ringPort $index)",
                    '-ring-proxy', "127.0.0.1:$(& $proxyPort $index)",
                    '-ready-file', $ready, '-log', (Join-Path $OutDir "sidecar-$index.log"))
            $processes += $sidecarProcess
            Wait-ForFile $ready '.' $sidecarProcess "sidecar $index"
        }
    }

    for ($index = 0; $index -lt $stageCount; ++$index) {
        $log = Join-Path $OutDir "stage-$index.log"
        $arguments = @('--model', "`"$Model`"", '--stage-start', $Boundaries[$index],
            '--stage-end', $Boundaries[$index + 1], '--host', '127.0.0.1',
            '--port', (& $controlPort $index))
        if ($Transport -ne 'hub') { $arguments += @('--ring-listen', "127.0.0.1:$(& $ringPort $index)") }
        if ($Transport -eq 'libp2p') {
            $arguments += @('--peer-header', '--ring-proxy', "127.0.0.1:$(& $proxyPort $index)")
        }
        $workerProcess = Start-Process -FilePath $worker -PassThru -NoNewWindow `
            -RedirectStandardError $log -RedirectStandardOutput "$log.out" -ArgumentList $arguments
        $processes += $workerProcess
        Wait-ForFile $log 'listening on' $workerProcess "stage $index"
    }

    # The explicit route. Nothing below is discovered yet.
    $routeArguments = @()
    for ($index = 0; $index -lt $stageCount; ++$index) {
        $control = if ($Transport -eq 'libp2p') { & $forwardPort $index } else { & $controlPort $index }
        $routeArguments += @('--provider', "127.0.0.1:$control")
    }
    if ($Transport -eq 'ring') {
        for ($index = 1; $index -lt $stageCount; ++$index) {
            $routeArguments += @('--ring-target', "127.0.0.1:$(& $ringPort $index)")
        }
        $routeArguments += @('--ring-return', "127.0.0.1:$returnPort", '--require-direct')
    }
    if ($Transport -eq 'libp2p') {
        $clientAddress = "/ip4/127.0.0.1/tcp/$(& $p2pPort $stageCount)/p2p/$clientPeer"
        $sidecarArguments = @('-key', (Join-Path $OutDir 'keys\client.key'),
            '-listen', "/ip4/127.0.0.1/tcp/$(& $p2pPort $stageCount)",
            '-ring-inbound', "127.0.0.1:$returnPort",
            '-ready-file', (Join-Path $OutDir 'sidecar-client.ready'),
            '-log', (Join-Path $OutDir 'sidecar-client.log'))
        for ($index = 0; $index -lt $stageCount; ++$index) {
            $sidecarArguments += @('-forward', ("127.0.0.1:$(& $forwardPort $index)=" +
                "/ip4/127.0.0.1/tcp/$(& $p2pPort $index)/p2p/$($peerIds[$index])"))
            $claimed = if ($WrongPredecessor -and $index -eq 0) { $clientPeer } else { $peerIds[$index] }
            $routeArguments += @('--peer-id', $claimed)
            if ($index -gt 0) {
                $routeArguments += @('--ring-target',
                    "/ip4/127.0.0.1/tcp/$(& $p2pPort $index)/p2p/$($peerIds[$index])")
            }
        }
        $sidecarProcess = Start-Process -FilePath $Sidecar -PassThru -NoNewWindow `
            -RedirectStandardOutput (Join-Path $OutDir 'sidecar-client.out') `
            -RedirectStandardError (Join-Path $OutDir 'sidecar-client.err') -ArgumentList $sidecarArguments
        $processes += $sidecarProcess
        Wait-ForFile (Join-Path $OutDir 'sidecar-client.ready') '.' $sidecarProcess 'client sidecar'
        $routeArguments += @('--ring-return', "127.0.0.1:$returnPort",
            '--ring-return-target', $clientAddress, '--require-direct')
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
            $arguments = @('--manifest', $Manifest) + $promptArguments +
                @('--requests', '2', '--tokens', "$Tokens", '--report', $report)
            # The coordinator's static mode takes only the hub route.
            $arguments += $(if ($name -eq 'coordinator') { $routeArguments[0..(2 * $stageCount - 1)] } else { $routeArguments })
            if ($mode -eq 'persistent') { $arguments += '--persistent' }
            # A plain array, not splatting: PowerShell 5.1 splatting mangles '--persistent'.
            # 'Continue' because 5.1 turns a native program's stderr lines into errors.
            $ErrorActionPreference = 'Continue'
            & $clients[$name] $arguments 2> (Join-Path $OutDir "$name-$mode.err") |
                Out-File -Encoding utf8 (Join-Path $OutDir "$name-$mode.out")
            $exitCode = $LASTEXITCODE
            $ErrorActionPreference = 'Stop'
            if ($WrongPredecessor) {
                $rejected = Select-String -Path (Join-Path $OutDir 'stage-1.log') `
                    -Pattern 'rejected unexpected predecessor peer' -Quiet
                if ($exitCode -eq 0 -or -not $rejected) { throw 'a wrong predecessor was not rejected' }
                Write-Host "PASS wrong predecessor rejected: $(Get-Content (Join-Path $OutDir "$name-$mode.err") -Tail 1)"
                break
            }
            if ($exitCode -ne 0) { throw "$name ($mode) failed; see $OutDir\$name-$mode.err" }
            Write-Host "ran $name ($mode): $((Select-String -Path (Join-Path $OutDir "$name-$mode.out") -Pattern '^mode=').Line)"
        }
    }
} finally {
    foreach ($process in $processes) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    }
}

if ($WrongPredecessor) { exit 0 }
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
