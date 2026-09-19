# Dynamic placement on 3 local serve-mode workers. dan-client gets only candidate control
# addresses; it plans the stages, reserves and assigns the workers, and links a direct ring
# (client -> A -> B -> C -> client). Output must match reports in -BaselineDir.
#   -Transport direct   plain TCP between processes
#   -Transport libp2p   every hop through dan-sidecar
#   -LeaseChecks        first run engine/tests/lease_integration.py against worker 0 (direct)
#   -Race               also start two clients at once: exactly one may win
# Workers keep their range caches in -CacheDir between runs.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$OutDir,
    [string]$BaselineDir,
    [string]$Manifest = (Join-Path $PSScriptRoot '..\config\provider-owned-qwen2.5-0.5b-q4km.json'),
    [ValidateSet('direct', 'libp2p')][string]$Transport = 'direct',
    [string]$Sidecar = (Join-Path $BuildDir 'sidecar\dan-sidecar.exe'),
    [string]$CacheDir = (Join-Path $BuildDir 'placement-cache'),
    [int[]]$OfferedMib = @(2560, 2048, 1536),
    [int]$MinStages = 3,
    [int]$BasePort = 7201,
    [int]$Tokens = 32,
    # Speculative decoding: every worker also offers this (smaller) model, and the client
    # asks the first stage to draft with it.
    [string]$DraftManifest,
    # How activations cross between stages (dan-client --activations).
    [ValidateSet('f32', 'f16', 'fp8')][string]$Activations = 'f32',
    [switch]$LeaseChecks,
    [switch]$Race
)

$ErrorActionPreference = 'Stop'
$bin = Join-Path $BuildDir 'Release'
$dlls = Join-Path $BuildDir 'bin\Release'
if (Test-Path -LiteralPath $dlls) { $env:PATH = "$((Resolve-Path $dlls).Path);$env:PATH" }
$worker = Join-Path $bin 'dan-stage-worker.exe'
$client = Join-Path $bin 'dan-client.exe'
$required = @($worker, $client, $Manifest)
if ($Transport -eq 'libp2p') { $required += $Sidecar }
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
if ($LeaseChecks -and $Transport -ne 'direct') { throw '-LeaseChecks needs -Transport direct' }
New-Item -ItemType Directory -Force -Path $OutDir, $CacheDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
$modelSha = (Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json).artifact_sha256.ToLower()
$prompts = @('The capital of France is', 'Name three colors of the rainbow.')
$count = $OfferedMib.Count
$controlPort = { param($i) $BasePort + $i }
$ringPort = { param($i) $BasePort + 10 + $i }
$returnPort = $BasePort + 20
$p2pPort = { param($i) $BasePort + 30 + $i }   # client sidecar uses index $count
$proxyPort = { param($i) $BasePort + 40 + $i }
$forwardPort = { param($i) $BasePort + 50 + $i }

function Wait-ForLog([string]$Path, [string]$Pattern, $Process, [string]$What) {
    $deadline = (Get-Date).AddMinutes(5)
    while (-not ((Test-Path $Path) -and (Select-String -Path $Path -Pattern $Pattern -Quiet))) {
        if ($Process.HasExited) { throw "$What exited; see $Path" }
        if ((Get-Date) -gt $deadline) { throw "$What did not start; see $Path" }
        Start-Sleep -Milliseconds 250
    }
}

function Start-Client([string]$Name, [string[]]$Arguments) {
    # Start-Process joins arguments with spaces, so quote each one.
    $quoted = $Arguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
    $process = Start-Process -FilePath $client -PassThru -NoNewWindow -ArgumentList $quoted `
        -RedirectStandardOutput (Join-Path $OutDir "$Name.out") `
        -RedirectStandardError (Join-Path $OutDir "$Name.err")
    # PowerShell 5.1 only reports ExitCode if the handle was opened while the process ran.
    $null = $process.Handle
    $process
}

$processes = @()
try {
    $peerIds = @()
    $clientPeer = ''
    if ($Transport -eq 'libp2p') {
        $keys = Join-Path $OutDir 'keys'
        New-Item -ItemType Directory -Force -Path $keys | Out-Null
        for ($index = 0; $index -le $count; ++$index) {
            $name = if ($index -eq $count) { 'client' } else { "worker-$index" }
            $id = (& $Sidecar -key (Join-Path $keys "$name.key") -id).Trim()
            if ($LASTEXITCODE -ne 0 -or -not $id) { throw "could not create $name identity" }
            if ($index -eq $count) { $clientPeer = $id } else { $peerIds += $id }
        }
        for ($index = 0; $index -lt $count; ++$index) {
            $ready = Join-Path $OutDir "sidecar-$index.ready"
            $sidecarProcess = Start-Process -FilePath $Sidecar -PassThru -NoNewWindow `
                -RedirectStandardOutput (Join-Path $OutDir "sidecar-$index.out") `
                -RedirectStandardError (Join-Path $OutDir "sidecar-$index.err") -ArgumentList @(
                    '-key', (Join-Path $keys "worker-$index.key"),
                    '-listen', "/ip4/127.0.0.1/tcp/$(& $p2pPort $index)",
                    '-inbound', "127.0.0.1:$(& $controlPort $index)", '-allow-any',
                    '-ring-inbound', "127.0.0.1:$(& $ringPort $index)",
                    '-ring-proxy', "127.0.0.1:$(& $proxyPort $index)",
                    '-ready-file', $ready, '-log', (Join-Path $OutDir "sidecar-$index.log"))
            $processes += $sidecarProcess
            Wait-ForLog $ready '.' $sidecarProcess "sidecar $index"
        }
    }

    # Workers: no stage range, only resources and a model catalog.
    for ($index = 0; $index -lt $count; ++$index) {
        $log = Join-Path $OutDir "worker-$index.log"
        $arguments = @('--control-listen', "127.0.0.1:$(& $controlPort $index)",
            '--ring-listen', "127.0.0.1:$(& $ringPort $index)",
            '--catalog', "`"$Manifest`"", '--cache-dir', "`"$(Join-Path $CacheDir "worker-$index")`"",
            '--provider-id', "worker-$index", '--gpu', 'CPU', '--vram-mib', $OfferedMib[$index],
            '--max-sessions', '1')
        if ($DraftManifest) { $arguments += @('--catalog', "`"$DraftManifest`"") }
        if ($Transport -eq 'libp2p') {
            $arguments += @('--peer-header', '--ring-proxy', "127.0.0.1:$(& $proxyPort $index)",
                '--ring-target', "/ip4/127.0.0.1/tcp/$(& $p2pPort $index)/p2p/$($peerIds[$index])")
        }
        $workerProcess = Start-Process -FilePath $worker -PassThru -NoNewWindow `
            -RedirectStandardError $log -RedirectStandardOutput "$log.out" -ArgumentList $arguments
        $processes += $workerProcess
        Wait-ForLog $log 'serving placement requests' $workerProcess "worker $index"
    }

    if ($LeaseChecks) {
        $ErrorActionPreference = 'Continue'
        python (Join-Path $PSScriptRoot '..\engine\tests\lease_integration.py') `
            "127.0.0.1:$(& $controlPort 0)" $modelSha 2>&1 | Tee-Object (Join-Path $OutDir 'lease.out')
        $leaseExit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($leaseExit -ne 0) { throw 'lease checks failed' }
    }

    # Candidates only: addresses (and, for libp2p, the PeerID each forward reaches).
    $clientArguments = @('--manifest', $Manifest)
    if ($DraftManifest) { $clientArguments += @('--manifest', $DraftManifest, '--speculate') }
    if ($Activations -ne 'f32') { $clientArguments += @('--activations', $Activations) }
    $clientArguments += @('--min-stages', "$MinStages",
        '--ring-return', "127.0.0.1:$returnPort", '--require-direct',
        '--requests', '2', '--tokens', "$Tokens")
    foreach ($prompt in $prompts) { $clientArguments += @('--prompt', $prompt) }
    if ($Transport -eq 'libp2p') {
        $sidecarArguments = @('-key', (Join-Path $OutDir 'keys\client.key'),
            '-listen', "/ip4/127.0.0.1/tcp/$(& $p2pPort $count)",
            '-ring-inbound', "127.0.0.1:$returnPort",
            '-ready-file', (Join-Path $OutDir 'sidecar-client.ready'),
            '-log', (Join-Path $OutDir 'sidecar-client.log'))
        for ($index = 0; $index -lt $count; ++$index) {
            $sidecarArguments += @('-forward', ("127.0.0.1:$(& $forwardPort $index)=" +
                "/ip4/127.0.0.1/tcp/$(& $p2pPort $index)/p2p/$($peerIds[$index])"))
            $clientArguments += @('--candidate', "127.0.0.1:$(& $forwardPort $index)",
                '--candidate-peer', $peerIds[$index])
        }
        $sidecarProcess = Start-Process -FilePath $Sidecar -PassThru -NoNewWindow `
            -RedirectStandardOutput (Join-Path $OutDir 'sidecar-client.out') `
            -RedirectStandardError (Join-Path $OutDir 'sidecar-client.err') -ArgumentList $sidecarArguments
        $processes += $sidecarProcess
        Wait-ForLog (Join-Path $OutDir 'sidecar-client.ready') '.' $sidecarProcess 'client sidecar'
        $clientArguments += @('--ring-return-target', "/ip4/127.0.0.1/tcp/$(& $p2pPort $count)/p2p/$clientPeer")
    } else {
        for ($index = 0; $index -lt $count; ++$index) {
            $clientArguments += @('--candidate', "127.0.0.1:$(& $controlPort $index)")
        }
    }

    foreach ($mode in @('once', 'persistent')) {
        $arguments = $clientArguments + @('--report', (Join-Path $OutDir "dan-client-$mode.json"))
        if ($mode -eq 'persistent') { $arguments += '--persistent' }
        $run = Start-Client "dan-client-$mode" $arguments
        $run.WaitForExit()
        if ($run.ExitCode -ne 0) { throw "dan-client ($mode) failed; see $OutDir\dan-client-$mode.err" }
        $summary = (Select-String -Path (Join-Path $OutDir "dan-client-$mode.out") -Pattern '^(route=|mode=)').Line
        Write-Host "ran dan-client ($mode):"
        $summary | ForEach-Object { Write-Host "  $_" }
    }

    if ($Race) {
        # Two clients at once need all three workers each: one wins, the other fails cleanly.
        $racers = @()
        foreach ($name in 'race-a', 'race-b') {
            $port = if ($name -eq 'race-a') { $returnPort } else { $returnPort + 1 }
            $arguments = $clientArguments -replace "^127\.0\.0\.1:$returnPort$", "127.0.0.1:$port"
            $racers += Start-Client $name $arguments
        }
        $racers | ForEach-Object { $_.WaitForExit() }
        $winners = @($racers | Where-Object { $_.ExitCode -eq 0 }).Count
        foreach ($name in 'race-a', 'race-b') {
            Write-Host "  $name`: $((Get-Content (Join-Path $OutDir "$name.err") -Tail 1))"
        }
        if ($Transport -eq 'libp2p') { Write-Host '  (libp2p race shares one client ring inbound; the loser may also fail there)' }
        if ($winners -ne 1) { throw "race: expected exactly one winner, got $winners" }
        Write-Host 'PASS race: exactly one client placed the route'
        $again = Start-Client 'after-race' ($clientArguments + @('--report', (Join-Path $OutDir 'after-race.json')))
        $again.WaitForExit()
        if ($again.ExitCode -ne 0) { throw 'workers were not free again after the race' }
        Write-Host 'PASS after the race every worker was free again'
    }
} finally {
    foreach ($process in $processes) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    }
}

$failed = $false
foreach ($mode in @('once', 'persistent')) {
    $reports = @(Get-ChildItem -Path $OutDir -Filter "dan-client-$mode.json")
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
Write-Host 'Dynamic placement produced identical output.'
