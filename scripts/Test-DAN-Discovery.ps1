# Coordinatorless end to end on one machine: a bootstrap DHT node, 3 serve-mode workers
# with sidecars, and a client sidecar. dan-client gets only the model manifest and its own
# local sidecar API; it discovers the workers, plans, reserves, assigns and runs over the
# direct ring. Then:
#   1. the bootstrap node is stopped and a new request still works (it was only an entry point);
#   2. a worker is stopped; discovery drops it and a 2-stage route still works.
# Every output must match the reports in -BaselineDir.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$OutDir,
    [string]$BaselineDir,
    [string]$Manifest = (Join-Path $PSScriptRoot '..\config\provider-owned-qwen2.5-0.5b-q4km.json'),
    [string]$Sidecar = (Join-Path $BuildDir 'sidecar\dan-sidecar.exe'),
    [string]$CacheDir = (Join-Path $BuildDir 'placement-cache'),
    [int[]]$OfferedMib = @(2560, 2048, 1536),
    [int]$BasePort = 7401,
    [int]$Tokens = 32
)

$ErrorActionPreference = 'Stop'
$bin = Join-Path $BuildDir 'Release'
$dlls = Join-Path $BuildDir 'bin\Release'
if (Test-Path -LiteralPath $dlls) { $env:PATH = "$((Resolve-Path $dlls).Path);$env:PATH" }
$worker = Join-Path $bin 'dan-stage-worker.exe'
$client = Join-Path $bin 'dan-client.exe'
foreach ($file in @($worker, $client, $Manifest, $Sidecar)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
New-Item -ItemType Directory -Force -Path $OutDir, $CacheDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
$modelSha = (Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json).artifact_sha256.ToLower()
$count = $OfferedMib.Count
$controlPort = { param($i) $BasePort + $i }
$ringPort = { param($i) $BasePort + 10 + $i }
$p2pPort = { param($i) $BasePort + 30 + $i }
$proxyPort = { param($i) $BasePort + 40 + $i }
$returnPort = $BasePort + 20
$clientP2pPort = $BasePort + 50
$bootstrapPort = $BasePort + 51
$apiPort = $BasePort + 52
$keys = Join-Path $OutDir 'keys'
New-Item -ItemType Directory -Force -Path $keys | Out-Null

function Wait-ForLog([string]$Path, [string]$Pattern, $Process, [string]$What) {
    $deadline = (Get-Date).AddMinutes(5)
    while (-not ((Test-Path $Path) -and (Select-String -Path $Path -Pattern $Pattern -Quiet))) {
        if ($Process.HasExited) { throw "$What exited; see $Path" }
        if ((Get-Date) -gt $deadline) { throw "$What did not start; see $Path" }
        Start-Sleep -Milliseconds 250
    }
}

function Start-Sidecar([string]$Name, [string[]]$Arguments) {
    $ready = Join-Path $OutDir "$Name.ready"
    $process = Start-Process -FilePath $Sidecar -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $OutDir "$Name.out") `
        -RedirectStandardError (Join-Path $OutDir "$Name.err") `
        -ArgumentList ($Arguments + @('-ready-file', $ready, '-log', (Join-Path $OutDir "$Name.log")))
    Wait-ForLog $ready '.' $process $Name
    $process
}

function Get-PeerId([string]$Name) {
    $id = (& $Sidecar -key (Join-Path $keys "$Name.key") -id).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $id) { throw "could not create $Name identity" }
    $id
}

# What the client's own sidecar currently finds for the model.
function Get-Candidates {
    $tcp = [System.Net.Sockets.TcpClient]::new('127.0.0.1', $apiPort)
    try {
        $stream = $tcp.GetStream()
        $writer = [System.IO.StreamWriter]::new($stream)
        $writer.Write("DAN-CANDIDATES/1 $modelSha`n")
        $writer.Flush()
        $text = [System.IO.StreamReader]::new($stream).ReadToEnd()
    } finally { $tcp.Close() }
    @($text -split "`n" | Where-Object { $_ -like 'CANDIDATE *' })
}

function Wait-ForCandidates([int]$Want) {
    $deadline = (Get-Date).AddSeconds(120)
    while ($true) {
        $found = Get-Candidates
        if ($found.Count -eq $Want) { return $found }
        if ((Get-Date) -gt $deadline) { throw "wanted $Want candidates, found $($found.Count)" }
        Start-Sleep -Seconds 2
    }
}

function Invoke-Client([string]$Name, [string[]]$Extra) {
    # The whole client configuration: model manifest + local sidecar API. No worker addresses.
    $arguments = @('--manifest', $Manifest, '--discover', "127.0.0.1:$apiPort", '--require-direct',
        '--prompt', 'The capital of France is', '--prompt', 'Name three colors of the rainbow.',
        '--requests', '2', '--tokens', "$Tokens", '--report', (Join-Path $OutDir "$Name.json")) + $Extra
    $quoted = $arguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
    $process = Start-Process -FilePath $client -PassThru -NoNewWindow -ArgumentList $quoted `
        -RedirectStandardOutput (Join-Path $OutDir "$Name.out") `
        -RedirectStandardError (Join-Path $OutDir "$Name.err")
    $null = $process.Handle
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "dan-client $Name failed; see $OutDir\$Name.err" }
    Write-Host "ran dan-client $Name ($($arguments -join ' ' -replace [regex]::Escape($OutDir), '<out>'))"
    (Select-String -Path (Join-Path $OutDir "$Name.out") -Pattern '^(discovered|route=|mode=)').Line |
        ForEach-Object { Write-Host "  $_" }
}

$processes = @{}
try {
    $bootstrap = Start-Sidecar 'bootstrap' @('-key', (Join-Path $keys 'bootstrap.key'),
        '-listen', "/ip4/127.0.0.1/tcp/$bootstrapPort", '-dht', 'server')
    $processes['bootstrap'] = @($bootstrap)
    $bootstrapAddress = (Get-Content (Join-Path $OutDir 'bootstrap.ready') |
        Where-Object { $_ -like '/ip4/*/tcp/*' } | Select-Object -First 1)
    Write-Host "bootstrap $bootstrapAddress"

    for ($index = 0; $index -lt $count; ++$index) {
        $name = "worker-$index"
        $peer = Get-PeerId $name
        $status = Join-Path $OutDir "$name.status.json"
        $sidecarProcess = Start-Sidecar "$name-sidecar" @('-key', (Join-Path $keys "$name.key"),
            '-listen', "/ip4/127.0.0.1/tcp/$(& $p2pPort $index)",
            '-dht', 'server', '-bootstrap', $bootstrapAddress, '-status-file', $status,
            '-inbound', "127.0.0.1:$(& $controlPort $index)", '-allow-any',
            '-ring-inbound', "127.0.0.1:$(& $ringPort $index)",
            '-ring-proxy', "127.0.0.1:$(& $proxyPort $index)")
        $log = Join-Path $OutDir "$name.log"
        $workerProcess = Start-Process -FilePath $worker -PassThru -NoNewWindow `
            -RedirectStandardError $log -RedirectStandardOutput "$log.out" -ArgumentList @(
                '--control-listen', "127.0.0.1:$(& $controlPort $index)",
                '--ring-listen', "127.0.0.1:$(& $ringPort $index)",
                '--peer-header', '--ring-proxy', "127.0.0.1:$(& $proxyPort $index)",
                '--ring-target', "/p2p/$peer", '--status-file', "`"$status`"",
                '--catalog', "`"$Manifest`"", '--cache-dir', "`"$(Join-Path $CacheDir $name)`"",
                '--provider-id', $name, '--gpu', 'CPU', '--vram-mib', $OfferedMib[$index],
                '--max-sessions', '1')
        $processes[$name] = @($workerProcess, $sidecarProcess)
        Wait-ForLog $log 'serving placement requests' $workerProcess $name
    }

    $clientSidecar = Start-Sidecar 'client-sidecar' @('-key', (Join-Path $keys 'client.key'),
        '-listen', "/ip4/127.0.0.1/tcp/$clientP2pPort", '-dht', 'server',
        '-bootstrap', $bootstrapAddress, '-candidate-api', "127.0.0.1:$apiPort",
        '-ring-inbound', "127.0.0.1:$returnPort")
    $processes['client'] = @($clientSidecar)

    $found = Wait-ForCandidates $count
    Write-Host "PASS discovery found $($found.Count) workers"
    Invoke-Client 'dan-client-once' @('--min-stages', "$count")
    Invoke-Client 'dan-client-persistent' @('--min-stages', "$count", '--persistent')

    # 1. The bootstrap node is only an entry point.
    Stop-Process -Id $bootstrap.Id -Force
    $processes.Remove('bootstrap')
    Start-Sleep -Seconds 2
    $found = Wait-ForCandidates $count
    Write-Host "PASS after stopping the bootstrap node discovery still found $($found.Count) workers"
    Invoke-Client 'after-bootstrap-once' @('--min-stages', "$count")

    # 2. A stopped worker is dropped even while its DHT record is still valid.
    $last = "worker-$($count - 1)"
    $processes[$last] | ForEach-Object { Stop-Process -Id $_.Id -Force }
    $processes.Remove($last)
    $found = Wait-ForCandidates ($count - 1)
    Write-Host "PASS the stopped worker is no longer a candidate ($($found.Count) left)"
    Invoke-Client 'without-worker-once' @('--min-stages', "$($count - 1)")
} finally {
    foreach ($group in $processes.Values) {
        foreach ($process in $group) {
            if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
        }
    }
}

$reference = if ($BaselineDir) { Join-Path $BaselineDir 'coordinator-once.json' } else { Join-Path $OutDir 'dan-client-once.json' }
$failed = $false
$pairs = @(
    @('dan-client-once.json', $reference),
    @('after-bootstrap-once.json', $reference),
    @('without-worker-once.json', $reference))
if ($BaselineDir) { $pairs += ,@('dan-client-persistent.json', (Join-Path $BaselineDir 'coordinator-persistent.json')) }
foreach ($pair in $pairs) {
    $actual = (Get-Content -LiteralPath (Join-Path $OutDir $pair[0]) -Raw | ConvertFrom-Json).outputs
    $expected = (Get-Content -LiteralPath $pair[1] -Raw | ConvertFrom-Json).outputs
    $same = $actual.Count -eq $expected.Count
    for ($index = 0; $same -and $index -lt $actual.Count; ++$index) {
        $same = [string]::Equals($actual[$index], $expected[$index], [StringComparison]::Ordinal)
    }
    if ($same) { Write-Host "PASS $($pair[0]) matches $($pair[1])" }
    else { Write-Host "FAIL $($pair[0]) differs from $($pair[1])"; $failed = $true }
}
if ($failed) { exit 1 }
Write-Host 'Coordinatorless discovery, placement and direct-ring inference produced identical output.'
