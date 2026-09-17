# One-machine rehearsal of the WAN beta. Every home node runs with -simulate-nat: it only
# accepts and makes relayed connections (except to the relay), so all DAN traffic must go
# through the local infrastructure node, exactly as behind a strict NAT.
#   infra:   dan-sidecar -infra (bootstrap + DHT server + relay)
#   workers: dan-provider network=dht (as a friend would start it)
#   client:  Start-DAN-Client.ps1 (discovery, placement, direct ring through the relay)
# Output must match -BaselineDir. The script also checks that no DAN stream went direct.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$OutDir,
    [string]$BaselineDir,
    [string]$Manifest = (Join-Path $PSScriptRoot '..\config\provider-owned-qwen2.5-0.5b-q4km.json'),
    [string]$Sidecar = (Join-Path $BuildDir 'sidecar\dan-sidecar.exe'),
    [string]$CacheDir = (Join-Path $BuildDir 'placement-cache'),
    # The fake GPU reports 8192 MiB; these reserves leave 2560 / 2048 / 1536 MiB for DAN.
    [int[]]$ReserveMib = @(5632, 6144, 6656),
    [int]$InfraPort = 7501,
    [int]$Tokens = 32
)

$ErrorActionPreference = 'Stop'
$bin = (Resolve-Path (Join-Path $BuildDir 'Release')).Path
$dlls = Join-Path $BuildDir 'bin\Release'
if (Test-Path -LiteralPath $dlls) { $env:PATH = "$((Resolve-Path $dlls).Path);$env:PATH" }
$provider = Join-Path $bin 'dan-provider.exe'
$worker = Join-Path $bin 'dan-stage-worker.exe'
$client = Join-Path $bin 'dan-client.exe'
$fakeGpu = Join-Path $bin 'platform_test.exe'
foreach ($file in @($provider, $worker, $client, $fakeGpu, $Manifest, $Sidecar)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
New-Item -ItemType Directory -Force -Path $OutDir, $CacheDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$CacheDir = (Resolve-Path -LiteralPath $CacheDir).Path
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
$Sidecar = (Resolve-Path -LiteralPath $Sidecar).Path

function Wait-ForLog([string]$Path, [string]$Pattern, $Process, [string]$What, [int]$Seconds = 300) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while (-not ((Test-Path $Path) -and (Select-String -Path $Path -Pattern $Pattern -Quiet))) {
        if ($Process.HasExited) { throw "$What exited; see $Path" }
        if ((Get-Date) -gt $deadline) { throw "$What did not get ready; see $Path" }
        Start-Sleep -Milliseconds 500
    }
}

function Stop-Tree($Process) {
    if ($Process -and -not $Process.HasExited) { & taskkill /T /F /PID $Process.Id *> $null }
}

$processes = @()
try {
    # Public infrastructure. Its announced 1.1.1.1 address is never dialed (everyone is
    # already connected); AutoRelay only builds relay addresses from public relay addresses.
    $infraReady = Join-Path $OutDir 'infra.ready'
    $infra = Start-Process -FilePath $Sidecar -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $OutDir 'infra.out') -RedirectStandardError (Join-Path $OutDir 'infra.err') `
        -ArgumentList @('-infra', '-key', (Join-Path $OutDir 'infra.key'),
            '-listen', "/ip4/127.0.0.1/tcp/$InfraPort", '-announce', '/ip4/1.1.1.1/tcp/9',
            '-ready-file', $infraReady, '-log', (Join-Path $OutDir 'infra.log'))
    $processes += $infra
    Wait-ForLog $infraReady '.' $infra 'infra node' 30
    $bootstrap = Get-Content $infraReady | Where-Object { $_ -like "/ip4/127.0.0.1/tcp/$InfraPort/p2p/*" } |
        Select-Object -First 1
    if (-not $bootstrap) { throw 'infra node reported no loopback address' }
    Write-Host "infra $bootstrap"

    # Workers, started the way a friend starts DAN.
    for ($index = 0; $index -lt $ReserveMib.Count; ++$index) {
        $state = Join-Path $OutDir "worker-$index"
        New-Item -ItemType Directory -Force -Path $state | Out-Null
        $config = Join-Path $state 'provider-dht.conf'
        [IO.File]::WriteAllLines($config, @(
            'network=dht', "bootstrap=$bootstrap", "catalog=$Manifest",
            "stage_worker=$worker", "sidecar=$Sidecar", "nvidia_smi=$fakeGpu",
            "state_dir=$state", "cache_dir=$(Join-Path $CacheDir "worker-$index")",
            "reserve_vram_mib=$($ReserveMib[$index])", 'max_context=512', 'simulate_nat=true'),
            [Text.UTF8Encoding]::new($false))
        $log = Join-Path $state 'provider.log'
        # Worker 0 runs with the node dashboard (as a friend sees it); the others log verbosely.
        $providerArguments = @('--config', "`"$config`"")
        if ($index -gt 0) { $providerArguments += '--verbose' }
        $process = Start-Process -FilePath $provider -PassThru -NoNewWindow `
            -RedirectStandardOutput "$log.out" -RedirectStandardError $log -ArgumentList $providerArguments
        $processes += $process
        if ($index -eq 0) {
            Wait-ForLog "$log.out" 'network connected' $process "worker $index dashboard"
        } else {
            Wait-ForLog $log 'serving placement requests' $process "worker $index"
        }
        Wait-ForLog (Join-Path $state 'logs\sidecar.log') 'advertising dan/model' $process "worker $index advertisement" 120
    }
    Write-Host 'workers joined and advertised'

    $clientScript = Join-Path $PSScriptRoot 'Start-DAN-Client.ps1'
    foreach ($mode in @('once', 'persistent')) {
        $state = Join-Path $OutDir "client-$mode"
        $arguments = @('--min-stages', "$($ReserveMib.Count)", '--require-direct',
            '--prompt', 'The capital of France is', '--prompt', 'Name three colors of the rainbow.',
            '--requests', '2', '--tokens', "$Tokens", '--report', (Join-Path $OutDir "dan-client-$mode.json"))
        if ($mode -eq 'persistent') { $arguments += '--persistent' }
        $ErrorActionPreference = 'Continue'
        & $clientScript -Bootstrap $bootstrap -Manifest $Manifest -Sidecar $Sidecar -Client $client `
            -StateDir $state -SimulateNat -- @arguments *> (Join-Path $OutDir "dan-client-$mode.out")
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($exitCode -ne 0) { throw "client ($mode) failed; see $OutDir\dan-client-$mode.out" }
        Write-Host "ran client ($mode):"
        (Select-String -Path (Join-Path $OutDir "dan-client-$mode.out") -Pattern '^(discovered|route=|timing|mode=)').Line |
            ForEach-Object { Write-Host "  $_" }
        (Select-String -Path (Join-Path $OutDir "dan-client-$mode.out") -Pattern '^request=').Line |
            ForEach-Object { Write-Host "  $($_ -replace ' output=.*$', '')" }
    }

    # A short scripted chat through the same network.
    $ErrorActionPreference = 'Continue'
    $chatInput = Join-Path $OutDir 'chat.in'
    [IO.File]::WriteAllLines($chatInput, @('What is the capital of France? Answer in one word.', '/quit'))
    & $clientScript -Bootstrap $bootstrap -Manifest $Manifest -Sidecar $Sidecar -Client $client `
            -StateDir (Join-Path $OutDir 'client-chat') -SimulateNat -InputFile $chatInput -- --chat --tokens 16 `
            --min-stages "$($ReserveMib.Count)" --require-direct *> (Join-Path $OutDir 'chat.out')
    $chatExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($chatExit -ne 0 -or -not (Select-String -Path (Join-Path $OutDir 'chat.out') -Pattern 'dan > .*Paris' -Quiet)) {
        throw "chat failed; see $OutDir\chat.out"
    }
    Write-Host "chat: $((Select-String -Path (Join-Path $OutDir 'chat.out') -Pattern 'dan > ').Line)"
    Write-Host "dashboard: $((Get-Content (Join-Path $OutDir 'worker-0\provider.log.out') -Tail 1))"
} finally {
    foreach ($process in $processes) { Stop-Tree $process }
}

# Every DAN stream (ring, control, capabilities) between home nodes must have used the relay.
$sidecarLogs = @(Get-ChildItem -Path $OutDir -Recurse -Filter 'sidecar.log')
$streams = @($sidecarLogs | Select-String -Pattern '(connected|accepted) peer=\S+ protocol=/dan/')
$relayed = @($streams | Where-Object { $_.Line -match 'path=relay' })
$direct = @($streams | Where-Object { $_.Line -match 'path=direct' })
Write-Host "DAN streams: $($streams.Count) total, $($relayed.Count) relayed, $($direct.Count) direct"
$failed = $streams.Count -eq 0 -or $direct.Count -gt 0

foreach ($mode in @('once', 'persistent')) {
    $actual = (Get-Content -LiteralPath (Join-Path $OutDir "dan-client-$mode.json") -Raw | ConvertFrom-Json).outputs
    if (-not $BaselineDir) { continue }
    $expected = (Get-Content -LiteralPath (Join-Path $BaselineDir "coordinator-$mode.json") -Raw | ConvertFrom-Json).outputs
    $same = $actual.Count -eq $expected.Count
    for ($index = 0; $same -and $index -lt $actual.Count; ++$index) {
        $same = [string]::Equals($actual[$index], $expected[$index], [StringComparison]::Ordinal)
    }
    if ($same) { Write-Host "PASS $mode output matches the baseline" }
    else { Write-Host "FAIL $mode output differs from the baseline"; $failed = $true }
}
if ($failed) { exit 1 }
Write-Host 'NAT rehearsal passed: discovery, placement and ring inference all went through the relay.'
