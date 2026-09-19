# One-machine rehearsal of persistent replicas (docs/PROJECT.md §9.8). Every home node runs
# with -simulate-nat, so all DAN traffic goes through the local infrastructure node's relay.
#
#   1. Persistence: nodes A (replica=auto), B and C start FREE. No client places anything:
#      A's owner forms A -> B -> C, warms it up and advertises it. Two clients then find the
#      SAME replica, chat, and leave; the workers log no new reserve, load or ring link.
#   2. Failure: B is killed during a long answer. The client gets an error, A and C are
#      released with their layers still loaded, and a replica re-forms (new ID) when B returns,
#      without downloading or reloading anything on A and C.
#   3. Race: four owners start at once and three GPUs are needed: exactly one replica forms,
#      one node stays free, nothing stays reserved.
# Outputs must match -BaselineDir (coordinator-once.json / coordinator-persistent.json).
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$OutDir,
    [string]$BaselineDir,
    [string]$Manifest = (Join-Path $PSScriptRoot '..\config\provider-owned-qwen2.5-0.5b-q4km.json'),
    [string]$Sidecar = (Join-Path $BuildDir 'sidecar\dan-sidecar.exe'),
    [string]$CacheDir = (Join-Path $BuildDir 'placement-cache'),
    [int]$InfraPort = 7521,
    [int]$Tokens = 32,
    [switch]$SkipFailure,
    [switch]$SkipRace,
    # Instead of the three tests: a two-node replica of -SpeculationManifest that drafts with
    # -Manifest, checked against -SpeculationBaselineDir (dan-client-once/persistent.json).
    [switch]$Speculation,
    # Instead of the three tests: a 3-node replica with 2 sessions and two clients at once.
    [switch]$Concurrent,
    [string]$SpeculationManifest = (Join-Path $PSScriptRoot '..\config\provider-owned-qwen2.5-1.5b-q4km.json'),
    [string]$SpeculationBaselineDir
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
$clientScript = Join-Path $PSScriptRoot 'Start-DAN-Client.ps1'
$failures = [System.Collections.Generic.List[string]]::new()

function Check([bool]$Condition, [string]$What) {
    if ($Condition) { Write-Host "PASS $What" } else { Write-Host "FAIL $What"; $script:failures.Add($What) }
}

function Wait-ForLog([string]$Path, [string]$Pattern, $Process, [string]$What, [int]$Seconds = 300,
    [int]$Count = 1) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Count-Lines $Path $Pattern) -lt $Count) {
        if ($Process -and $Process.HasExited) { throw "$What exited; see $Path" }
        if ((Get-Date) -gt $deadline) { throw "$What did not happen in $Seconds s; see $Path" }
        Start-Sleep -Milliseconds 500
    }
}

function Count-Lines([string]$Path, [string]$Pattern) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    return @(Select-String -LiteralPath $Path -Pattern $Pattern).Count
}

function Stop-Tree($Process) {
    if ($Process -and -not $Process.HasExited) {
        # A child that already exited makes taskkill complain; that is fine here.
        $ErrorActionPreference = 'Continue'
        & taskkill /T /F /PID $Process.Id *> $null
    }
}

function Start-Infra([string]$Name, [int]$Port) {
    $ready = Join-Path $OutDir "$Name.ready"
    Remove-Item -LiteralPath $ready -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath $Sidecar -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $OutDir "$Name.out") -RedirectStandardError (Join-Path $OutDir "$Name.err") `
        -ArgumentList @('-infra', '-key', (Join-Path $OutDir "$Name.key"),
            '-listen', "/ip4/127.0.0.1/tcp/$Port", '-announce', '/ip4/1.1.1.1/tcp/9',
            '-ready-file', $ready, '-log', (Join-Path $OutDir "$Name.log"))
    Wait-ForLog $ready '.' $process 'infra node' 30
    $address = Get-Content $ready | Where-Object { $_ -like "/ip4/127.0.0.1/tcp/$Port/p2p/*" } | Select-Object -First 1
    if (-not $address) { throw 'infra node reported no loopback address' }
    return @{ Process = $process; Address = $address }
}

# A node as a friend runs it (dan-provider network=dht), fake 8192 MiB GPU.
function Start-Node([string]$Name, [int]$ReserveMib, [string]$Replica, [string]$Bootstrap,
    [string[]]$Catalog = @($Manifest), [int]$MinStages = 3, [string[]]$Extra = @()) {
    $state = Join-Path $OutDir $Name
    New-Item -ItemType Directory -Force -Path $state | Out-Null
    $config = Join-Path $state 'provider-dht.conf'
    [IO.File]::WriteAllLines($config, @(
        'network=dht', "bootstrap=$Bootstrap") + @($Catalog | ForEach-Object { "catalog=$_" }) + @(
        "stage_worker=$worker", "sidecar=$Sidecar", "nvidia_smi=$fakeGpu",
        "state_dir=$state", "cache_dir=$(Join-Path $CacheDir $Name)",
        "reserve_vram_mib=$ReserveMib", 'max_context=512', 'simulate_nat=true',
        "replica=$Replica", "replica_client=$client", "replica_min_stages=$MinStages") + $Extra,
        [Text.UTF8Encoding]::new($false))
    $log = Join-Path $state 'provider.log'
    $process = Start-Process -FilePath $provider -PassThru -NoNewWindow `
        -RedirectStandardOutput "$log.out" -RedirectStandardError $log `
        -ArgumentList @('--config', "`"$config`"", '--verbose')
    return @{ Name = $Name; Process = $process; State = $state; Log = $log
        Owner = (Join-Path $state 'logs\replica-owner.log')
        Status = (Join-Path $state 'replica-status.json') }
}

function Run-Client([string]$Name, [string]$Bootstrap, [string[]]$Arguments, [string]$InputFile) {
    $out = Join-Path $OutDir "$Name.out"
    $extra = @{}
    if ($InputFile) { $extra.InputFile = $InputFile }
    $ErrorActionPreference = 'Continue'
    & $clientScript -Bootstrap $Bootstrap -Manifest $Manifest -Sidecar $Sidecar -Client $client `
        -StateDir (Join-Path $OutDir $Name) -SimulateNat @extra -- @Arguments *> $out
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    return @{ Exit = $exitCode; Out = $out }
}

function Replica-Of([string]$Out) {
    $line = (Select-String -LiteralPath $Out -Pattern '^replica=(\S+)' | Select-Object -First 1)
    if ($line) { return $line.Matches[0].Groups[1].Value }
    return $null
}

function Worker-Counts($Nodes) {
    $counts = @{}
    foreach ($node in $Nodes) {
        $counts[$node.Name] = @(
            (Count-Lines $node.Log 'reserved layers'),
            (Count-Lines $node.Log 'serving layers'),
            (Count-Lines $node.Log 'connecting to route next hop'),
            (Count-Lines $node.Log 'Downloading required model data'))
    }
    return $counts
}

function Same-Outputs([string]$Actual, [string]$Expected) {
    $a = (Get-Content -LiteralPath $Actual -Raw | ConvertFrom-Json).outputs
    $e = (Get-Content -LiteralPath $Expected -Raw | ConvertFrom-Json).outputs
    if ($a.Count -ne $e.Count) { return $false }
    for ($index = 0; $index -lt $a.Count; ++$index) {
        if (-not [string]::Equals($a[$index], $e[$index], [StringComparison]::Ordinal)) { return $false }
    }
    return $true
}

function Start-Background([string]$Name, [string]$Bootstrap, [string[]]$Arguments) {
    $all = @('-Bootstrap', $Bootstrap, '-Manifest', $Manifest, '-Sidecar', $Sidecar, '-Client', $client,
        '-StateDir', (Join-Path $OutDir $Name), '-SimulateNat', '--') + $Arguments
    # Script parameter names (before "--") stay bare; every value is quoted.
    $afterDashes = $false
    $quoted = foreach ($argument in $all) {
        if ($argument -eq '--') { $afterDashes = $true; '--' }
        elseif (-not $afterDashes -and $argument -match '^-[A-Za-z]') { $argument }
        else { "'" + $argument.Replace("'", "''") + "'" }
    }
    $command = "& '$clientScript' " + ($quoted -join ' ')
    $out = Join-Path $OutDir "$Name.out"
    $process = Start-Process -FilePath 'powershell.exe' -PassThru -NoNewWindow -RedirectStandardOutput $out `
        -RedirectStandardError "$out.err" -ArgumentList @('-NoProfile', '-EncodedCommand',
            [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command)))
    $null = $process.Handle  # without this, ExitCode reads as empty after the process ends
    return $process
}

if ($Concurrent) {
    $processes = @()
    try {
        $infra = Start-Infra 'multi-infra' ($InfraPort + 3)
        $processes += $infra.Process
        $sessions = @('replica_sessions=2', 'max_sessions=2')
        $a = Start-Node 'multi-a' 5632 'auto' $infra.Address @($Manifest) 3 $sessions
        $b = Start-Node 'multi-b' 6144 'off' $infra.Address @($Manifest) 3 @('max_sessions=2')
        $c = Start-Node 'multi-c' 6656 'off' $infra.Address @($Manifest) 3 @('max_sessions=2')
        $processes += @($a.Process, $b.Process, $c.Process)
        Wait-ForLog $a.Owner 'READY: model' $a.Process 'replica formation' 300
        Wait-ForLog (Join-Path $a.State 'logs\sidecar.log') 'advertising dan/replica' $a.Process 'replica advertisement' 120
        $prompts = @('--replica-only', '--prompt', 'The capital of France is', '--prompt',
            'Name three colors of the rainbow.', '--requests', '2', '--tokens', "$Tokens")
        # One at a time first, for the time a single client takes.
        $started = Get-Date
        $one = Run-Client 'multi-alone' $infra.Address ($prompts + @('--report', (Join-Path $OutDir 'multi-alone.json')))
        $alone = ((Get-Date) - $started).TotalSeconds
        Check ($one.Exit -eq 0) 'one client alone'
        $started = Get-Date
        $first = Start-Background 'multi-1' $infra.Address ($prompts + @('--report', (Join-Path $OutDir 'multi-1.json')))
        $second = Start-Background 'multi-2' $infra.Address ($prompts + @('--report', (Join-Path $OutDir 'multi-2.json')))
        $processes += @($first, $second)
        foreach ($process in @($first, $second)) { if (-not $process.WaitForExit(300000)) { Stop-Tree $process } }
        $together = ((Get-Date) - $started).TotalSeconds
        Check ($first.ExitCode -eq 0 -and $second.ExitCode -eq 0) 'two clients at once both finished'
        $same = (Replica-Of (Join-Path $OutDir 'multi-1.out')) -eq (Replica-Of (Join-Path $OutDir 'multi-2.out'))
        Check $same 'both used the same replica'
        if ($BaselineDir) {
            foreach ($name in @('multi-alone', 'multi-1', 'multi-2')) {
                Check (Same-Outputs (Join-Path $OutDir "$name.json") (Join-Path $BaselineDir 'coordinator-once.json')) "$name output matches the baseline"
            }
        }
        # Did their tokens interleave on the ring? Look for decode steps of two sessions mixed.
        $sessionsSeen = @(Select-String -LiteralPath $b.Log -Pattern 'session=(\d+) request=\d+ stage=middle.*phase=decode' |
            ForEach-Object { $_.Matches[0].Groups[1].Value })
        $switches = 0
        for ($index = 1; $index -lt $sessionsSeen.Count; ++$index) { if ($sessionsSeen[$index] -ne $sessionsSeen[$index - 1]) { ++$switches } }
        Check ($switches -gt 4) "the two chats interleaved token by token on the ring ($switches switches)"
        Write-Host ("  one client alone: {0:N1} s; two clients at once: {1:N1} s (one after the other would be ~{2:N1} s)" -f $alone, $together, (2 * $alone))
        foreach ($line in (Select-String -LiteralPath (Join-Path $OutDir 'multi-1.out'), (Join-Path $OutDir 'multi-2.out') -Pattern '^request=').Line) {
            Write-Host "  $($line -replace ' output=.*$', '')"
        }
    } finally {
        foreach ($process in $processes) { Stop-Tree $process }
    }
    if ($failures.Count -gt 0) { Write-Host "Concurrency rehearsal FAILED: $($failures -join '; ')"; exit 1 }
    Write-Host 'Concurrency rehearsal passed.'
    exit 0
}

if ($Speculation) {
    $SpeculationManifest = (Resolve-Path -LiteralPath $SpeculationManifest).Path
    $processes = @()
    try {
        $infra = Start-Infra 'spec-infra' ($InfraPort + 2)
        $processes += $infra.Process
        $catalog = @($SpeculationManifest, $Manifest)
        $a = Start-Node 'spec-a' 5632 'auto' $infra.Address $catalog 2 @('replica_speculate=true')
        $b = Start-Node 'spec-b' 5632 'off' $infra.Address $catalog 2
        $processes += @($a.Process, $b.Process)
        Wait-ForLog $a.Owner 'READY: model' $a.Process 'replica formation' 600
        Write-Host "  $((Select-String -LiteralPath $a.Owner -Pattern 'READY: model' | Select-Object -Last 1).Line)"
        Wait-ForLog (Join-Path $a.State 'logs\sidecar.log') 'advertising dan/replica' $a.Process 'replica advertisement' 120
        Check (Select-String -LiteralPath $a.Log -Pattern 'speculation: draft model .* ready' -Quiet) 'the head loaded the draft model'
        $prompts = @('--prompt', 'The capital of France is', '--prompt', 'Name three colors of the rainbow.',
            '--requests', '2', '--tokens', "$Tokens")
        foreach ($mode in @('once', 'persistent')) {
            $arguments = @('--replica-only', '--manifest', $SpeculationManifest) + $prompts +
                @('--report', (Join-Path $OutDir "spec-$mode.json"))
            if ($mode -eq 'persistent') { $arguments += '--persistent' }
            $run = Run-Client "spec-client-$mode" $infra.Address $arguments
            Check ($run.Exit -eq 0 -and (Select-String -LiteralPath $run.Out -Pattern 'draft=yes' -Quiet)) "speculating replica answered ($mode)"
            foreach ($line in (Select-String -LiteralPath $run.Out -Pattern '^request=').Line) {
                Write-Host "  $($line -replace ' output=.*$', '')"
            }
            if ($SpeculationBaselineDir) {
                Check (Same-Outputs (Join-Path $OutDir "spec-$mode.json") (Join-Path $SpeculationBaselineDir "dan-client-$mode.json")) "speculating replica output matches the placed speculating route ($mode)"
            }
        }

    } finally {
        foreach ($process in $processes) { Stop-Tree $process }
    }
    if ($failures.Count -gt 0) { Write-Host "Speculation rehearsal FAILED: $($failures -join '; ')"; exit 1 }
    Write-Host 'Speculation rehearsal passed.'
    exit 0
}

$processes = @()
try {
    # ---- 1. Persistence ----
    $infra = Start-Infra 'infra' $InfraPort
    $processes += $infra.Process
    $bootstrap = $infra.Address
    Write-Host "infra $bootstrap"
    # The fake GPU reports 8192 MiB; these reserves leave 2560 / 2048 / 1536 MiB for DAN.
    $a = Start-Node 'node-a' 5632 'auto' $bootstrap
    $b = Start-Node 'node-b' 6144 'off' $bootstrap
    $c = Start-Node 'node-c' 6656 'off' $bootstrap
    $nodes = @($a, $b, $c)
    $processes += $nodes | ForEach-Object { $_.Process }
    foreach ($node in $nodes) {
        Wait-ForLog (Join-Path $node.State 'logs\sidecar.log') 'advertising dan/model' $node.Process "$($node.Name) advertisement" 120
    }
    Write-Host 'nodes joined; waiting for the replica to form by itself'
    Wait-ForLog $a.Owner 'READY: model' $a.Process 'replica formation' 300
    Write-Host "  $((Select-String -LiteralPath $a.Owner -Pattern 'READY: model' | Select-Object -Last 1).Line)"
    Wait-ForLog (Join-Path $a.State 'logs\sidecar.log') 'advertising dan/replica' $a.Process 'replica advertisement' 120
    $status = Get-Content -LiteralPath $a.Status -Raw | ConvertFrom-Json
    Check ($status.state -eq 'ready' -and $status.members.Count -eq 3) 'replica READY with three members, formed without a client'
    $formedId = $status.replica_id
    $before = Worker-Counts $nodes

    $prompts = @('--prompt', 'The capital of France is', '--prompt', 'Name three colors of the rainbow.',
        '--requests', '2', '--tokens', "$Tokens")
    $first = Run-Client 'client-1' $bootstrap (@('--replica-only') + $prompts + @('--report', (Join-Path $OutDir 'client-1.json')))
    Check ($first.Exit -eq 0) 'client 1 chatted through the replica'
    $firstId = Replica-Of $first.Out
    $second = Run-Client 'client-2' $bootstrap (@('--replica-only', '--persistent') + $prompts + @('--report', (Join-Path $OutDir 'client-2.json')))
    Check ($second.Exit -eq 0) 'client 2 chatted through the replica (one conversation)'
    $secondId = Replica-Of $second.Out
    Check ($firstId -eq $formedId -and $secondId -eq $formedId) "both clients used the same replica ($formedId)"
    $after = Worker-Counts $nodes
    $unchanged = $true
    foreach ($node in $nodes) {
        if (($before[$node.Name] -join ',') -ne ($after[$node.Name] -join ',')) { $unchanged = $false }
    }
    Check $unchanged 'no reserve, assignment, download or ring link after the replica formed'
    $status = Get-Content -LiteralPath $a.Status -Raw | ConvertFrom-Json
    Check ($status.state -eq 'ready' -and $status.replica_id -eq $formedId -and $status.sessions_in_use -eq 0) 'replica still READY and empty after both clients left'
    foreach ($line in (Select-String -LiteralPath $first.Out, $second.Out -Pattern '^(replica=|request=)').Line) {
        Write-Host "  $($line -replace ' output=.*$', '')"
    }
    if ($BaselineDir) {
        Check (Same-Outputs (Join-Path $OutDir 'client-1.json') (Join-Path $BaselineDir 'coordinator-once.json')) 'client 1 output matches the baseline'
        Check (Same-Outputs (Join-Path $OutDir 'client-2.json') (Join-Path $BaselineDir 'coordinator-persistent.json')) 'client 2 output matches the baseline'
    }
    $chatInput = Join-Path $OutDir 'chat.in'
    [IO.File]::WriteAllLines($chatInput, @('What is the capital of France? Answer in one word.', '/quit'))
    $chat = Run-Client 'client-chat' $bootstrap @('--replica-only', '--chat', '--tokens', '16') $chatInput
    Check ($chat.Exit -eq 0 -and (Select-String -LiteralPath $chat.Out -Pattern 'dan > .*Paris' -Quiet)) 'chat through the replica'

    # ---- 2. Failure ----
    if (-not $SkipFailure) {
        $longOut = Join-Path $OutDir 'client-long.out'
        $longArguments = @('-Bootstrap', $bootstrap, '-Manifest', $Manifest, '-Sidecar', $Sidecar, '-Client', $client,
            '-StateDir', (Join-Path $OutDir 'client-long'), '-SimulateNat', '--', '--replica-only',
            '--prompt', 'Count from one to one thousand in words, separated by commas.', '--tokens', '480')
        $sessionsBefore = Count-Lines $a.Owner 'opened a session'
        # -File would take "--" as a parameter name; a command keeps PowerShell's own parsing.
        # Script parameter names (before "--") stay bare; every value is quoted.
        $afterDashes = $false
        $quoted = foreach ($argument in $longArguments) {
            if ($argument -eq '--') { $afterDashes = $true; '--' }
            elseif (-not $afterDashes -and $argument -match '^-[A-Za-z]') { $argument }
            else { "'" + $argument.Replace("'", "''") + "'" }
        }
        $command = "& '$clientScript' " + ($quoted -join ' ')
        $long = Start-Process -FilePath 'powershell.exe' -PassThru -NoNewWindow -RedirectStandardOutput $longOut `
            -RedirectStandardError "$longOut.err" -ArgumentList @('-NoProfile', '-EncodedCommand',
                [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command)))
        $processes += $long
        Wait-ForLog $a.Owner 'opened a session' $long 'long request' 120 ($sessionsBefore + 1)
        Start-Sleep -Seconds 3
        Write-Host 'killing node B during the answer'
        Stop-Tree $b.Process
        if (-not $long.WaitForExit(120000)) { Stop-Tree $long }
        Check ($long.ExitCode -ne 0) 'the client whose answer was cut got an error'
        Wait-ForLog $a.Owner 'dissolved' $a.Process 'dissolution' 120
        Write-Host "  $((Select-String -LiteralPath $a.Owner -Pattern 'dissolved' | Select-Object -Last 1).Line)"
        Wait-ForLog $c.Log 'route \S+ ended' $c.Process 'node C release' 60
        $status = Get-Content -LiteralPath $a.Status -Raw | ConvertFrom-Json
        Check ($status.state -ne 'ready') "replica no longer advertised as ready (state $($status.state))"
        $reuseBefore = (Count-Lines $a.Log 'Reusing loaded stage') + (Count-Lines $c.Log 'Reusing loaded stage')
        $downloadsBefore = (Count-Lines $a.Log 'Downloading required model data') + (Count-Lines $c.Log 'Downloading required model data')
        Write-Host 'restarting node B'
        $b = Start-Node 'node-b' 6144 'off' $bootstrap
        $processes += $b.Process
        Wait-ForLog $a.Owner 'READY: model' $a.Process 're-formation' 300 2
        $status = Get-Content -LiteralPath $a.Status -Raw | ConvertFrom-Json
        Check ($status.state -eq 'ready' -and $status.replica_id -ne $formedId) "a new replica formed ($($status.replica_id))"
        $reuseAfter = (Count-Lines $a.Log 'Reusing loaded stage') + (Count-Lines $c.Log 'Reusing loaded stage')
        $downloadsAfter = (Count-Lines $a.Log 'Downloading required model data') + (Count-Lines $c.Log 'Downloading required model data')
        Check ($reuseAfter -eq $reuseBefore + 2 -and $downloadsAfter -eq $downloadsBefore) 'A and C reused their loaded layers (no download, no reload)'
        $again = Run-Client 'client-3' $bootstrap (@('--replica-only') + $prompts + @('--report', (Join-Path $OutDir 'client-3.json')))
        Check ($again.Exit -eq 0 -and (Replica-Of $again.Out) -eq $status.replica_id) 'a client uses the new replica'

        # The owner process itself dies: its connections close, so every member is released
        # (no election, no second owner).
        $ended = @($a, $b, $c | ForEach-Object { Count-Lines $_.Log 'route \S+ ended' })
        $ownerProcess = Get-CimInstance Win32_Process -Filter "Name = 'dan-client.exe'" |
            Where-Object { $_.CommandLine -like '*--form*' -and $_.CommandLine -like "*$($a.State)*" }
        Write-Host 'killing the replica owner process'
        $ownerProcess | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        foreach ($index in 0..2) {
            $node = @($a, $b, $c)[$index]
            Wait-ForLog $node.Log 'route \S+ ended' $null "$($node.Name) release" 60 ($ended[$index] + 1)
        }
        Check $true 'owner death released every member'
    }
} finally {
    foreach ($process in $processes) { Stop-Tree $process }
}

# ---- 3. Race ----
if (-not $SkipRace) {
    $processes = @()
    try {
        $infra = Start-Infra 'race-infra' ($InfraPort + 1)
        $processes += $infra.Process
        # Each offers 2048 MiB; three are needed, four owners compete.
        $racers = @(1..4 | ForEach-Object { Start-Node "race-$_" 6144 'auto' $infra.Address })
        $processes += $racers | ForEach-Object { $_.Process }
        $deadline = (Get-Date).AddSeconds(300)
        do {
            Start-Sleep -Seconds 5
            $ready = @($racers | Where-Object {
                (Test-Path $_.Status) -and ((Get-Content -LiteralPath $_.Status -Raw | ConvertFrom-Json).state -eq 'ready') })
        } while ($ready.Count -lt 1 -and (Get-Date) -lt $deadline)
        # Give any second (impossible) formation time to show up, and any lease time to expire.
        Start-Sleep -Seconds 75
        $ready = @($racers | Where-Object {
            (Test-Path $_.Status) -and ((Get-Content -LiteralPath $_.Status -Raw | ConvertFrom-Json).state -eq 'ready') })
        Check ($ready.Count -eq 1) "exactly one replica formed among four racing owners ($($ready.Count))"
        $states = @($racers | ForEach-Object { (Get-Content -LiteralPath (Join-Path $_.State 'worker-status.json') -Raw | ConvertFrom-Json).state })
        Write-Host "  worker states: $($states -join ', ')"
        Check ((@($states | Where-Object { $_ -eq 'serving' }).Count -eq 3) -and (@($states | Where-Object { $_ -eq 'available' }).Count -eq 1)) 'three members serving, one node free, nothing left reserved'
        $lost = 0
        foreach ($racer in $racers) {
            if (Test-Path $racer.Status) { $lost += [int](Get-Content -LiteralPath $racer.Status -Raw | ConvertFrom-Json).races_lost }
        }
        Write-Host "  races lost (released and replanned): $lost"
    } finally {
        foreach ($process in $processes) { Stop-Tree $process }
    }
}

if ($failures.Count -gt 0) {
    Write-Host "Replica rehearsal FAILED: $($failures -join '; ')"
    exit 1
}
Write-Host 'Replica rehearsal passed.'
