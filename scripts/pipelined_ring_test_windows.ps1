<#
.SYNOPSIS
  Windows-side commands for docs/PIPELINED_RING_PHYSICAL_TEST.md.

.DESCRIPTION
  Three real phases: prereqs, build, run. 'run' does everything needed to
  actually execute baseline/pipelined in one command -- it prints this
  machine's PeerID, starts the sidecar tunnel and the local provider in the
  background, runs the coordinator in the foreground, and cleans up after.
  'run-ring' does the same for ring mode's three tunnels plus stage 0.
  check-chunks/stats/package are evaluation utilities, not part of running a
  test, and stay separate on purpose.

  Run without arguments, or with an unknown step, to see this list.

.EXAMPLE
  .\pipelined_ring_test_windows.ps1 prereqs
  .\pipelined_ring_test_windows.ps1 build
  .\pipelined_ring_test_windows.ps1 run -PeerId 12D3Koo...
  .\pipelined_ring_test_windows.ps1 run -PeerId 12D3Koo... -Draft
  .\pipelined_ring_test_windows.ps1 run-ring -LinuxIp 203.0.113.10 -ControlPeerId 12D3Koo... -RingPeerId 12D3Koo... -RingReturnPeerId 12D3Koo...
  .\pipelined_ring_test_windows.ps1 check-chunks
  .\pipelined_ring_test_windows.ps1 stats
  .\pipelined_ring_test_windows.ps1 package
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Step,

    [string]$Branch,
    [string]$LinuxIp,
    [string]$PeerId,
    [string]$ControlPeerId,
    [string]$RingPeerId,
    [string]$RingReturnPeerId,
    [switch]$Draft,
    [int]$PipelineDepth = 6,
    [int]$PrefillChunk = 24,
    [int]$DraftTokens = 8,
    [int]$Tokens = 120,
    [int]$Requests = 3,
    [string]$Prompt = 'Explain in detail why Paris became the capital of France, including historical, political, and geographical reasons.'
)

$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Halakim Family\Desktop\DAN'

$Manifest = '.\build\pipelined-ring-manifest.json'
$Weights = '.\build\pipelined-ring-weights\qwen2.5-1.5b.gguf'
$ProviderCache = '.\build\pipelined-ring-provider-cache'
$MetadataCache = '.\build\pipelined-ring-model-index.tmp'
$DanBin = '.\build-provider-owned-cuda\Release'
$SidecarBin = '.\build\sidecar\dan-sidecar-windows-amd64.exe'
$StatsCsv = '.\build\pipelined-ring-stats.csv'
$env:PATH = "$PWD\build-provider-owned-cuda\bin\Release;$env:PATH"

# Fixed constants for the pinned test model.
$ModelRevision = '91cad51170dc346986eccefdc2dd33a9da36ead9'
$ModelSha256 = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
$ModelUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$ModelRevision/qwen2.5-1.5b-instruct-q4_k_m.gguf"

function Show-Usage {
    Write-Output @'
Steps (see docs/PIPELINED_RING_PHYSICAL_TEST.md for background):

  prereqs                                     step 1: check Go/CMake/CUDA toolchain are present
  build                                       step 2: build DAN + sidecar, check CUDA flags
  run -PeerId Y [-Draft]                      step 3: baseline (default) or pipelined (-Draft).
                                               Writes the manifest and (if -Draft) downloads the
                                               draft weights if missing, prints this machine's
                                               PeerID, runs the tunnel and local provider in the
                                               background, the coordinator in the foreground.
  run-ring -LinuxIp X -ControlPeerId Y        step 3, ring mode: downloads weights if missing,
    -RingPeerId Y -RingReturnPeerId Y         prints PeerIDs, runs all three tunnels and stage 0
                                               in the background, the coordinator in the foreground.
                                               Needs 'run' completed at least once first (reuses its
                                               manifest). Ring can't auto-register -- see the doc.
  check-chunks                                confirm chunked prefill actually fired (ring only)
  stats                                       show/record decode_tok_s, latency, draft accept
                                               across every run so far ($StatsCsv)
  package                                     zip up logs, reports, and stats

'run'/'run-ring' need this machine's printed PeerID given to the matching
Linux command, and (for run-ring) the Linux PeerIDs given back here -- that
handshake is manual and can't be scripted away. Everything else -- writing
the manifest, downloading weights, starting the tunnel and provider
processes, cleaning them up afterward -- happens automatically inside the
one command.
'@
}

function Confirm-Path([string]$Path, [string]$Hint) {
    if (-not (Test-Path $Path)) {
        throw "Expected '$Path' to exist. $Hint"
    }
}

function Write-Manifest {
    if (Test-Path $Manifest) { return }
    New-Item -ItemType Directory -Force .\build | Out-Null
    @"
{
  "model_id": "qwen25-1-5b-pipelined-ring",
  "architecture": "qwen2",
  "layers": 28,
  "hidden_size": 1536,
  "context_size": 1024,
  "artifact_revision": "$ModelRevision",
  "artifact_sha256": "$ModelSha256",
  "artifact_url": "$ModelUrl"
}
"@ | Set-Content -Encoding utf8 $Manifest
    Write-Output "Manifest written to $Manifest"
}

function Get-Weights {
    if (Test-Path $Weights) { return }
    Write-Output 'Downloading draft/ring weights (one-time; not needed for plain baseline)...'
    New-Item -ItemType Directory -Force .\build\pipelined-ring-weights | Out-Null
    Invoke-WebRequest -Uri $ModelUrl -OutFile $Weights
    $actual = (Get-FileHash $Weights -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $ModelSha256) {
        Remove-Item $Weights -ErrorAction SilentlyContinue
        throw "Weight SHA-256 mismatch: expected $ModelSha256, got $actual"
    }
    Write-Output "Weights OK: $actual"
}

# Reads the JSON --report a coordinator run just wrote and appends one row
# to $StatsCsv, so 'stats' can show every run so far, not just the last one.
function Save-RunStats([string]$RunName, [string]$ReportPath) {
    if (-not (Test-Path $ReportPath)) {
        Write-Warning "No report at '$ReportPath' -- coordinator may have failed before writing it. Not recording stats for '$RunName'."
        return
    }
    $report = Get-Content $ReportPath -Raw | ConvertFrom-Json
    $row = [PSCustomObject]@{
        timestamp             = (Get-Date -Format 'o')
        run                   = $RunName
        requests_served       = $report.requests_served
        tokens_generated      = $report.tokens_generated
        decode_tok_s          = $report.decode_tok_s
        request_latency_p50_ms = $report.request_latency_p50_ms
        request_latency_p95_ms = $report.request_latency_p95_ms
        speculative_rounds    = $report.speculative_rounds
        draft_acceptance      = $report.draft_acceptance
        draft_ms_per_generated_token = $report.draft_ms_per_generated_token
        network_ms_per_token  = $report.network_ms_per_token
    }
    $row | Export-Csv -Path $StatsCsv -Append -NoTypeInformation -Force
    Write-Output ("Recorded '{0}' to {1}: decode_tok_s={2:N3} p50_ms={3:N1} draft_accept={4:N3}" -f `
        $RunName, $StatsCsv, $report.decode_tok_s, $report.request_latency_p50_ms, $report.draft_acceptance)
}

# Starts a long-running child process in the background, logging its output,
# and returns the Process object so the caller can Stop-Process it later.
function Start-Background([string]$FilePath, [string[]]$Arguments, [string]$LogPath) {
    return Start-Process -FilePath $FilePath -ArgumentList $Arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.err"
}

switch ($Step) {
    'prereqs' {
        Write-Output '--- go ---'
        go version
        Write-Output '--- cmake ---'
        cmake --version
        Write-Output '--- nvidia-smi ---'
        nvidia-smi
        Write-Output 'All three must have printed real output above before continuing.'
    }

    'build' {
        if ($Branch) {
            git pull --ff-only origin $Branch
        } else {
            Write-Warning 'No -Branch given; building whatever is currently checked out.'
        }
        git rev-parse HEAD | Tee-Object revision-windows.txt

        cmake --build build-provider-owned-cuda --config Release --target dan-provider-owned-coordinator
        cmake --build build-provider-owned-cuda --config Release --target dan-stage-worker

        $flags = Select-String -Path .\build-provider-owned-cuda\CMakeCache.txt -Pattern '^GGML_CUDA(_GRAPHS)?:BOOL=ON$'
        if (($flags | Measure-Object).Count -lt 2) {
            Write-Warning 'GGML_CUDA and/or GGML_CUDA_GRAPHS are not both ON in build-provider-owned-cuda. Reconfigure with -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON before continuing.'
        } else {
            Write-Output 'GGML_CUDA and GGML_CUDA_GRAPHS both confirmed ON.'
        }

        .\scripts\build_sidecar.ps1
    }

    'run' {
        if (-not $PeerId) { throw 'Usage: run -PeerId <provider1 PeerID from Linux, from the Linux run command''s own output> [-Draft]' }
        Write-Manifest
        if ($Draft) { Get-Weights }
        New-Item -ItemType Directory -Force $ProviderCache | Out-Null

        Write-Output '--- This machine''s PeerID (give it to the Linux run command) ---'
        & $SidecarBin -key coordinator.key -id
        Write-Output '---'

        $tunnel = Start-Background $SidecarBin @(
            '-key', 'coordinator.key', '-listen', '/ip4/0.0.0.0/tcp/4100',
            '-inbound', '127.0.0.1:50200', '-allow', $PeerId
        ) '.\build\sidecar-coordinator.log'
        $provider0 = Start-Background "$DanBin\dan-stage-worker.exe" @(
            '--coordinator', '127.0.0.1:50200', '--provider-id', 'windows-provider0', '--cache-dir', $ProviderCache
        ) '.\build\provider0.log'

        try {
            Start-Sleep -Seconds 2
            if ($Draft) {
                & "$DanBin\dan-provider-owned-coordinator.exe" `
                    --manifest $Manifest `
                    --provider-listen 127.0.0.1:50200 --metadata-cache $MetadataCache --provider-peer-auth `
                    --draft-model $Weights --draft-tokens $DraftTokens --draft-gpu-layers 999 `
                    --pipeline-depth $PipelineDepth `
                    --prompt $Prompt `
                    --tokens $Tokens --requests $Requests --shutdown-workers `
                    --report .\build\pipelined-hubspoke-report.json `
                    2>&1 | Tee-Object .\build\pipelined-hubspoke-report.log
                Save-RunStats -RunName 'pipelined' -ReportPath .\build\pipelined-hubspoke-report.json
            } else {
                & "$DanBin\dan-provider-owned-coordinator.exe" `
                    --manifest $Manifest `
                    --provider-listen 127.0.0.1:50200 --metadata-cache $MetadataCache --provider-peer-auth `
                    --prompt 'The capital of France is' --tokens 20 --requests 3 --shutdown-workers `
                    --report .\build\baseline-report.json `
                    2>&1 | Tee-Object .\build\baseline-report.log
                Save-RunStats -RunName 'baseline' -ReportPath .\build\baseline-report.json
            }
        } finally {
            Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
            Stop-Process -Id $provider0.Id -ErrorAction SilentlyContinue
        }
    }

    'run-ring' {
        if (-not $LinuxIp -or -not $ControlPeerId -or -not $RingPeerId -or -not $RingReturnPeerId) {
            throw 'Usage: run-ring -LinuxIp <ip> -ControlPeerId <Linux control-server PeerID> -RingPeerId <Linux ring-server PeerID> -RingReturnPeerId <Linux ringreturn-client PeerID>'
        }
        Confirm-Path $Manifest "Run 'run' at least once first (it writes the manifest this reuses)."
        Get-Weights

        Write-Output '--- This machine''s PeerIDs (give them to the matching Linux run-ring command) ---'
        Write-Output 'control-client:'; & $SidecarBin -key control-client.key -id
        Write-Output 'ring-client:'; & $SidecarBin -key ring-client.key -id
        Write-Output 'ringreturn-server:'; & $SidecarBin -key ringreturn-server.key -id
        Write-Output '---'

        $controlTunnel = Start-Background $SidecarBin @(
            '-key', 'control-client.key', '-listen', '/ip4/0.0.0.0/tcp/4101',
            '-forward', "127.0.0.1:50102=/ip4/$LinuxIp/tcp/4101/p2p/$ControlPeerId"
        ) '.\build\sidecar-control-client.log'
        $ringTunnel = Start-Background $SidecarBin @(
            '-key', 'ring-client.key', '-listen', '/ip4/0.0.0.0/tcp/4102',
            '-forward', "127.0.0.1:50103=/ip4/$LinuxIp/tcp/4102/p2p/$RingPeerId"
        ) '.\build\sidecar-ring-client.log'
        $ringReturnTunnel = Start-Background $SidecarBin @(
            '-key', 'ringreturn-server.key', '-listen', '/ip4/0.0.0.0/tcp/4103',
            '-inbound', '127.0.0.1:50105', '-allow', $RingReturnPeerId
        ) '.\build\sidecar-ringreturn-server.log'
        $stage0 = Start-Background "$DanBin\dan-stage-worker.exe" @(
            '--model', $Weights, '--stage-start', '0', '--stage-end', '14', '--host', '127.0.0.1', '--port', '50101',
            '--next', '127.0.0.1:50103', '--prefill-chunk', "$PrefillChunk",
            '--ctx', '1024', '--gpu-layers', '999', '--max-sessions', '4'
        ) '.\build\stageA-ring.log'

        try {
            Start-Sleep -Seconds 3
            & "$DanBin\dan-provider-owned-coordinator.exe" `
                --manifest $Manifest `
                --provider 127.0.0.1:50101 --provider 127.0.0.1:50102 `
                --ring-return 127.0.0.1:50105 `
                --draft-model $Weights --draft-tokens $DraftTokens --draft-gpu-layers 999 `
                --pipeline-depth $PipelineDepth `
                --prompt $Prompt `
                --tokens $Tokens --requests $Requests `
                --report .\build\ring-report.json `
                2>&1 | Tee-Object .\build\ring-report.log
            Save-RunStats -RunName 'ring' -ReportPath .\build\ring-report.json
        } finally {
            Stop-Process -Id $controlTunnel.Id -ErrorAction SilentlyContinue
            Stop-Process -Id $ringTunnel.Id -ErrorAction SilentlyContinue
            Stop-Process -Id $ringReturnTunnel.Id -ErrorAction SilentlyContinue
            Stop-Process -Id $stage0.Id -ErrorAction SilentlyContinue
        }
    }

    'check-chunks' {
        $matches = Select-String -Path .\build\stageA-ring.log -Pattern 'phase=prefill-chunk'
        $count = ($matches | Measure-Object).Count
        Write-Output "prefill-chunk lines in stageA-ring.log: $count"
        if ($count -lt 2) {
            Write-Warning 'Expected more than one non-final chunk. Chunking may not have fired -- check --prefill-chunk against the actual prompt length.'
        }
    }

    'stats' {
        if (-not (Test-Path $StatsCsv)) {
            Write-Output "No runs recorded yet in $StatsCsv -- run 'run' or 'run-ring' first."
        } else {
            Import-Csv $StatsCsv | Format-Table -AutoSize `
                timestamp, run, decode_tok_s, request_latency_p50_ms, draft_acceptance, network_ms_per_token
        }
    }

    'package' {
        $paths = @(
            '.\build\stageA-*.log', '.\build\sidecar-*.log', '.\build\provider0.log',
            '.\build\baseline-report.log', '.\build\pipelined-hubspoke-report.log',
            '.\build\ring-report.log',
            '.\build\baseline-report.json', '.\build\pipelined-hubspoke-report.json',
            '.\build\ring-report.json', $StatsCsv,
            '.\revision-windows.txt'
        ) | Where-Object { Test-Path $_ }
        if (-not $paths) { throw 'No result files found yet -- run the earlier steps first.' }
        Compress-Archive -Path $paths -DestinationPath .\pipelined-ring-results.zip -Force
        Write-Output 'Wrote .\pipelined-ring-results.zip'
    }

    default {
        Show-Usage
    }
}
