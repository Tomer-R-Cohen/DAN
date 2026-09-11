<#
.SYNOPSIS
  Windows-side commands for docs/PIPELINED_RING_PHYSICAL_TEST.md.

.DESCRIPTION
  Covers each step of that doc as a subcommand instead of typing the full
  command by hand. Several steps run in separate windows at the same time
  (e.g. stage 0, the coordinator, and up to three sidecar tunnels), so this
  is a menu of steps, not a single "run everything" script -- open one
  PowerShell window per step that needs to stay running.

  Run without arguments, or with an unknown step, to see this list.

.EXAMPLE
  .\pipelined_ring_test_windows.ps1 build
  .\pipelined_ring_test_windows.ps1 weights
  .\pipelined_ring_test_windows.ps1 sidecar-id -KeyName control-client
  .\pipelined_ring_test_windows.ps1 tunnel-control-client -LinuxIp 203.0.113.10 -PeerId 12D3Koo...
  .\pipelined_ring_test_windows.ps1 stage0
  .\pipelined_ring_test_windows.ps1 baseline
  .\pipelined_ring_test_windows.ps1 pipelined -PipelineDepth 6
  .\pipelined_ring_test_windows.ps1 tunnel-ring-client -LinuxIp 203.0.113.10 -PeerId 12D3Koo...
  .\pipelined_ring_test_windows.ps1 tunnel-ringreturn-server -PeerId 12D3Koo...
  .\pipelined_ring_test_windows.ps1 stage0-ring
  .\pipelined_ring_test_windows.ps1 ring -PipelineDepth 6
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
    [string]$KeyName,
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
$DanBin = '.\build-provider-owned-cuda\Release'
$SidecarBin = '.\build\sidecar\dan-sidecar-windows-amd64.exe'
$StatsCsv = '.\build\pipelined-ring-stats.csv'
$env:PATH = "$PWD\build-provider-owned-cuda\bin\Release;$env:PATH"

function Show-Usage {
    Write-Output @'
Steps (see docs/PIPELINED_RING_PHYSICAL_TEST.md for what each one means):

  build                                       step 3: build DAN + sidecar, check CUDA flags
  weights                                     step 4: download weights, write the manifest
  sidecar-id -KeyName X                       step 5: create one sidecar identity, print its PeerID
  tunnel-control-client -LinuxIp X -PeerId Y  step 5, pair 1: Windows side (needed for every section)
  tunnel-ring-client -LinuxIp X -PeerId Y     step 5, pair 2: Windows side (ring only)
  tunnel-ringreturn-server -PeerId Y          step 5, pair 3: Windows side (ring only)
  stage0                                      step 6/7: stage 0, plain (hub-and-spoke)
  baseline                                    step 6: coordinator, no draft model
  pipelined [-PipelineDepth N]                step 7: coordinator, pipelined speculative decoding
  stage0-ring                                 step 8: stage 0, with --next/--prefill-chunk
  ring [-PipelineDepth N]                     step 8: coordinator, with --ring-return
  check-chunks                                step 8: confirm chunked prefill actually fired
  stats                                       show/record decode_tok_s, latency, draft accept
                                               across every run so far ($StatsCsv)
  package                                     step 9: zip up logs, reports, and stats

Sidecar tunnel processes and stage/coordinator processes are long-running --
each of the steps above that starts one blocks its window until you Ctrl+C.
Open a separate window per step you need running at the same time.

baseline/pipelined/ring each write a JSON --report next to their log and
append one row to $StatsCsv (created on first run) so results from repeat
runs accumulate instead of overwriting each other. Run 'stats' any time to
see the table so far.
'@
}

function Confirm-Path([string]$Path, [string]$Hint) {
    if (-not (Test-Path $Path)) {
        throw "Expected '$Path' to exist. $Hint"
    }
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

switch ($Step) {
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

    'weights' {
        $Revision = '91cad51170dc346986eccefdc2dd33a9da36ead9'
        $Sha256 = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
        $Url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$Revision/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        New-Item -ItemType Directory -Force .\build\pipelined-ring-weights | Out-Null
        Invoke-WebRequest -Uri $Url -OutFile $Weights
        $actual = (Get-FileHash $Weights -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $Sha256) {
            throw "Weight SHA-256 mismatch: expected $Sha256, got $actual"
        }
        Write-Output "Weights OK: $actual"

        @"
{
  "model_id": "qwen25-1-5b-pipelined-ring",
  "architecture": "qwen2",
  "layers": 28,
  "hidden_size": 1536,
  "context_size": 1024,
  "artifact_revision": "$Revision",
  "artifact_sha256": "$Sha256",
  "artifact_url": "$Url"
}
"@ | Set-Content -Encoding utf8 $Manifest
        Write-Output "Manifest written to $Manifest"
    }

    'sidecar-id' {
        if (-not $KeyName) { throw 'Usage: sidecar-id -KeyName <name>  (e.g. control-client, ring-client, ringreturn-server)' }
        & $SidecarBin -key "$KeyName.key" -id
    }

    'tunnel-control-client' {
        if (-not $LinuxIp -or -not $PeerId) { throw 'Usage: tunnel-control-client -LinuxIp <ip> -PeerId <control-server PeerID from Linux>' }
        & $SidecarBin -key control-client.key `
            -listen /ip4/0.0.0.0/tcp/4101 `
            -forward "127.0.0.1:50102=/ip4/$LinuxIp/tcp/4101/p2p/$PeerId" `
            2>&1 | Tee-Object .\build\sidecar-control-client.log
    }

    'tunnel-ring-client' {
        if (-not $LinuxIp -or -not $PeerId) { throw 'Usage: tunnel-ring-client -LinuxIp <ip> -PeerId <ring-server PeerID from Linux>' }
        & $SidecarBin -key ring-client.key `
            -listen /ip4/0.0.0.0/tcp/4102 `
            -forward "127.0.0.1:50103=/ip4/$LinuxIp/tcp/4102/p2p/$PeerId" `
            2>&1 | Tee-Object .\build\sidecar-ring-client.log
    }

    'tunnel-ringreturn-server' {
        if (-not $PeerId) { throw 'Usage: tunnel-ringreturn-server -PeerId <ringreturn-client PeerID from Linux>' }
        & $SidecarBin -key ringreturn-server.key `
            -listen /ip4/0.0.0.0/tcp/4103 -inbound 127.0.0.1:50105 `
            -allow $PeerId `
            2>&1 | Tee-Object .\build\sidecar-ringreturn-server.log
    }

    'stage0' {
        Confirm-Path $Weights 'Run: .\pipelined_ring_test_windows.ps1 weights'
        & "$DanBin\dan-stage-worker.exe" `
            --model $Weights `
            --stage-start 0 --stage-end 14 --host 127.0.0.1 --port 50101 `
            --ctx 1024 --gpu-layers 999 --max-sessions 4 `
            2>&1 | Tee-Object .\build\stageA-baseline.log
    }

    'baseline' {
        Confirm-Path $Manifest 'Run: .\pipelined_ring_test_windows.ps1 weights'
        & "$DanBin\dan-provider-owned-coordinator.exe" `
            --manifest $Manifest `
            --provider 127.0.0.1:50101 --provider 127.0.0.1:50102 `
            --prompt 'The capital of France is' --tokens 20 --requests 3 `
            --report .\build\baseline-report.json `
            2>&1 | Tee-Object .\build\baseline-report.log
        Save-RunStats -RunName 'baseline' -ReportPath .\build\baseline-report.json
    }

    'pipelined' {
        Confirm-Path $Manifest 'Run: .\pipelined_ring_test_windows.ps1 weights'
        & "$DanBin\dan-provider-owned-coordinator.exe" `
            --manifest $Manifest `
            --provider 127.0.0.1:50101 --provider 127.0.0.1:50102 `
            --draft-model $Weights --draft-tokens $DraftTokens --draft-gpu-layers 999 `
            --pipeline-depth $PipelineDepth `
            --prompt $Prompt `
            --tokens $Tokens --requests $Requests `
            --report .\build\pipelined-hubspoke-report.json `
            2>&1 | Tee-Object .\build\pipelined-hubspoke-report.log
        Save-RunStats -RunName 'pipelined' -ReportPath .\build\pipelined-hubspoke-report.json
    }

    'stage0-ring' {
        Confirm-Path $Weights 'Run: .\pipelined_ring_test_windows.ps1 weights'
        & "$DanBin\dan-stage-worker.exe" `
            --model $Weights `
            --stage-start 0 --stage-end 14 --host 127.0.0.1 --port 50101 `
            --next 127.0.0.1:50103 --prefill-chunk $PrefillChunk `
            --ctx 1024 --gpu-layers 999 --max-sessions 4 `
            2>&1 | Tee-Object .\build\stageA-ring.log
    }

    'ring' {
        Confirm-Path $Manifest 'Run: .\pipelined_ring_test_windows.ps1 weights'
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
            Write-Output "No runs recorded yet in $StatsCsv -- run 'baseline', 'pipelined', or 'ring' first."
        } else {
            Import-Csv $StatsCsv | Format-Table -AutoSize `
                timestamp, run, decode_tok_s, request_latency_p50_ms, draft_acceptance, network_ms_per_token
        }
    }

    'package' {
        $paths = @(
            '.\build\stageA-*.log', '.\build\sidecar-*.log',
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
