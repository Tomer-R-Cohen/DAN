[CmdletBinding()]
param(
    [string]$Listen = '127.0.0.1:8080',
    [string]$ApiKey = $env:DAN_API_KEY,
    [string]$Model = '',
    [string]$ProviderNetwork = 'libp2p',
    [string]$LogDirectory = '',
    [int]$CoordinatorProcessId = 0,
    [int]$GatewayProcessId = 0
)

$ErrorActionPreference = 'Stop'
$base = "http://$Listen"
$headers = @{}
if ($ApiKey) { $headers['Authorization'] = "Bearer $ApiKey" }

function ConvertFrom-Prometheus {
    param([string]$Text)
    $metrics = @{}
    foreach ($line in ($Text -split "`n")) {
        if (-not $line -or $line.StartsWith('#')) { continue }
        if ($line -match '^([a-zA-Z_:][a-zA-Z0-9_:]*)\{replica="(\d+)"\}\s+([0-9eE+\-.]+)\s*$') {
            $name = $Matches[1]; $replica = [int]$Matches[2]; $value = [double]$Matches[3]
            if (-not $metrics.ContainsKey($replica)) { $metrics[$replica] = @{} }
            $metrics[$replica][$name] = $value
        }
    }
    return $metrics
}

function Format-Bar {
    param([double]$Percent, [int]$Width = 24)
    $Percent = [Math]::Max(0, [Math]::Min(100, $Percent))
    $filled = [int]([Math]::Round($Percent * $Width / 100))
    return ('#' * $filled) + ('-' * ($Width - $filled)) + (' {0,3:N0}%' -f $Percent)
}

function Write-Center {
    param([string]$Text, [int]$Width, [string]$Color = 'Gray')
    $pad = [Math]::Max(0, [Math]::Floor(($Width - $Text.Length) / 2.0))
    Write-Host ((' ' * $pad) + $Text) -ForegroundColor $Color
}

$frame = 0
$lastError = $null
while ($true) {
    if ($CoordinatorProcessId -and -not (Get-Process -Id $CoordinatorProcessId -ErrorAction SilentlyContinue)) {
        Write-Host "`nDAN Coordinator process stopped unexpectedly." -ForegroundColor Red
        break
    }
    if ($GatewayProcessId -and -not (Get-Process -Id $GatewayProcessId -ErrorAction SilentlyContinue)) {
        Write-Host "`nDAN API process stopped unexpectedly." -ForegroundColor Red
        break
    }
    $frame++
    $health = $null
    $healthy = $false
    try {
        $health = Invoke-RestMethod -Uri "$base/health" -Headers $headers -TimeoutSec 3
        $healthy = $health.status -eq 'ok'
        $lastError = $null
    } catch { $lastError = $_.Exception.Message }

    $metricsText = $null
    try { $metricsText = Invoke-RestMethod -Uri "$base/metrics" -Headers $headers -TimeoutSec 3 } catch {}
    $byReplica = if ($metricsText) { ConvertFrom-Prometheus -Text $metricsText } else { @{} }

    Clear-Host
    $width = 66
    $title = if ($Model) { "DAN Coordinator - $Model" } else { 'DAN Coordinator' }
    Write-Center -Text $title -Width $width -Color Cyan
    Write-Center -Text ('=' * $title.Length) -Width $width -Color DarkCyan
    Write-Host ''
    $statusColor = if ($healthy) { 'Green' } else { 'Red' }
    $statusText = if ($healthy) { 'READY' } else { 'STARTING / UNAVAILABLE' }
    Write-Host "  API      : http://$Listen"  -ForegroundColor Gray
    Write-Host "  Network  : $ProviderNetwork" -ForegroundColor Gray
    Write-Host ("  Status   : {0}" -f $statusText) -ForegroundColor $statusColor
    if (-not $healthy -and $lastError) {
        Write-Host "  Detail   : $lastError" -ForegroundColor DarkYellow
    }
    Write-Host ''

    if ($byReplica.Count -eq 0) {
        Write-Host '  Waiting for the coordinator and providers to come online...' -ForegroundColor DarkGray
    }
    foreach ($replica in ($byReplica.Keys | Sort-Object)) {
        $m = $byReplica[$replica]
        $up = $m['dan_replica_up'] -eq 1
        $available = $m['dan_replica_available'] -eq 1
        $active = $m['dan_active_request'] -eq 1
        $speculative = $m['dan_speculative_enabled'] -eq 1
        Write-Host ("  Replica $replica " + ('-' * 50)) -ForegroundColor DarkCyan
        if (-not $up) {
            Write-Host '    unreachable' -ForegroundColor Red
            continue
        }
        $state = if ($active) { 'GENERATING' } elseif ($available) { 'IDLE / READY' } else { 'REFORMING' }
        $stateColor = if ($active) { 'Yellow' } elseif ($available) { 'Green' } else { 'DarkYellow' }
        Write-Host ("    State           : {0}" -f $state) -ForegroundColor $stateColor
        $queueDepth = [int]($m['dan_queue_depth'])
        $queueCap = [int]($m['dan_queue_capacity'])
        $queuePercent = if ($queueCap -gt 0) { 100.0 * $queueDepth / $queueCap } else { 0 }
        Write-Host ("    Queue           : [{0}] {1}/{2}" -f (Format-Bar $queuePercent 20), $queueDepth, $queueCap) -ForegroundColor Gray
        Write-Host ("    Tokens/sec      : {0:N2}" -f $m['dan_generated_tokens_per_second']) -ForegroundColor Gray
        Write-Host ("    Tokens total    : {0:N0}" -f $m['dan_generated_tokens_total']) -ForegroundColor Gray
        Write-Host ("    Latency p50/p95 : {0:N0} ms / {1:N0} ms" -f $m['dan_request_latency_p50_ms'], $m['dan_request_latency_p95_ms']) -ForegroundColor Gray
        Write-Host ("    Requests ok/fail: {0:N0} / {1:N0}" -f $m['dan_requests_completed_total'], $m['dan_requests_failed_total']) -ForegroundColor Gray
        Write-Host ("    Reformations    : {0:N0}" -f $m['dan_replica_reformations_total']) -ForegroundColor Gray
        if ($speculative) {
            $proposed = $m['dan_draft_tokens_proposed_total']
            $accepted = $m['dan_draft_tokens_accepted_total']
            $acceptPct = if ($proposed -gt 0) { 100.0 * $accepted / $proposed } else { 0 }
            Write-Host ("    Speculative     : pipeline depth {0}, {1:N0} rounds, {2:N0}% draft accept" -f $m['dan_pipeline_depth'], $m['dan_speculative_rounds_total'], $acceptPct) -ForegroundColor Magenta
        } else {
            Write-Host '    Speculative     : off' -ForegroundColor DarkGray
        }
        Write-Host ''
    }
    Write-Host ('-' * $width) -ForegroundColor DarkGray
    if ($LogDirectory) { Write-Host "  Logs: $LogDirectory" -ForegroundColor DarkGray }
    Write-Host '  Press Ctrl+C to stop DAN.' -ForegroundColor DarkGray
    Start-Sleep -Seconds 1
}
