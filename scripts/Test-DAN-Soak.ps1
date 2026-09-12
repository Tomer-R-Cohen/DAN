[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://127.0.0.1:8080',
    [string]$ApiKey = $env:DAN_API_KEY,
    [string]$Model,
    [TimeSpan]$Duration = '1.00:00:00',
    [int]$IntervalSeconds = 1,
    [int]$RecoveryTimeoutSeconds = 180,
    [int]$ExpectedOutages = 0
)

$ErrorActionPreference = 'Stop'
if (-not $Model) {
    $manifest = Join-Path $PSScriptRoot 'config\active-model.json'
    if (Test-Path -LiteralPath $manifest) {
        $Model = (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).model_id
    }
}
if (-not $Model -or $Duration -le [TimeSpan]::Zero -or $IntervalSeconds -lt 0 `
    -or $RecoveryTimeoutSeconds -lt 1 -or $ExpectedOutages -lt 0) {
    throw 'Provide a model, positive duration/recovery timeout, and non-negative interval/outage count'
}
$headers = @{}
if ($ApiKey) { $headers.Authorization = "Bearer $ApiKey" }

function Wait-DANReady {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    do {
        try {
            $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Headers $headers
            if ($health.status -eq 'ok') { return }
        } catch {}
        Start-Sleep -Seconds 1
    } while ($timer.Elapsed.TotalSeconds -lt $RecoveryTimeoutSeconds)
    throw "DAN did not recover within $RecoveryTimeoutSeconds seconds"
}

function Get-MetricValues([string]$Text, [string]$Name) {
    $pattern = '(?m)^' + [regex]::Escape($Name) + '\{[^}]*\} ([-+0-9.eE]+)\r?$'
    return @([regex]::Matches($Text, $pattern) | ForEach-Object {
        [double]::Parse($_.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
    })
}

function Test-DANOutageError([Management.Automation.ErrorRecord]$Failure) {
    return $Failure.ErrorDetails.Message -match
        '"message"\s*:\s*"(?:replica_unavailable|provider_disconnected|provider_failure:)'
}

$powerHeld = $false
if ($env:OS -eq 'Windows_NT') {
    Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class DANSoakPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
    $powerHeld = [DANSoakPower]::SetThreadExecutionState(2147483649) -ne 0
    if (-not $powerHeld) { throw 'Could not prevent Windows from sleeping during the soak' }
}

try {
$timer = [Diagnostics.Stopwatch]::StartNew()
$requests = 0
$outages = 0
while ($timer.Elapsed -lt $Duration) {
    try {
        $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Headers $headers
        if ($health.status -ne 'ok') { throw 'DAN is not ready' }
    } catch {
        ++$outages
        Wait-DANReady
    }

    $body = @{
        model = $Model
        messages = @(@{ role = 'user'; content = 'Reply with one short word.' })
        max_tokens = 8
    } | ConvertTo-Json -Compress -Depth 4
    try {
        $response = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/chat/completions" `
            -Headers $headers -ContentType 'application/json' -Body $body
    } catch {
        if (-not (Test-DANOutageError $_)) { throw }
        ++$outages
        Wait-DANReady
        continue
    }
    if (-not $response.choices[0].message.content) { throw 'DAN returned an empty completion' }
    ++$requests

    $metrics = Invoke-RestMethod -Uri "$BaseUrl/metrics" -Headers $headers
    $sessions = Get-MetricValues $metrics 'dan_resident_sessions'
    $kvBytes = Get-MetricValues $metrics 'dan_kv_memory_bytes'
    if (-not $sessions.Count -or -not $kvBytes.Count `
        -or ($sessions | Where-Object { $_ -ne 0 }) `
        -or ($kvBytes | Where-Object { $_ -ne 0 })) {
        throw 'Stateless request leaked a resident session or KV allocation'
    }
    if ($requests % 100 -eq 0) { Write-Host "DAN soak: $requests requests, $outages outages" }
    if ($IntervalSeconds) { Start-Sleep -Seconds $IntervalSeconds }
}

if ($outages -ne $ExpectedOutages) {
    throw "Expected $ExpectedOutages recovered outages, observed $outages"
}
Write-Host "DAN soak passed: $requests requests, $outages recovered outages, duration $Duration"
} finally {
    if ($powerHeld) { [void][DANSoakPower]::SetThreadExecutionState(2147483648) }
}
