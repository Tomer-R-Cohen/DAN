[CmdletBinding()]
param(
    [string]$Listen = $(if ($env:DAN_API_LISTEN) { $env:DAN_API_LISTEN } else { '127.0.0.1:8080' }),
    [string]$ApiKey = $env:DAN_API_KEY,
    [ValidateSet('libp2p', 'tailscale')][string]$ProviderNetwork = 'libp2p'
)

$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$coordinatorExe = Join-Path $package 'dan-coordinator.exe'
$gatewayExe = Join-Path $package 'dan-api-gateway.exe'
$manifest = Join-Path $package 'config\active-model.json'
foreach ($file in @($coordinatorExe, $gatewayExe, $manifest)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
$data = Join-Path $package 'data'
$logs = Join-Path $data 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$model = (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).model_id
if (-not $model) { throw 'The active model has no model_id' }
$env:DAN_API_KEY = $ApiKey
$coordinatorArguments = @('--manifest', 'config\active-model.json', '--metadata-cache',
    'data\model-index.tmp', '--listen', '127.0.0.1:50100')
if ($ProviderNetwork -eq 'libp2p') {
    $coordinatorArguments += @('--provider-listen', '127.0.0.1:50201',
        '--provider-peer-auth', '--packaged-network')
} else {
    $tailscale = Get-Command tailscale.exe -ErrorAction Stop
    $privateAddress = (& $tailscale.Source ip -4 | Select-Object -First 1).Trim()
    $parsed = $null
    if (-not [Net.IPAddress]::TryParse($privateAddress, [ref]$parsed)) {
        throw 'Tailscale did not report a private IPv4 address'
    }
    $coordinatorArguments += @('--provider-listen', "$privateAddress`:50201",
        '--ring-return', "$privateAddress`:50205")
}

$powerHeld = $false
if ($env:OS -eq 'Windows_NT') {
    Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class DANServicePower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
    $powerHeld = [DANServicePower]::SetThreadExecutionState(2147483649) -ne 0
    if (-not $powerHeld) { throw 'Could not prevent Windows from sleeping while DAN is running' }
}

$coordinator = Start-Process -FilePath $coordinatorExe -WorkingDirectory $package `
    -ArgumentList $coordinatorArguments `
    -RedirectStandardOutput (Join-Path $logs 'coordinator.out.log') `
    -RedirectStandardError (Join-Path $logs 'coordinator.err.log') `
    -WindowStyle Hidden -PassThru
try {
    $gateway = Start-Process -FilePath $gatewayExe -WorkingDirectory $package `
        -ArgumentList @('-listen', $Listen, '-coordinator', '127.0.0.1:50100',
            '-model', $model) `
        -RedirectStandardOutput (Join-Path $logs 'api.out.log') `
        -RedirectStandardError (Join-Path $logs 'api.err.log') `
        -WindowStyle Hidden -PassThru
    Write-Host "DAN API starting at http://$Listen"
    Write-Host "Provider network: $ProviderNetwork"
    Write-Host "Readiness: http://$Listen/health"
    Write-Host "Press Ctrl+C to stop. Logs: $logs"
    while (-not $coordinator.HasExited -and -not $gateway.HasExited) {
        Start-Sleep -Milliseconds 500
        $coordinator.Refresh(); $gateway.Refresh()
    }
    if ($coordinator.HasExited) { throw "Coordinator stopped; see $logs" }
    if ($gateway.ExitCode -ne 0) { throw "DAN API stopped; see $logs" }
} finally {
    if ($gateway -and -not $gateway.HasExited) { Stop-Process -Id $gateway.Id -Force }
    if (-not $coordinator.HasExited) { Stop-Process -Id $coordinator.Id -Force }
    if ($powerHeld) { [void][DANServicePower]::SetThreadExecutionState(2147483648) }
}
