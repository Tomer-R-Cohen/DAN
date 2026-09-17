# Runs dan-client against the DAN network from any home connection, no coordinator and no
# port forwarding. Starts this machine's client sidecar (its own identity, DHT client,
# relay reservation, candidate API, ring return), runs dan-client --discover, then stops it.
#
#   .\Start-DAN-Client.ps1 -Bootstrap /ip4/VPS_IP/tcp/4001/p2p/VPS_PEERID `
#       -Manifest config\provider-owned-qwen2.5-0.5b-q4km.json -- --prompt "Hello" --tokens 32
#
# Everything after -- goes to dan-client (placement, prompts, reports).
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory)][string[]]$Bootstrap,
    [string[]]$Relay,
    [Parameter(Mandatory)][string]$Manifest,
    [string]$Sidecar = (Join-Path $PSScriptRoot '..\runtime\dan-sidecar.exe'),
    [string]$Client = (Join-Path $PSScriptRoot '..\dan-client.exe'),
    [string]$StateDir = (Join-Path $env:LOCALAPPDATA 'DAN\client'),
    [switch]$SimulateNat,  # test only
    [switch]$KeepSidecar,
    [Parameter(ValueFromRemainingArguments)][string[]]$ClientArguments
)

$ErrorActionPreference = 'Stop'
foreach ($file in @($Sidecar, $Client, $Manifest)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
if (-not $Relay) { $Relay = $Bootstrap }
New-Item -ItemType Directory -Force -Path (Join-Path $StateDir 'logs') | Out-Null

function Get-FreePort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try { $listener.LocalEndpoint.Port } finally { $listener.Stop() }
}

$apiPort = Get-FreePort
do { $returnPort = Get-FreePort } while ($returnPort -eq $apiPort)
$ready = Join-Path $StateDir "sidecar-ready-$PID.txt"
Remove-Item -LiteralPath $ready -ErrorAction SilentlyContinue
$arguments = @('-key', (Join-Path $StateDir 'identity.key'), '-listen', '/ip4/0.0.0.0/tcp/0',
    '-dht', 'client', '-reachability', 'private',
    '-candidate-api', "127.0.0.1:$apiPort", '-ring-inbound', "127.0.0.1:$returnPort",
    '-ready-file', $ready, '-log', (Join-Path $StateDir 'logs\sidecar.log'))
foreach ($peer in $Bootstrap) { $arguments += @('-bootstrap', $peer) }
foreach ($peer in $Relay) { $arguments += @('-relay', $peer) }
if ($SimulateNat) { $arguments += '-simulate-nat' }

$sidecarProcess = Start-Process -FilePath $Sidecar -ArgumentList $arguments -PassThru -NoNewWindow `
    -RedirectStandardOutput (Join-Path $StateDir 'logs\sidecar.out') `
    -RedirectStandardError (Join-Path $StateDir 'logs\sidecar.err')
try {
    $deadline = (Get-Date).AddSeconds(90)
    while (-not (Test-Path -LiteralPath $ready)) {
        if ($sidecarProcess.HasExited) { throw "the client sidecar stopped; see $StateDir\logs\sidecar.log" }
        if ((Get-Date) -gt $deadline) { throw "the client sidecar did not start; see $StateDir\logs\sidecar.log" }
        Start-Sleep -Milliseconds 250
    }
    Write-Host "DAN client identity: $((Get-Content -LiteralPath $ready -TotalCount 1).Trim())"
    # A plain array, not splatting: PowerShell 5.1 splatting mangles options like '--persistent'.
    $clientCommand = @('--manifest', $Manifest, '--discover', "127.0.0.1:$apiPort") + @($ClientArguments)
    # 'Continue': when output is redirected, PowerShell 5.1 turns dan-client's stderr
    # progress lines into errors.
    $ErrorActionPreference = 'Continue'
    & $Client $clientCommand
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
} finally {
    Remove-Item -LiteralPath $ready -ErrorAction SilentlyContinue
    if (-not $KeepSidecar -and -not $sidecarProcess.HasExited) { Stop-Process -Id $sidecarProcess.Id -Force }
}
exit $exitCode
