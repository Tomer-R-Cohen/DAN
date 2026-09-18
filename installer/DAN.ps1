# DAN launcher used by the Start-menu and desktop shortcuts.
#   DAN.ps1 node   share this PC's GPU (dashboard)
#   DAN.ps1 chat   talk to a model running on the DAN network
# The network comes from config\provider.conf (bootstrap=...). A local-test install
# (config\local-network present) instead runs its own network node on this PC.
param([ValidateSet('node', 'chat')][string]$Mode = 'node')

$ErrorActionPreference = 'Stop'
$app = $PSScriptRoot
$config = Join-Path $app 'config\provider.conf'
$sidecar = Join-Path $app 'runtime\dan-sidecar.exe'
$state = Join-Path $env:LOCALAPPDATA 'DAN'
$host.UI.RawUI.WindowTitle = if ($Mode -eq 'node') { 'DAN Node' } else { 'DAN Chat' }

function Stop-WithMessage([string]$Message) {
    Write-Host ''
    Write-Host "  $Message" -ForegroundColor Yellow
    Write-Host ''
    Read-Host '  Press Enter to close'
    exit 1
}

function Get-Setting([string]$Key) {
    @(Get-Content -LiteralPath $config | Where-Object { $_ -like "$Key=*" } |
        ForEach-Object { $_.Substring($Key.Length + 1).Trim() })
}

# Local test network: one DAN network node on this PC, shared by node and chat.
function Get-LocalBootstrap([switch]$Start) {
    $key = Join-Path $state 'network-node.key'
    $ready = Join-Path $state 'network-node.ready'
    $pidFile = Join-Path $state 'network-node.pid'
    New-Item -ItemType Directory -Force -Path (Join-Path $state 'logs') | Out-Null
    $running = $false
    if (Test-Path -LiteralPath $pidFile) {
        $process = Get-Process -Id ([int](Get-Content -LiteralPath $pidFile)) -ErrorAction SilentlyContinue
        $running = $process -and $process.ProcessName -eq 'dan-sidecar'
    }
    if (-not $running) {
        if (-not $Start) { Stop-WithMessage 'Start "DAN Node" first, then open DAN Chat.' }
        Remove-Item -LiteralPath $ready -ErrorAction SilentlyContinue
        $process = Start-Process -FilePath $sidecar -WindowStyle Hidden -PassThru -ArgumentList @(
            '-infra', '-listen', '/ip4/127.0.0.1/tcp/4001', '-key', "`"$key`"",
            '-ready-file', "`"$ready`"", '-log', "`"$(Join-Path $state 'logs\network-node.log')`"")
        Set-Content -LiteralPath $pidFile -Value $process.Id
        $deadline = (Get-Date).AddSeconds(30)
        while (-not (Test-Path -LiteralPath $ready)) {
            if ($process.HasExited) { Stop-WithMessage "The local network node stopped. Log: $state\logs\network-node.log" }
            if ((Get-Date) -gt $deadline) { Stop-WithMessage 'The local network node did not start.' }
            Start-Sleep -Milliseconds 250
        }
    }
    $id = (& $sidecar -key $key -id).Trim()
    "/ip4/127.0.0.1/tcp/4001/p2p/$id"
}

$local = Test-Path -LiteralPath (Join-Path $app 'config\local-network')
$bootstrap = @(@(if ($local) { Get-LocalBootstrap -Start:($Mode -eq 'node') } else { Get-Setting 'bootstrap' }) |
    Where-Object { $_ })
if (-not $bootstrap) { Stop-WithMessage "No DAN network address in $config" }
$catalog = @(Get-Setting 'catalog' | ForEach-Object {
    if ([IO.Path]::IsPathRooted($_)) { $_ } else { Join-Path $app $_ }
})

if ($Mode -eq 'node') {
    $nodeConfig = $config
    if ($local) {
        # Same settings, plus the local network address.
        $nodeConfig = Join-Path $state 'node-local.conf'
        $lines = @(Get-Content -LiteralPath $config | Where-Object { $_ -notlike 'bootstrap=*' -and $_ -notlike 'catalog=*' -and
            $_ -notlike 'stage_worker=*' -and $_ -notlike 'sidecar=*' })
        $lines += @("bootstrap=$($bootstrap[0])") + @($catalog | ForEach-Object { "catalog=$_" }) + @(
            "stage_worker=$(Join-Path $app 'runtime\dan-stage-worker.exe')", "sidecar=$sidecar")
        [IO.File]::WriteAllLines($nodeConfig, $lines, [Text.UTF8Encoding]::new($false))
    }
    try {
        & (Join-Path $app 'dan-provider.exe') --config $nodeConfig
    } finally {
        if ($local) {
            $pidFile = Join-Path $state 'network-node.pid'
            if (Test-Path -LiteralPath $pidFile) {
                Stop-Process -Id ([int](Get-Content -LiteralPath $pidFile)) -Force -ErrorAction SilentlyContinue
                Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
            }
        }
    }
    if ($LASTEXITCODE -ne 0) { Stop-WithMessage "DAN Node stopped (code $LASTEXITCODE). Logs: $state\logs" }
} else {
    $arguments = @{
        Bootstrap = $bootstrap
        Manifest = $catalog  # every model this install knows; the largest that fits wins
        Sidecar = $sidecar
        Client = (Join-Path $app 'dan-client.exe')
        StateDir = (Join-Path $state 'client')
    }
    & (Join-Path $app 'Start-DAN-Client.ps1') @arguments -- --chat --replica
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Chat could not start (code $LASTEXITCODE). Is at least one DAN Node running? Logs: $state\client\logs"
    }
}
