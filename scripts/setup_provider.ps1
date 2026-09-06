[CmdletBinding()]
param(
    [string]$Coordinator,
    [string]$AdvertiseAddress,
    [string]$ProviderName = $env:COMPUTERNAME,
    [string]$RpcWorker,
    [int]$ReserveVramMiB = 1536
)

$ErrorActionPreference = 'Stop'

function Fail([string]$Message) {
    Write-Error $Message
    exit 1
}

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -notin @('AMD64', 'ARM64')) {
    Fail 'DAN Provider requires 64-bit Windows 10 or 11 on an x64 PC.'
}
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { Fail 'This package supports Windows x64 only.' }

$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) {
    $driverSmi = Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    if (Test-Path -LiteralPath $driverSmi -PathType Leaf) { $smi = Get-Item -LiteralPath $driverSmi }
}
if (-not $smi) { Fail 'nvidia-smi was not found. Install or repair the NVIDIA driver, then try again.' }
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) { Fail 'curl.exe was not found. Install current Windows updates, then try again.' }
$smiPath = if ($smi.Source) { $smi.Source } else { $smi.FullName }
$gpu = & $smiPath --query-gpu=index,name,memory.total,uuid --format=csv,noheader,nounits 2>&1
if ($LASTEXITCODE -ne 0 -or -not $gpu) { Fail "NVIDIA GPU detection failed: $gpu" }
Write-Host "Detected NVIDIA GPU: $($gpu[0])"

$packageDir = Split-Path -Parent $PSScriptRoot
$providerCandidates = @(
    (Join-Path $PSScriptRoot 'dan-provider.exe'),
    (Join-Path $packageDir 'dan-provider.exe'),
    (Join-Path $packageDir 'build\Release\dan-provider.exe')
)
$provider = $providerCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $provider) { Fail 'dan-provider.exe was not found in the package or build\Release directory.' }
$managed = @((Join-Path (Split-Path -Parent $provider) 'runtime\managed_provider.exe'),
    (Join-Path (Split-Path -Parent $provider) 'managed_provider.exe')) |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $managed) {
    Fail 'managed_provider.exe must be beside dan-provider.exe.'
}
if (-not $RpcWorker) { $RpcWorker = Join-Path (Split-Path -Parent $provider) 'runtime\rpc-server.exe' }
if (-not (Test-Path -LiteralPath $RpcWorker -PathType Leaf)) {
    Fail 'rpc-server.exe was not found. Put the pinned compatible prebuilt worker beside dan-provider.exe or pass -RpcWorker.'
}

if (-not $Coordinator) { $Coordinator = Read-Host 'Coordinator private address (HOST:PORT)' }
if ($Coordinator -notmatch '^.+:[1-9][0-9]{0,4}$') { Fail 'Coordinator must be HOST:PORT.' }
if (-not $AdvertiseAddress) {
    $tailscale = Get-Command tailscale.exe -ErrorAction SilentlyContinue
    if ($tailscale) { $AdvertiseAddress = (& $tailscale.Source ip -4 2>$null | Select-Object -First 1) }
}
if (-not $AdvertiseAddress) { $AdvertiseAddress = Read-Host 'This PC private/Tailscale IPv4 address' }
if ($AdvertiseAddress -match '[\r\n]' -or $Coordinator -match '[\r\n]' -or $RpcWorker -match '[\r\n]') {
    Fail 'Configuration values cannot contain newlines.'
}
if ($ProviderName -notmatch '^[A-Za-z0-9._-]+$') { Fail 'Provider name may contain only letters, numbers, dot, underscore, and dash.' }

$stateDir = Join-Path $env:LOCALAPPDATA 'DAN'
$modelDir = Join-Path $stateDir 'models'
$logDir = Join-Path $stateDir 'logs'
New-Item -ItemType Directory -Force -Path $modelDir, $logDir | Out-Null
$config = Join-Path $stateDir 'provider.conf'
if (Test-Path -LiteralPath $config) {
    Write-Host "Keeping existing configuration: $config"
} else {
    $lines = @(
        "coordinator=$Coordinator",
        "provider_name=$ProviderName",
        'network=tailscale',
        "advertise_host=$AdvertiseAddress",
        "cache_dir=$modelDir",
        "rpc_worker=$RpcWorker",
        "nvidia_smi=$smiPath",
        'worker_port=50052',
        "reserve_vram_mib=$ReserveVramMiB",
        'reconnect_seconds=2'
    )
    [IO.File]::WriteAllLines($config, $lines, [Text.UTF8Encoding]::new($false))
    Write-Host "Created configuration: $config"
}

Write-Host ''
Write-Host 'Checking DAN Provider...'
& $provider --config $config --check
if ($LASTEXITCODE -ne 0) { Fail 'DAN Provider check failed. Fix the reported item and run setup again.' }
Write-Host ''
Write-Host 'DAN Provider setup succeeded.' -ForegroundColor Green
Write-Host 'Start contributing with:'
Write-Host "  `"$provider`""
Write-Host 'Press Ctrl+C or close its window to stop. Cache and identity are preserved.'
