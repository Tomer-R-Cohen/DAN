[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RpcWorker,
    [Parameter(Mandatory = $true)][string]$TailscaleInstaller,
    [string]$BuildDirectory,
    [string[]]$RuntimeDll
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build\Release' }
$stage = Join-Path $root 'build\DAN-Provider-Windows-x64'
$zip = "$stage.zip"
$required = @('dan-provider.exe', 'managed_provider.exe')
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $BuildDirectory $name))) { throw "Missing $name in $BuildDirectory" }
}
if (-not (Test-Path -LiteralPath $RpcWorker -PathType Leaf)) { throw "Missing RPC worker: $RpcWorker" }
if (-not (Test-Path -LiteralPath $TailscaleInstaller -PathType Leaf)) { throw "Missing Tailscale installer: $TailscaleInstaller" }
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'runtime') | Out-Null
Copy-Item -LiteralPath (Join-Path $BuildDirectory 'dan-provider.exe') -Destination $stage
Copy-Item -LiteralPath (Join-Path $BuildDirectory 'managed_provider.exe') -Destination (Join-Path $stage 'runtime')
Copy-Item -LiteralPath $RpcWorker -Destination (Join-Path $stage 'runtime\rpc-server.exe')
Copy-Item -LiteralPath $TailscaleInstaller -Destination (Join-Path $stage 'runtime\tailscale.msi')
foreach ($dll in $RuntimeDll) { Copy-Item -LiteralPath $dll -Destination (Join-Path $stage 'runtime') }
Copy-Item -LiteralPath (Join-Path $root 'scripts\setup_provider.ps1') -Destination (Join-Path $stage 'runtime\setup-provider.ps1')
Copy-Item -LiteralPath (Join-Path $root 'docs\FRIENDS_TESTNET_WINDOWS.md') -Destination (Join-Path $stage 'README.txt')
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
Write-Host "Created $zip"
