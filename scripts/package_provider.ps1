[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [Parameter(Mandatory = $true)][string]$TailscaleInstaller,
    [string[]]$RuntimeDll
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$exeDir = Join-Path $build 'Release'
$stage = Join-Path $root 'build\DAN-Provider-v1.0.1-Windows-x64'
$zip = "$stage.zip"

foreach ($name in @('dan-provider.exe', 'dan-stage-worker.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $exeDir $name) -PathType Leaf)) {
        throw "Missing $name in $exeDir"
    }
}
if (-not (Test-Path -LiteralPath $TailscaleInstaller -PathType Leaf)) {
    throw "Missing Tailscale installer: $TailscaleInstaller"
}

if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $stage 'runtime') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $exeDir 'dan-provider.exe') -Destination $stage
Copy-Item -LiteralPath (Join-Path $exeDir 'dan-stage-worker.exe') -Destination (Join-Path $stage 'runtime')
Copy-Item -LiteralPath $TailscaleInstaller -Destination (Join-Path $stage 'runtime\tailscale.msi')
foreach ($dll in $RuntimeDll) {
    if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { throw "Missing runtime file: $dll" }
    Copy-Item -LiteralPath $dll -Destination (Join-Path $stage 'runtime')
    if ((Split-Path -Leaf $dll) -match '^(msvcp|vcruntime|concrt|vccorlib)') {
        Copy-Item -LiteralPath $dll -Destination $stage
    }
}
Copy-Item -LiteralPath (Join-Path $root 'docs\CONTRIBUTOR_RELEASE_V1.0.1.md') `
    -Destination (Join-Path $stage 'README.txt')

if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
Write-Host "Created $zip"
