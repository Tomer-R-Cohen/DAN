[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [string[]]$RuntimeDll
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$source = Join-Path $build 'Release\dan-provider-owned-coordinator.exe'
$stage = Join-Path $root 'build\DAN-Coordinator-v1.0.1-Windows-x64'
$zip = "$stage.zip"
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing $source" }

if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $stage 'config\models') -Force | Out-Null
Copy-Item -LiteralPath $source -Destination (Join-Path $stage 'dan-coordinator.exe')
Copy-Item -LiteralPath (Join-Path $root 'config\provider-owned-qwen2.5-0.5b-q4km.json') `
    -Destination (Join-Path $stage 'config\active-model.json')
Copy-Item -Path (Join-Path $root 'config\provider-owned-qwen2.5-*.json') `
    -Destination (Join-Path $stage 'config\models')
Copy-Item -LiteralPath (Join-Path $root 'docs\COORDINATOR_RELEASE_V1.0.1.md') `
    -Destination (Join-Path $stage 'README.txt')
foreach ($dll in $RuntimeDll) {
    if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { throw "Missing runtime file: $dll" }
    Copy-Item -LiteralPath $dll -Destination $stage
}

if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
Write-Host "Created $zip"
