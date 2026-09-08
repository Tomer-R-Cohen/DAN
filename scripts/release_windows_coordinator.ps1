[CmdletBinding()]
param([string]$BuildDirectory)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build-coordinator-release-v1.0.1' }

cmake -S $root -B $BuildDirectory
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
cmake --build $BuildDirectory --config Release --target dan-provider-owned-coordinator
if ($LASTEXITCODE -ne 0) { throw 'Coordinator build failed' }

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = if (Test-Path -LiteralPath $vswhere) {
    & $vswhere -latest -property installationPath
}
if (-not $visualStudio) { throw 'Visual Studio runtime files were not found' }
$crt = Get-ChildItem (Join-Path $visualStudio 'VC\Redist\MSVC') -Directory -Recurse |
    Where-Object { $_.FullName -match '\\x64\\Microsoft\.VC\d+\.CRT$' } |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $crt) { throw 'Visual C++ x64 runtime directory was not found' }
$runtime = (Get-ChildItem -LiteralPath $crt.FullName -Filter '*.dll' -File).FullName

& (Join-Path $PSScriptRoot 'package_coordinator.ps1') `
    -BuildDirectory $BuildDirectory -RuntimeDll $runtime
