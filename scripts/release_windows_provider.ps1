[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CoordinatorPeer,
    [string]$BuildDirectory,
    [string]$LlamaSource
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build-provider-release-v1.0.1' }
if (-not $LlamaSource) { $LlamaSource = Join-Path $root 'build\provider-owned-release\llama.cpp' }
& (Join-Path $PSScriptRoot 'build_provider_owned.ps1') -Cuda `
    -BuildDirectory $BuildDirectory -LlamaSource $LlamaSource

$sidecarDirectory = Join-Path $root 'build\sidecar'
& (Join-Path $PSScriptRoot 'build_sidecar.ps1') -OutputDirectory $sidecarDirectory
if ($LASTEXITCODE -ne 0) { throw 'Sidecar build failed' }
$sidecar = Join-Path $sidecarDirectory 'dan-sidecar-windows-amd64.exe'
$runtime = (Get-ChildItem -LiteralPath (Join-Path $BuildDirectory 'bin\Release') `
    -Filter '*.dll' -File).FullName
$cudaCache = Select-String -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt') `
    -Pattern '^CUDAToolkit_BIN_DIR:PATH=(.+)$'
if (-not $cudaCache) { throw 'CUDA toolkit location is missing from CMakeCache.txt' }
$cudaBin = $cudaCache.Matches[0].Groups[1].Value
$runtime += (Get-ChildItem -LiteralPath (Join-Path $cudaBin 'x64') -File |
    Where-Object Name -Match '^(cublas|cublasLt|cudart)64_\d+\.dll$').FullName

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = if (Test-Path -LiteralPath $vswhere) {
    & $vswhere -latest -property installationPath
}
if (-not $visualStudio) { throw 'Visual Studio runtime files were not found' }
$crt = Get-ChildItem (Join-Path $visualStudio 'VC\Redist\MSVC') -Directory -Recurse |
    Where-Object { $_.FullName -match '\\x64\\Microsoft\.VC\d+\.CRT$' } |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $crt) { throw 'Visual C++ x64 runtime directory was not found' }
$runtime += (Get-ChildItem -LiteralPath $crt.FullName -Filter '*.dll' -File).FullName

& (Join-Path $PSScriptRoot 'package_provider.ps1') `
    -BuildDirectory $BuildDirectory -Sidecar $sidecar `
    -CoordinatorPeer $CoordinatorPeer -RuntimeDll $runtime
