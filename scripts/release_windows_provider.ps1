[CmdletBinding()]
param(
    [string]$BuildDirectory,
    [string]$LlamaSource
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build-provider-release-v1.0.1' }
if (-not $LlamaSource) { $LlamaSource = Join-Path $root 'build\provider-owned-release\llama.cpp' }
$work = Join-Path $root 'build\windows-release-input'
New-Item -ItemType Directory -Force -Path $work | Out-Null

function Fetch([string]$Url, [string]$Name, [string]$Sha256) {
    $path = Join-Path $work $Name
    if (-not (Test-Path -LiteralPath $path) -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $Sha256) {
        Invoke-WebRequest -Uri $Url -OutFile $path
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA-256 mismatch for $Name"
    }
    return $path
}

& (Join-Path $PSScriptRoot 'build_provider_owned.ps1') -Cuda `
    -BuildDirectory $BuildDirectory -LlamaSource $LlamaSource

$tailscale = Fetch `
    'https://pkgs.tailscale.com/stable/tailscale-setup-1.102.3-amd64.msi' `
    'tailscale.msi' `
    '03AC8183C6E3CE276E9B44281EBE7E4C02AEF28A971034CA170C4B665DF42DCE'
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
    -BuildDirectory $BuildDirectory -TailscaleInstaller $tailscale -RuntimeDll $runtime
