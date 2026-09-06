[CmdletBinding()]
param([string]$BuildDirectory)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build\Release' }
$work = Join-Path $root 'build\windows-release-input'
New-Item -ItemType Directory -Force -Path $work | Out-Null

function Fetch([string]$Url, [string]$Name, [string]$Sha256) {
    $path = Join-Path $work $Name
    if (-not (Test-Path -LiteralPath $path) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $Sha256) {
        Invoke-WebRequest -Uri $Url -OutFile $path
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA-256 mismatch for $Name"
    }
    return $path
}

# b10791 resolves exactly to DAN's pinned commit 95ef7fc16054e63b427a3ef00188e055ef7586d8.
$llama = Fetch 'https://github.com/ggml-org/llama.cpp/releases/download/b10791/llama-b10791-bin-win-cuda-12.4-x64.zip' 'llama.zip' '39654A5019BF699CAEC79E32D3A2923BC37DB6F5EED47E1C32F34417A428BC23'
$cuda = Fetch 'https://github.com/ggml-org/llama.cpp/releases/download/b10791/cudart-llama-bin-win-cuda-12.4-x64.zip' 'cudart.zip' '8C79A9B226DE4B3CACFD1F83D24F962D0773BE79F1E7B75C6AF4DED7E32AE1D6'
$tailscale = Fetch 'https://pkgs.tailscale.com/stable/tailscale-setup-1.102.3-amd64.msi' 'tailscale.msi' '03AC8183C6E3CE276E9B44281EBE7E4C02AEF28A971034CA170C4B665DF42DCE'
$license = Fetch 'https://raw.githubusercontent.com/ggml-org/llama.cpp/b10791/LICENSE' 'LICENSE-llama.cpp' '94F29BBED6A22C35B992C5C6EBF0E7C92F13B836B90F36F461C9CF2F0F1D010D'

$llamaDir = Join-Path $work 'llama'
$cudaDir = Join-Path $work 'cudart'
New-Item -ItemType Directory -Force -Path $llamaDir,$cudaDir | Out-Null
Expand-Archive -LiteralPath $llama -DestinationPath $llamaDir -Force
Expand-Archive -LiteralPath $cuda -DestinationPath $cudaDir -Force

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = if (Test-Path -LiteralPath $vswhere) { & $vswhere -latest -property installationPath }
if (-not $visualStudio) { throw 'Visual Studio with the MSVC redistributable files was not found.' }
$crt = Get-ChildItem (Join-Path $visualStudio 'VC\Redist\MSVC') -Directory -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '\\x64\\Microsoft\.VC\d+\.CRT$' } |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $crt) { throw 'Visual C++ x64 redistributable DLL directory was not found.' }

$runtime = @(
    (Join-Path $llamaDir 'ggml.dll'), (Join-Path $llamaDir 'ggml-base.dll'),
    (Join-Path $llamaDir 'ggml-rpc.dll'), (Join-Path $llamaDir 'ggml-cuda.dll'),
    (Join-Path $llamaDir 'libomp.dll'), (Join-Path $llamaDir 'LICENSE-LLVM-OpenMP'),
    $license,
    (Join-Path $cudaDir 'cudart64_12.dll'), (Join-Path $cudaDir 'cublas64_12.dll'),
    (Join-Path $cudaDir 'cublasLt64_12.dll'),
    (Join-Path $crt.FullName 'msvcp140.dll'),
    (Join-Path $crt.FullName 'vcruntime140.dll'),
    (Join-Path $crt.FullName 'vcruntime140_1.dll')
)
$runtime += (Get-ChildItem -LiteralPath $llamaDir -Filter 'ggml-cpu-*.dll' -File).FullName
foreach ($path in $runtime) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing runtime dependency: $path" } }

& (Join-Path $PSScriptRoot 'package_provider.ps1') -BuildDirectory $BuildDirectory `
    -RpcWorker (Join-Path $llamaDir 'ggml-rpc-server.exe') `
    -TailscaleInstaller $tailscale -RuntimeDll $runtime
