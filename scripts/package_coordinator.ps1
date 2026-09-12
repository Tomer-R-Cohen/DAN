[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [Parameter(Mandatory = $true)][string]$Sidecar,
    [Parameter(Mandatory = $true)][string]$Gateway,
    [string[]]$Relay,
    [string[]]$RuntimeDll
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
foreach ($address in $Relay) {
    if ($address -notmatch '/p2p/[A-Za-z0-9]+$' -or $address -match '/p2p-circuit/') {
        throw "Relay must be a direct relay peer address: $address"
    }
}
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$source = Join-Path $build 'Release\dan-provider-owned-coordinator.exe'
$stage = Join-Path $root 'build\DAN-Coordinator-v1.0.1-Windows-x64'
$zip = "$stage.zip"
$checksum = "$zip.sha256"
$llamaCache = Select-String -LiteralPath (Join-Path $build 'CMakeCache.txt') `
    -Pattern '^DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR:PATH=(.+)$'
$llamaLicense = if ($llamaCache) {
    Join-Path $llamaCache.Matches[0].Groups[1].Value 'LICENSE'
}
$cudaCache = Select-String -LiteralPath (Join-Path $build 'CMakeCache.txt') `
    -Pattern '^CUDAToolkit_BIN_DIR:PATH=(.+)$'
$cudaLicense = if ($cudaCache) {
    Join-Path (Split-Path -Parent $cudaCache.Matches[0].Groups[1].Value) 'LICENSE'
}
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing $source" }
if (-not (Test-Path -LiteralPath $Sidecar -PathType Leaf)) { throw "Missing $Sidecar" }
if (-not (Test-Path -LiteralPath $Gateway -PathType Leaf)) { throw "Missing $Gateway" }
if (-not $llamaLicense -or -not (Test-Path -LiteralPath $llamaLicense -PathType Leaf)) {
    throw 'Missing llama.cpp license from configured source'
}
if (-not $cudaLicense -or -not (Test-Path -LiteralPath $cudaLicense -PathType Leaf)) {
    throw 'Missing CUDA license from configured toolkit'
}
$runtimeNames = @($RuntimeDll | ForEach-Object { Split-Path -Leaf $_ })
foreach ($pattern in @('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll',
    'ggml-cuda.dll', 'cublas64_*.dll', 'cublasLt64_*.dll', 'cudart64_*.dll',
    'msvcp140.dll', 'vcruntime140.dll')) {
    if (-not ($runtimeNames -like $pattern)) { throw "Missing runtime family: $pattern" }
}
foreach ($dll in $RuntimeDll) {
    if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { throw "Missing runtime file: $dll" }
}

if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $stage 'config\models') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'runtime') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'licenses') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'monitoring') -Force | Out-Null
& (Join-Path $PSScriptRoot 'collect_go_licenses.ps1') `
    -OutputDirectory (Join-Path $stage 'licenses\go')
Copy-Item -LiteralPath $source -Destination (Join-Path $stage 'dan-coordinator.exe')
Copy-Item -LiteralPath $Sidecar -Destination (Join-Path $stage 'runtime\dan-sidecar.exe')
Copy-Item -LiteralPath $Gateway -Destination (Join-Path $stage 'dan-api-gateway.exe')
Copy-Item -LiteralPath (Join-Path $root 'scripts\Start-DAN-Service.ps1') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'scripts\Start-DAN-Service.cmd') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'scripts\Test-DAN-Soak.ps1') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'sidecar\LICENSE') -Destination (Join-Path $stage 'licenses\sidecar-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $root 'sidecar\NOTICE') -Destination (Join-Path $stage 'licenses\sidecar-NOTICE.txt')
Copy-Item -LiteralPath (Join-Path $root 'licenses\Qwen2.5-LICENSE.txt') `
    -Destination (Join-Path $stage 'licenses\Qwen2.5-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $root 'LICENSE') -Destination (Join-Path $stage 'LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $root 'NOTICE') -Destination (Join-Path $stage 'NOTICE.txt')
Copy-Item -LiteralPath $llamaLicense -Destination (Join-Path $stage 'licenses\llama.cpp-LICENSE.txt')
Copy-Item -LiteralPath $cudaLicense -Destination (Join-Path $stage 'licenses\NVIDIA-CUDA-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $root 'config\provider-owned-qwen2.5-0.5b-q4km.json') `
    -Destination (Join-Path $stage 'config\active-model.json')
Copy-Item -Path (Join-Path $root 'config\provider-owned-qwen2.5-*.json') `
    -Destination (Join-Path $stage 'config\models')
if ($Relay) {
    [IO.File]::WriteAllLines((Join-Path $stage 'config\relays.txt'), $Relay,
        [Text.UTF8Encoding]::new($false))
}
Copy-Item -LiteralPath (Join-Path $root 'docs\COORDINATOR_RELEASE_V1.0.1.md') `
    -Destination (Join-Path $stage 'README.txt')
Copy-Item -LiteralPath (Join-Path $root 'docs\DAN_OPERATIONS.md') `
    -Destination (Join-Path $stage 'OPERATIONS.md')
Copy-Item -LiteralPath (Join-Path $root 'docs\P2P_TRANSPORT.md') `
    -Destination (Join-Path $stage 'P2P_TRANSPORT.md')
Copy-Item -LiteralPath (Join-Path $root 'deploy\prometheus\dan-alerts.yml') `
    -Destination (Join-Path $stage 'monitoring')
foreach ($dll in $RuntimeDll) {
    Copy-Item -LiteralPath $dll -Destination $stage
}

if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
if (Test-Path -LiteralPath $checksum) { Remove-Item -LiteralPath $checksum -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($checksum, "$hash  $(Split-Path -Leaf $zip)`n",
    [Text.UTF8Encoding]::new($false))
Write-Host "Created $zip and $checksum"
