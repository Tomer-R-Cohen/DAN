[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [Parameter(Mandatory = $true)][string]$Sidecar,
    [Parameter(Mandatory = $true)][string]$Gateway,
    [Parameter(Mandatory = $true)][string]$CoordinatorPeer,
    [string[]]$Relay,
    [string[]]$RuntimeDll
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$exeDir = Join-Path $build 'Release'
$stage = Join-Path $root 'build\DAN-Provider-v1.0.1-Windows-x64'
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

foreach ($name in @('dan-provider.exe', 'dan-stage-worker.exe',
    'dan-provider-owned-coordinator.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $exeDir $name) -PathType Leaf)) {
        throw "Missing $name in $exeDir"
    }
}
if (-not (Test-Path -LiteralPath $Sidecar -PathType Leaf)) { throw "Missing sidecar: $Sidecar" }
if (-not (Test-Path -LiteralPath $Gateway -PathType Leaf)) { throw "Missing gateway: $Gateway" }
if ($CoordinatorPeer -notmatch '/p2p/[A-Za-z0-9]+') { throw 'CoordinatorPeer must include /p2p/PEER_ID' }
if ($CoordinatorPeer -match '/p2p-circuit/' -and -not $Relay) {
    throw 'A circuit coordinator route requires at least one -Relay reservation address'
}
foreach ($address in $Relay) {
    if ($address -notmatch '/p2p/[A-Za-z0-9]+$' -or $address -match '/p2p-circuit/') {
        throw "Relay must be a direct relay peer address: $address"
    }
}
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
New-Item -ItemType Directory -Path (Join-Path $stage 'runtime') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'config') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'licenses') -Force | Out-Null
& (Join-Path $PSScriptRoot 'collect_go_licenses.ps1') `
    -OutputDirectory (Join-Path $stage 'licenses\go')
Copy-Item -LiteralPath (Join-Path $exeDir 'dan-provider.exe') -Destination $stage
Copy-Item -LiteralPath (Join-Path $exeDir 'dan-stage-worker.exe') -Destination (Join-Path $stage 'runtime')
Copy-Item -LiteralPath (Join-Path $exeDir 'dan-provider-owned-coordinator.exe') `
    -Destination (Join-Path $stage 'runtime')
Copy-Item -LiteralPath $Sidecar -Destination (Join-Path $stage 'runtime\dan-sidecar.exe')
Copy-Item -LiteralPath $Gateway -Destination (Join-Path $stage 'runtime\dan-api-gateway.exe')
Copy-Item -Path (Join-Path $root 'config\provider-owned-qwen2.5-*.json') `
    -Destination (Join-Path $stage 'config')
$providerConfig = @(
    'network=libp2p',
    "coordinator_peer=$CoordinatorPeer",
    'provider_name=windows-pc',
    'stage_worker=runtime\dan-stage-worker.exe',
    'sidecar=runtime\dan-sidecar.exe',
    'reserve_vram_mib=1536'
)
$providerConfig += @($Relay | ForEach-Object { "relay=$_" })
[IO.File]::WriteAllLines((Join-Path $stage 'config\provider.conf'), $providerConfig,
    [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $root 'sidecar\LICENSE') -Destination (Join-Path $stage 'licenses\sidecar-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $root 'sidecar\NOTICE') -Destination (Join-Path $stage 'licenses\sidecar-NOTICE.txt')
Copy-Item -LiteralPath (Join-Path $root 'licenses\Qwen2.5-LICENSE.txt') `
    -Destination (Join-Path $stage 'licenses\Qwen2.5-LICENSE.txt')
Copy-Item -LiteralPath $llamaLicense -Destination (Join-Path $stage 'licenses\llama.cpp-LICENSE.txt')
Copy-Item -LiteralPath $cudaLicense -Destination (Join-Path $stage 'licenses\NVIDIA-CUDA-LICENSE.txt')
foreach ($dll in $RuntimeDll) {
    Copy-Item -LiteralPath $dll -Destination (Join-Path $stage 'runtime')
    if ((Split-Path -Leaf $dll) -match '^(msvcp|vcruntime|concrt|vccorlib)') {
        Copy-Item -LiteralPath $dll -Destination $stage
    }
}
Copy-Item -LiteralPath (Join-Path $root 'docs\CONTRIBUTOR_RELEASE_V1.0.1.md') `
    -Destination (Join-Path $stage 'README.txt')

if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
if (Test-Path -LiteralPath $checksum) { Remove-Item -LiteralPath $checksum -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($checksum, "$hash  $(Split-Path -Leaf $zip)`n",
    [Text.UTF8Encoding]::new($false))
Write-Host "Created $zip and $checksum"
