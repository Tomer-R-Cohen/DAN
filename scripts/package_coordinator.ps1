[CmdletBinding()]
param([string]$BuildDirectory, [string]$LlamaDirectory, [string]$Manifest,
    [string]$OutputName = 'DAN-Coordinator-Windows-x64',
    [switch]$NoArchive)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build-msvc\Release' }
if (-not $LlamaDirectory) { $LlamaDirectory = Join-Path $root 'build\windows-release-input\llama' }
if (-not $Manifest) { $Manifest = Join-Path $root 'config\smollm2-test.manifest' }
$source = Join-Path $BuildDirectory 'dan-coordinator.exe'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing $source" }
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw "Missing $Manifest" }
$stage = Join-Path $root "build\$OutputName"
$zip = "$stage.zip"
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'runtime'),(Join-Path $stage 'config') | Out-Null
Copy-Item -LiteralPath $source -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\COORDINATOR_WINDOWS.txt') `
    -Destination (Join-Path $stage 'README.txt')
Copy-Item -LiteralPath $Manifest -Destination (Join-Path $stage 'config\managed-model.manifest')
$runtime = @('llama-server.exe','llama-server-impl.dll','llama-common.dll','llama.dll',
    'ggml.dll','ggml-base.dll','ggml-rpc.dll','libomp.dll','mtmd.dll','LICENSE-LLVM-OpenMP')
$runtime += (Get-ChildItem -LiteralPath $LlamaDirectory -Filter 'ggml-cpu-*.dll' -File).Name
foreach ($name in $runtime) {
    Copy-Item -LiteralPath (Join-Path $LlamaDirectory $name) -Destination (Join-Path $stage 'runtime')
}
foreach ($name in @('msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE-llama.cpp')) {
    Copy-Item -LiteralPath (Join-Path $root "build\DAN-Provider-Windows-x64\runtime\$name") `
        -Destination (Join-Path $stage 'runtime')
}
if (-not $NoArchive) {
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -LiteralPath $stage -DestinationPath $zip
    Write-Host "Created $zip"
} else { Write-Host "Created $stage" }
