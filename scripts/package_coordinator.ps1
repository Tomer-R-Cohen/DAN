[CmdletBinding()]
param([string]$BuildDirectory, [string]$LlamaDirectory, [string]$Model)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build-msvc\Release' }
if (-not $LlamaDirectory) { $LlamaDirectory = Join-Path $root 'build\windows-release-input\llama' }
if (-not $Model) { $Model = Join-Path $root 'build\simple-model\SmolLM2-360M-Instruct-Q4_K_M.gguf' }
$source = Join-Path $BuildDirectory 'dan-coordinator.exe'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing $source" }
if ((Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash -ne
    '16C7F1667FEA34BACAD196A57B548EFFCB37614DB4AB5677A20C8C7B823B9E63') {
    throw 'SmolLM2 test model SHA-256 mismatch'
}
$stage = Join-Path $root 'build\DAN-Coordinator-Windows-x64'
$zip = "$stage.zip"
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'runtime'),(Join-Path $stage 'models'),(Join-Path $stage 'config') | Out-Null
Copy-Item -LiteralPath $source -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\COORDINATOR_WINDOWS.txt') `
    -Destination (Join-Path $stage 'README.txt')
Copy-Item -LiteralPath $Model -Destination (Join-Path $stage 'models')
Copy-Item -LiteralPath (Join-Path $root 'config\smollm2-test.manifest') -Destination (Join-Path $stage 'config')
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
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $stage -DestinationPath $zip
Write-Host "Created $zip"
