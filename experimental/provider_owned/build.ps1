param(
    [string]$LlamaSource = "",
    [string]$BuildDirectory = "",
    [switch]$Cuda
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$revision = '95ef7fc16054e63b427a3ef00188e055ef7586d8'
if (-not $LlamaSource) { $LlamaSource = Join-Path $root 'build\provider-owned-v0\llama.cpp' }
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build\provider-owned-v0\stage-build' }
$patch = Join-Path $PSScriptRoot 'llama-provider-owned-v0.patch'

if (-not (Test-Path -LiteralPath (Join-Path $LlamaSource '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LlamaSource) | Out-Null
    git clone https://github.com/ggml-org/llama.cpp.git $LlamaSource
    git -C $LlamaSource checkout $revision
}
if ((git -C $LlamaSource rev-parse HEAD).Trim() -ne $revision) {
    throw "llama.cpp must be at $revision"
}
git -C $LlamaSource apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) { git -C $LlamaSource apply --check $patch; git -C $LlamaSource apply $patch }

$cudaValue = if ($Cuda) { 'ON' } else { 'OFF' }
cmake -S $PSScriptRoot -B $BuildDirectory "-DLLAMA_SOURCE_DIR=$LlamaSource" "-DGGML_CUDA=$cudaValue" -DGGML_CCACHE=OFF
cmake --build $BuildDirectory --config Release --target dan-stage-worker --parallel 4
