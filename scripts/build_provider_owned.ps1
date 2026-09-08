param(
    [string]$LlamaSource = "",
    [string]$BuildDirectory = "",
    [switch]$Cuda
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$revision = '95ef7fc16054e63b427a3ef00188e055ef7586d8'
if (-not $LlamaSource) { $LlamaSource = Join-Path $root 'build\provider-owned-v1\llama.cpp' }
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build\provider-owned-v1' }
$patch = Join-Path $root 'patches\llama-provider-owned.patch'

if (-not (Test-Path -LiteralPath (Join-Path $LlamaSource '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LlamaSource) | Out-Null
    git clone https://github.com/ggml-org/llama.cpp.git $LlamaSource
    git -C $LlamaSource checkout $revision
}
if ((git -C $LlamaSource rev-parse HEAD).Trim() -ne $revision) {
    throw "llama.cpp must be at $revision"
}
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
git -C $LlamaSource apply --reverse --check $patch 2>$null
$alreadyPatched = $LASTEXITCODE -eq 0
$ErrorActionPreference = $previousPreference
if (-not $alreadyPatched) {
    git -C $LlamaSource apply --check $patch
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp provider-owned patch does not apply cleanly' }
    git -C $LlamaSource apply $patch
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp provider-owned patch failed' }
}

$cudaValue = if ($Cuda) { 'ON' } else { 'OFF' }
cmake -S $root -B $BuildDirectory `
    "-DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=$LlamaSource" `
    "-DGGML_CUDA=$cudaValue" -DGGML_CCACHE=OFF
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
cmake --build $BuildDirectory --config Release `
    --target dan-provider dan-stage-worker dan-provider-owned-coordinator provider_ui_test `
        provider_owned_protocol_test `
        provider_owned_range_model_test provider_owned_formation_test `
        provider_owned_concurrency_client --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Provider-owned build failed' }
ctest --test-dir $BuildDirectory -C Release --output-on-failure `
    -R '^(provider_ui|provider_owned_(protocol|range_model|formation))_test$'
if ($LASTEXITCODE -ne 0) { throw 'Provider-owned tests failed' }
