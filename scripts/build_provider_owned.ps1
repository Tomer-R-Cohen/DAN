# Backends of the beta release matrix (docs/BETA_SELECTION_PLAN.md M7):
#   cuda  NVIDIA (Windows/Linux; CUDA toolkit)
#   rocm  AMD (Linux ROCm, or Windows with the AMD HIP SDK); set $env:AMDGPU_TARGETS, e.g. gfx1100
#   metal Apple Silicon (macOS only; run with pwsh)
#   cpu   development and tests
# A backend counts as supported only after provider_owned_stage_reference and the placement
# and replica rehearsals pass on that hardware (BETA_SELECTION_PLAN.md M7).
param(
    [string]$LlamaSource = "",
    [string]$BuildDirectory = "",
    [ValidateSet('cpu', 'cuda', 'rocm', 'metal')][string]$Backend = 'cpu',
    [switch]$Cuda  # older spelling of -Backend cuda
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

if ($Cuda) { $Backend = 'cuda' }
$backendFlags = switch ($Backend) {
    'cuda' { @('-DGGML_CUDA=ON', '-DGGML_CUDA_GRAPHS=ON') }
    'rocm' {
        if (-not $env:AMDGPU_TARGETS) { throw 'set AMDGPU_TARGETS (e.g. gfx1100) for a ROCm build' }
        @('-DGGML_HIP=ON', "-DAMDGPU_TARGETS=$env:AMDGPU_TARGETS")
    }
    'metal' {
        if (-not $IsMacOS) { throw 'Metal builds run on macOS only' }
        @('-DGGML_METAL=ON')
    }
    default { @('-DGGML_CUDA=OFF') }
}
cmake -S $root -B $BuildDirectory `
    "-DDAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR=$LlamaSource" @backendFlags -DGGML_CCACHE=OFF
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
