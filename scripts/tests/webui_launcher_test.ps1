# Runs the launcher with upstream/network calls stubbed; never launches the UI.
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot '..\Start-DAN-WebUI.ps1'
$tokens = $null; $errors = $null
[System.Management.Automation.Language.Parser]::ParseFile($launcher, [ref]$tokens, [ref]$errors) | Out-Null
if ($errors.Count) { throw ($errors | Out-String) }
function Invoke-RestMethod { @{ status = 'ok' } }
function uvx {
    $global:danWebUICaptured = @{}
    foreach ($item in Get-ChildItem Env:) { $global:danWebUICaptured[$item.Name] = $item.Value }
    $global:LASTEXITCODE = 0
}
$saved = [Environment]::GetEnvironmentVariable('DEFAULT_MODEL_PARAMS', 'Process')
try {
    Remove-Item Env:DEFAULT_MODEL_PARAMS -ErrorAction SilentlyContinue
    & $launcher -DataDir 'C:\DAN test data'
    if (Test-Path Env:DEFAULT_MODEL_PARAMS) { throw 'Unset environment variable leaked after success' }
    $params = $global:danWebUICaptured.DEFAULT_MODEL_PARAMS | ConvertFrom-Json
    $interface = $global:danWebUICaptured.DEFAULT_INTERFACE_SETTINGS | ConvertFrom-Json
    $meta = $global:danWebUICaptured.DEFAULT_MODEL_METADATA | ConvertFrom-Json
    if ($params.function_calling -ne 'native' -or -not $interface.system) { throw 'Missing native agent defaults' }
    if (-not $meta.capabilities.builtin_tools -or $meta.capabilities.file_context) { throw 'Incorrect file tool gates' }
    foreach ($feature in 'web_search', 'code_interpreter') {
        if ($feature -notin $meta.defaultFeatureIds -or -not $meta.capabilities.$feature) {
            throw "Missing default feature: $feature"
        }
    }
    if ($global:danWebUICaptured.DATA_DIR -ne 'C:\DAN test data') { throw 'Data path with spaces changed' }
    if ($global:danWebUICaptured.USE_CUDA_DOCKER -ne 'false') { throw 'Embeddings must stay off the inference GPU' }
    $env:DEFAULT_MODEL_PARAMS = 'existing personal setting'
    function uvx { $global:LASTEXITCODE = 9 }
    $failed = $false
    try { & $launcher } catch { $failed = $true }
    if (-not $failed -or $env:DEFAULT_MODEL_PARAMS -ne 'existing personal setting') {
        throw 'Launcher did not report failure or restore the original setting'
    }
    'PASS: native tool defaults and environment cleanup; no UI launched.'
} finally {
    if ($null -eq $saved) { Remove-Item Env:DEFAULT_MODEL_PARAMS -ErrorAction SilentlyContinue }
    else { [Environment]::SetEnvironmentVariable('DEFAULT_MODEL_PARAMS', $saved, 'Process') }
    Remove-Variable -Name danWebUICaptured -Scope Global -ErrorAction SilentlyContinue
}
