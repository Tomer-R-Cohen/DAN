[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "License output already exists: $OutputDirectory"
}
$hadToolchain = Test-Path Env:GOTOOLCHAIN
$previousToolchain = $env:GOTOOLCHAIN

Push-Location (Join-Path $root 'sidecar')
try {
    $env:GOTOOLCHAIN = 'auto'
    go run golang.org/x/vuln/cmd/govulncheck@v1.8.0 ./...
    if ($LASTEXITCODE -ne 0) { throw 'Reachable Go dependency vulnerability found' }
    $env:GOTOOLCHAIN = 'go1.25.7'
    go run github.com/google/go-licenses/v2@v2.0.1 save . `
        ./cmd/dan-api-gateway --save_path=$OutputDirectory `
        --ignore=dan/sidecar `
        --ignore=github.com/jackpal/go-nat-pmp `
        --ignore=github.com/multiformats/go-base36
    if ($LASTEXITCODE -ne 0) { throw 'Go dependency license collection failed' }

    foreach ($item in @(
        @{ Module = 'github.com/jackpal/go-nat-pmp'; File = 'LICENSE' },
        @{ Module = 'github.com/multiformats/go-base36'; File = 'LICENSE.md' }
    )) {
        $moduleDirectory = (& go list -m -f '{{.Dir}}' $item.Module).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $moduleDirectory) {
            throw "Cannot resolve license source for $($item.Module)"
        }
        $destination = Join-Path $OutputDirectory ($item.Module.Replace('/', '\'))
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Copy-Item -LiteralPath (Join-Path $moduleDirectory $item.File) -Destination $destination
    }
} finally {
    if ($hadToolchain) { $env:GOTOOLCHAIN = $previousToolchain }
    else { Remove-Item Env:GOTOOLCHAIN -ErrorAction SilentlyContinue }
    Pop-Location
}
