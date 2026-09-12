[CmdletBinding()]
param([string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'build\gateway' }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
Push-Location (Join-Path $root 'sidecar')
try {
    $env:GOTOOLCHAIN = 'go1.25.7'
    go test ./cmd/dan-api-gateway
    if ($LASTEXITCODE -ne 0) { throw 'Gateway tests failed' }
    $env:CGO_ENABLED = '0'
    $env:GOOS = 'windows'; $env:GOARCH = 'amd64'
    go build -trimpath -buildvcs=false `
        -o (Join-Path $OutputDirectory 'dan-api-gateway-windows-amd64.exe') `
        ./cmd/dan-api-gateway
    if ($LASTEXITCODE -ne 0) { throw 'Windows gateway build failed' }
    $env:GOOS = 'linux'
    go build -trimpath -buildvcs=false `
        -o (Join-Path $OutputDirectory 'dan-api-gateway-linux-amd64') `
        ./cmd/dan-api-gateway
    if ($LASTEXITCODE -ne 0) { throw 'Linux gateway build failed' }
} finally {
    Remove-Item Env:GOTOOLCHAIN, Env:GOOS, Env:GOARCH, Env:CGO_ENABLED -ErrorAction SilentlyContinue
    Pop-Location
}
Get-ChildItem -LiteralPath $OutputDirectory
