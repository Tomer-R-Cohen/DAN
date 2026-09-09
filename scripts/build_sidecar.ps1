[CmdletBinding()]
param([string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'build\sidecar' }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
Push-Location (Join-Path $root 'sidecar')
try {
    $env:GOTOOLCHAIN = 'go1.25.7'
    go test ./...
    if ($LASTEXITCODE -ne 0) { throw 'Sidecar tests failed' }
    $env:CGO_ENABLED = '0'
    $env:GOOS = 'windows'; $env:GOARCH = 'amd64'
    go build -trimpath -buildvcs=false -o (Join-Path $OutputDirectory 'dan-sidecar-windows-amd64.exe') .
    if ($LASTEXITCODE -ne 0) { throw 'Windows sidecar build failed' }
    $env:GOOS = 'linux'
    go build -trimpath -buildvcs=false -o (Join-Path $OutputDirectory 'dan-sidecar-linux-amd64') .
    if ($LASTEXITCODE -ne 0) { throw 'Linux sidecar build failed' }
} finally {
    Remove-Item Env:GOTOOLCHAIN, Env:GOOS, Env:GOARCH, Env:CGO_ENABLED -ErrorAction SilentlyContinue
    Pop-Location
}
Get-ChildItem -LiteralPath $OutputDirectory
