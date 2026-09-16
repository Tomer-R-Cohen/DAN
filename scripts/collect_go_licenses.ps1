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

# Accepted risks. Each entry matches one exact advisory ID in one module, and only while the
# Go vulnerability database reports no fixed version. Anything else stays fatal, including
# any other advisory in the same module.
#
# GO-2024-3218 -- https://pkg.go.dev/vuln/GO-2024-3218
#   Content censorship via Kademlia DHT abuse in github.com/libp2p/go-libp2p-kad-dht.
#   The Go vulnerability DB lists every version as affected and no fixed version; GitHub and
#   NVD describe the original issue as affecting <= 0.20.0. DAN uses v0.42.0, so this entry
#   is a risk acceptance for this advisory, not a claim that the issue is fixed.
#   It is the known Sybil / eclipse / content-censorship weakness of Kademlia DHTs: enough
#   attacker-controlled peers can hide or crowd out provider records. DAN accepts that risk
#   for the first decentralized milestone (discovery only; workers still validate every
#   reservation, and routes are checked end to end). Trust and Sybil resistance remain
#   future work. Revisit when a fixed version is published or before a public network.
$acceptedVulnerabilities = @(
    @{ Id = 'GO-2024-3218'; Module = 'github.com/libp2p/go-libp2p-kad-dht' }
)

# Runs govulncheck and fails on every reachable (called) finding that is not accepted above.
function Assert-NoReachableVulnerability {
    $previousEncoding = [Console]::OutputEncoding
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    try {
        $lines = & go run golang.org/x/vuln/cmd/govulncheck@v1.8.0 -format json ./...
        $exitCode = $LASTEXITCODE
    } finally {
        [Console]::OutputEncoding = $previousEncoding
    }
    if ($exitCode -ne 0) { throw 'govulncheck failed to run' }

    # The JSON stream is a sequence of pretty-printed objects, each closing with "}" alone.
    $findings = @()
    $buffer = New-Object System.Text.StringBuilder
    foreach ($line in $lines) {
        [void]$buffer.AppendLine($line)
        if ($line -ceq '}') {
            $message = $buffer.ToString() | ConvertFrom-Json
            if ($message.finding) { $findings += $message.finding }
            [void]$buffer.Clear()
        }
    }
    if ($buffer.ToString().Trim()) { throw 'govulncheck output could not be parsed' }

    # A finding is reachable when its trace starts at a called function (symbol scan level).
    $reachable = @($findings | Where-Object { $_.trace -and $_.trace[0].function })
    $fatal = @{}
    $accepted = @{}
    foreach ($finding in $reachable) {
        $module = $finding.trace[0].module
        $isAccepted = [string]::IsNullOrEmpty($finding.fixed_version) -and @($acceptedVulnerabilities |
            Where-Object { $_.Id -ceq $finding.osv -and $_.Module -ceq $module }).Count -gt 0
        $key = "$($finding.osv) in $module"
        if ($isAccepted) { $accepted[$key] = $true } else { $fatal[$key] = $finding.fixed_version }
    }
    foreach ($key in $accepted.Keys) {
        Write-Warning "Accepted Go vulnerability (documented in collect_go_licenses.ps1): $key -- https://pkg.go.dev/vuln/$($key.Split(' ')[0])"
    }
    if ($fatal.Count -gt 0) {
        foreach ($key in $fatal.Keys) {
            $fixed = if ($fatal[$key]) { "fixed in $($fatal[$key])" } else { 'no fixed version' }
            Write-Host "Reachable Go vulnerability: $key ($fixed)"
        }
        throw 'Reachable Go dependency vulnerability found'
    }
    Write-Host "govulncheck: no unaccepted reachable vulnerabilities ($($findings.Count) findings scanned)"
}

Push-Location (Join-Path $root 'sidecar')
try {
    $env:GOTOOLCHAIN = 'auto'
    Assert-NoReachableVulnerability
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
