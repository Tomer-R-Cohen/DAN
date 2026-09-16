# One-line installer for a friend contributing a GPU to DAN.
#   irm https://raw.githubusercontent.com/Tomer-R-Cohen/DAN/main/scripts/Install-DAN-Provider.ps1 | iex
#
# Downloads the latest DAN Provider release from GitHub Releases, verifies its
# SHA-256 checksum, extracts it, and launches dan-provider.exe. The provider's
# own first-run setup then asks only for the coordinator address and takes it
# from there (private-network onboarding, GPU detection, live dashboard).

[CmdletBinding()]
param(
    [string]$Repo = 'Tomer-R-Cohen/DAN',
    [string]$InstallDirectory = (Join-Path $env:LOCALAPPDATA 'DAN\app'),
    [switch]$NoLaunch
)

$ErrorActionPreference = 'Stop'

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'DAN Provider requires 64-bit Windows.'
}

Write-Host 'DAN Provider installer'
Write-Host 'Looking up the latest release...'
$release = Invoke-RestMethod -UseBasicParsing `
    -Uri "https://api.github.com/repos/$Repo/releases/latest" `
    -Headers @{ 'User-Agent' = 'DAN-Installer' }

$asset = $release.assets | Where-Object { $_.name -match '^DAN-Provider-.*-Windows-x64\.zip$' } |
    Select-Object -First 1
if (-not $asset) { throw "No Windows provider asset found on release $($release.tag_name)" }
$checksumAsset = $release.assets | Where-Object { $_.name -eq "$($asset.name).sha256" } |
    Select-Object -First 1
if (-not $checksumAsset) { throw "No checksum file found for $($asset.name)" }

$downloadDirectory = Join-Path $env:TEMP 'dan-provider-install'
New-Item -ItemType Directory -Force -Path $downloadDirectory | Out-Null
$zipPath = Join-Path $downloadDirectory $asset.name
$checksumPath = Join-Path $downloadDirectory $checksumAsset.name

Write-Host "Downloading $($asset.name) ($([Math]::Round($asset.size / 1MB, 1)) MB)..."
Invoke-WebRequest -UseBasicParsing -Uri $asset.browser_download_url -OutFile $zipPath
Invoke-WebRequest -UseBasicParsing -Uri $checksumAsset.browser_download_url -OutFile $checksumPath

Write-Host 'Verifying checksum...'
$expected = ((Get-Content -LiteralPath $checksumPath -Raw) -split '\s+')[0].ToLowerInvariant()
$actual = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($expected -ne $actual) {
    Remove-Item -LiteralPath $zipPath, $checksumPath -Force
    throw "Checksum mismatch for $($asset.name); download was corrupted or tampered with. Aborting."
}

if (Test-Path -LiteralPath $InstallDirectory) {
    Write-Host "Removing the previous install at $InstallDirectory..."
    Remove-Item -LiteralPath $InstallDirectory -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $InstallDirectory | Out-Null
Write-Host "Extracting to $InstallDirectory..."
Expand-Archive -LiteralPath $zipPath -DestinationPath $InstallDirectory -Force
Remove-Item -LiteralPath $zipPath, $checksumPath -Force

$extracted = Get-ChildItem -LiteralPath $InstallDirectory -Directory | Select-Object -First 1
$providerExe = if ($extracted) { Join-Path $extracted.FullName 'dan-provider.exe' }
if (-not $providerExe -or -not (Test-Path -LiteralPath $providerExe -PathType Leaf)) {
    throw "dan-provider.exe was not found after extraction ($InstallDirectory)"
}

$shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'DAN Provider.lnk'
try {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $providerExe
    $shortcut.WorkingDirectory = Split-Path -Parent $providerExe
    $shortcut.Save()
    Write-Host "Desktop shortcut created: DAN Provider"
} catch {
    Write-Host 'Could not create a desktop shortcut (not fatal).' -ForegroundColor DarkYellow
}

Write-Host "`nInstalled DAN Provider $($release.tag_name)."
if (-not $NoLaunch) {
    Write-Host "Starting DAN Provider. Have the coordinator's DAN address ready.`n"
    & $providerExe
} else {
    Write-Host "Run `"$providerExe`" (or the desktop shortcut) to start contributing."
}
