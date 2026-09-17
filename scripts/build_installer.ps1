# Builds DAN-Setup.exe: one installer with the GPU node, chat, runtime and shortcuts.
#
#   .\scripts\build_installer.ps1                       # local test: network node on the installing PC
#   .\scripts\build_installer.ps1 -Bootstrap /ip4/<VPS_IP>/tcp/4001/p2p/<VPS_PEERID>
#
# Needs a CUDA build (default build-cuda), the Windows sidecar, the gateway and Inno Setup 6.
[CmdletBinding()]
param(
    [string[]]$Bootstrap,
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '..\build-cuda'),
    [string]$Sidecar = (Join-Path $PSScriptRoot '..\build-client\sidecar\dan-sidecar.exe'),
    [string]$Gateway = (Join-Path $PSScriptRoot '..\build\gateway\dan-api-gateway-windows-amd64.exe'),
    [string]$Version = '1.1.0',
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\build\installer'),
    [string]$InnoCompiler
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$local = -not $Bootstrap

if (-not $InnoCompiler) {
    $InnoCompiler = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $InnoCompiler) { throw 'Inno Setup 6 not found. Install it: winget install JRSoftware.InnoSetup' }

# Runtime DLLs: llama.cpp/ggml from the build, CUDA from the toolkit it used, MSVC runtime.
$cache = Get-Content -LiteralPath (Join-Path $build 'CMakeCache.txt')
$cudaBin = ($cache | Select-String '^CUDAToolkit_BIN_DIR:PATH=(.+)$').Matches[0].Groups[1].Value
$cudaDlls = @('cublas64_*.dll', 'cublasLt64_*.dll', 'cudart64_*.dll') | ForEach-Object {
    Get-ChildItem -Path $cudaBin, (Join-Path $cudaBin 'x64') -Filter $_ -ErrorAction SilentlyContinue |
        Select-Object -First 1
}
$vsRoot = & (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe') `
    -latest -prerelease -products * -property installationPath
$vcRuntime = Get-ChildItem -Path (Join-Path $vsRoot 'VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT') -Directory |
    Sort-Object FullName | Select-Object -Last 1
if (-not $vcRuntime) { throw "No MSVC runtime found under $vsRoot" }
$runtimeDlls = @(
    @('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll') |
        ForEach-Object { Join-Path $build "bin\Release\$_" }
) + @($cudaDlls.FullName) + @((Get-ChildItem -Path $vcRuntime.FullName -Filter '*.dll').FullName)

$stage = Join-Path $root 'build\installer-stage'
$packageArguments = @{
    BuildDirectory = $build
    Sidecar = (Resolve-Path -LiteralPath $Sidecar).Path
    Gateway = (Resolve-Path -LiteralPath $Gateway).Path
    RuntimeDll = $runtimeDlls
    StageDirectory = $stage
    NoZip = $true
}
if ($local) { $packageArguments.LocalNetwork = $true } else { $packageArguments.Bootstrap = $Bootstrap }
& (Join-Path $PSScriptRoot 'package_provider.ps1') @packageArguments

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDirectory).Path
$setupName = if ($local) { "DAN-Setup-$Version-local-test" } else { "DAN-Setup-$Version" }
& $InnoCompiler /Qp "/DSourceDir=$stage" "/DOutputDir=$output" "/DAppVersion=$Version" `
    "/DSetupName=$setupName" (Join-Path $root 'installer\dan.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed ($LASTEXITCODE)" }
$setup = Join-Path $output "$setupName.exe"
$hash = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$setup.sha256", "$hash  $setupName.exe`n", [Text.UTF8Encoding]::new($false))
Write-Host ''
Write-Host "Installer: $setup"
if ($local) { Write-Host 'Local test build: it joins a network node on the installing PC only.' }
