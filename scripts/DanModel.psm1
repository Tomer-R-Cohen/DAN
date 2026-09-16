# Shared helpers for resolving and fetching DAN model configs, used by
# Select-DAN-Model.ps1 and Start-DAN-Service.ps1. Mirrors the URL-derivation
# logic in engine/coordinator.cpp's load_manifest() so a config with either
# an explicit artifact_url or an hf_repo/gguf_filename/artifact_revision
# triple resolves to the same download location the coordinator itself uses.

function Get-DanModelConfigs {
    param([Parameter(Mandatory)][string]$Directory)
    if (-not (Test-Path -LiteralPath $Directory)) { return @() }
    Get-ChildItem -LiteralPath $Directory -Filter '*.json' -File | ForEach-Object {
        $config = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        [PSCustomObject]@{ File = $_.Name; Path = $_.FullName; Config = $config }
    }
}

function Get-DanModelDownloadUrl {
    param([Parameter(Mandatory)]$Config)
    if ($Config.PSObject.Properties.Name -contains 'artifact_url' -and $Config.artifact_url) {
        return $Config.artifact_url
    }
    if (-not $Config.hf_repo -or -not $Config.gguf_filename -or -not $Config.artifact_revision) {
        throw "model config '$($Config.model_id)' has neither artifact_url nor hf_repo/gguf_filename/artifact_revision"
    }
    return "https://huggingface.co/$($Config.hf_repo)/resolve/$($Config.artifact_revision)/$($Config.gguf_filename)"
}

function Get-DanModelLocalFileName {
    param([Parameter(Mandatory)]$Config)
    if ($Config.PSObject.Properties.Name -contains 'gguf_filename' -and $Config.gguf_filename) {
        return $Config.gguf_filename
    }
    return [System.IO.Path]::GetFileName((Get-DanModelDownloadUrl -Config $Config))
}

# Downloads (or reuses a checksum-verified cached copy of) a model's GGUF into
# CacheDirectory, for use as a coordinator-local draft model. Target models are
# NOT downloaded this way -- providers range-fetch only their own layers.
function Install-DanDraftModel {
    param(
        [Parameter(Mandatory)]$Config,
        [Parameter(Mandatory)][string]$CacheDirectory
    )
    New-Item -ItemType Directory -Force -Path $CacheDirectory | Out-Null
    $path = Join-Path $CacheDirectory (Get-DanModelLocalFileName -Config $Config)
    $expectedHash = $Config.artifact_sha256.ToLowerInvariant()
    $needsDownload = $true
    if (Test-Path -LiteralPath $path) {
        $existing = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        $needsDownload = $existing -ne $expectedHash
    }
    if ($needsDownload) {
        $url = Get-DanModelDownloadUrl -Config $Config
        Write-Host "Fetching $($Config.model_id) from $url ..."
        Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
        $downloadedHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($downloadedHash -ne $expectedHash) {
            Remove-Item -LiteralPath $path -Force
            throw "checksum mismatch for $($Config.model_id)"
        }
    }
    return $path
}

Export-ModuleMember -Function Get-DanModelConfigs, Get-DanModelDownloadUrl, `
    Get-DanModelLocalFileName, Install-DanDraftModel
