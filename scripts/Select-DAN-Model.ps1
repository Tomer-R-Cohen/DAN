[CmdletBinding()]
param(
    # Skip the prompts and just (re)apply the last saved selection, downloading
    # the draft model if needed. Used by Start-DAN-Service.ps1's -NoPicker path.
    [switch]$Auto
)

$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
Import-Module (Join-Path $package 'DanModel.psm1') -Force

$modelsDirectory = Join-Path $package 'config\models'
$activeManifest = Join-Path $package 'config\active-model.json'
$selectionPath = Join-Path $package 'config\selection.json'
$resultPath = Join-Path $package 'data\selected-run.json'

$models = @(Get-DanModelConfigs -Directory $modelsDirectory)
if ($models.Count -eq 0) { throw "No model configs found under $modelsDirectory" }

$selection = if (Test-Path -LiteralPath $selectionPath) {
    Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
} else {
    [PSCustomObject]@{ target = $models[0].File; draft = 'none' }
}
if (-not ($models | Where-Object File -eq $selection.target)) { $selection.target = $models[0].File }

function Show-ModelTable {
    param([array]$Models, [string]$CurrentFile, [switch]$AllowNone)
    Write-Host ''
    if ($AllowNone) {
        $marker = if ($CurrentFile -eq 'none') { '*' } else { ' ' }
        Write-Host ("  {0} 0) none" -f $marker) -ForegroundColor (if ($marker -eq '*') { 'Green' } else { 'DarkGray' })
    }
    for ($i = 0; $i -lt $Models.Count; $i++) {
        $m = $Models[$i].Config
        $marker = if ($Models[$i].File -eq $CurrentFile) { '*' } else { ' ' }
        $size = if ($m.PSObject.Properties.Name -contains 'artifact_bytes' -and $m.artifact_bytes) {
            "{0:N1} GB" -f ($m.artifact_bytes / 1GB)
        } else { '' }
        $context = if ($m.context_size) { $m.context_size } else { '-' }
        Write-Host ("  {0}{1,2}) {2,-34} ctx={3,-6} {4}" -f $marker, ($i + 1), $m.model_id, $context, $size) `
            -ForegroundColor (if ($marker -eq '*') { 'Green' } else { 'Gray' })
    }
}

if (-not $Auto) {
    Write-Host ''
    Write-Host '=== DAN model selection ===' -ForegroundColor Cyan
    Write-Host 'Target model, split across your providers and served over the WAN:' -ForegroundColor Cyan
    Show-ModelTable -Models $models -CurrentFile $selection.target
    $targetInput = Read-Host 'Enter a number, or press Enter to keep the current (*) one'
    if ($targetInput) {
        $index = 0
        if (-not [int]::TryParse($targetInput, [ref]$index) -or $index -lt 1 -or $index -gt $models.Count) {
            throw "invalid selection: $targetInput"
        }
        $selection.target = $models[$index - 1].File
    }

    $targetConfig = ($models | Where-Object File -eq $selection.target).Config
    $draftCandidates = @($models | Where-Object {
        $_.File -ne $selection.target -and
        (-not $_.Config.architecture -or -not $targetConfig.architecture -or
            $_.Config.architecture -eq $targetConfig.architecture)
    })
    Write-Host ''
    Write-Host 'Draft model for speculative decoding (small, loaded locally; optional):' -ForegroundColor Cyan
    if ($draftCandidates.Count -eq 0) {
        Write-Host '  (no other bundled model is available as a draft)' -ForegroundColor DarkGray
        $selection.draft = 'none'
    } else {
        Show-ModelTable -Models $draftCandidates -CurrentFile $selection.draft -AllowNone
        $draftInput = Read-Host 'Enter a number, 0 for none, or press Enter to keep the current (*) one'
        if ($draftInput -eq '0') {
            $selection.draft = 'none'
        } elseif ($draftInput) {
            $index = 0
            if (-not [int]::TryParse($draftInput, [ref]$index) -or $index -lt 1 -or $index -gt $draftCandidates.Count) {
                throw "invalid selection: $draftInput"
            }
            $selection.draft = $draftCandidates[$index - 1].File
        }
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $selectionPath) | Out-Null
    $selection | ConvertTo-Json | Set-Content -LiteralPath $selectionPath -Encoding utf8
}

$targetEntry = $models | Where-Object File -eq $selection.target
Copy-Item -LiteralPath $targetEntry.Path -Destination $activeManifest -Force
Write-Host ''
Write-Host "Target model: $($targetEntry.Config.model_id)" -ForegroundColor Green

$draftModelPath = ''
$draftModelId = ''
if ($selection.draft -and $selection.draft -ne 'none') {
    $draftEntry = $models | Where-Object File -eq $selection.draft
    if ($draftEntry) {
        $draftCache = Join-Path $package 'data\draft-model'
        try {
            $draftModelPath = Install-DanDraftModel -Config $draftEntry.Config -CacheDirectory $draftCache
            $draftModelId = $draftEntry.Config.model_id
            Write-Host "Draft model: $draftModelId" -ForegroundColor Green
        } catch {
            Write-Host "Speculative decoding disabled: could not prepare the draft model ($($_.Exception.Message))" `
                -ForegroundColor DarkYellow
        }
    }
} else {
    Write-Host 'Draft model: none (speculative decoding off)' -ForegroundColor DarkGray
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resultPath) | Out-Null
[PSCustomObject]@{
    TargetModelId  = $targetEntry.Config.model_id
    DraftModelId   = $draftModelId
    DraftModelPath = $draftModelPath
} | ConvertTo-Json | Set-Content -LiteralPath $resultPath -Encoding utf8
