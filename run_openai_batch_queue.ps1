param(
    [string]$WorkDir = ".\artifacts\openai_batch_strict_20k_small",
    [int]$PollSeconds = 120,
    [string]$PythonExe = "python",
    [int]$MaxConcurrentJobs = 1,
    [switch]$Finalize,
    [string]$Output = ".\djinni_ner_annotations_openai_20k.jsonl",
    [switch]$RequireEnglish,
    [switch]$RequireIt
)

$ErrorActionPreference = "Stop"

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Invoke-BatchPython {
    param([string[]]$Arguments)
    & $PythonExe $BatchScript @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: {1} {2} {3}" -f $LASTEXITCODE, $PythonExe, $BatchScript, ($Arguments -join ' '))
    }
}

function Get-TrackedJobs {
    param($JobsFile)
    if ($null -eq $JobsFile -or $null -eq $JobsFile.jobs) {
        return @()
    }
    return @($JobsFile.jobs)
}

function Get-PendingShardIds {
    param($Manifest, $JobsFile)
    $allShardIds = @($Manifest.shards | ForEach-Object { $_.shard_id })
    $submittedShardIds = @{}
    foreach ($job in (Get-TrackedJobs $JobsFile)) {
        $submittedShardIds[$job.shard_id] = $true
    }
    return @($allShardIds | Where-Object { -not $submittedShardIds.ContainsKey($_) })
}

function Get-ActiveJobs {
    param($JobsFile)
    $activeStates = @("validating", "in_progress", "finalizing", "cancelling")
    return @(Get-TrackedJobs $JobsFile | Where-Object { $_.batch.status -in $activeStates })
}

function Get-CompletedUndownloadedJobs {
    param($JobsFile)
    return @(Get-TrackedJobs $JobsFile | Where-Object {
        $_.batch.status -eq "completed" -and -not $_.PSObject.Properties.Name.Contains("local_output_path")
    })
}

function Get-FailedJobs {
    param($JobsFile)
    $failedStates = @("failed", "expired", "cancelled")
    return @(Get-TrackedJobs $JobsFile | Where-Object { $_.batch.status -in $failedStates })
}

function Show-JobSummary {
    param($Manifest, $JobsFile)
    $trackedJobs = @(Get-TrackedJobs $JobsFile)
    $activeJobs = @(Get-ActiveJobs $JobsFile)
    $pendingShardIds = @(Get-PendingShardIds $Manifest $JobsFile)
    $completedCount = @($trackedJobs | Where-Object { $_.batch.status -eq "completed" }).Count
    $failedCount = @(Get-FailedJobs $JobsFile).Count
    Write-Host ("Tracked shards: {0}/{1} | Active: {2} | Completed: {3} | Failed: {4} | Pending submit: {5}" -f `
        $trackedJobs.Count, @($Manifest.shards).Count, $activeJobs.Count, $completedCount, $failedCount, $pendingShardIds.Count)
}

$ResolvedWorkDir = [System.IO.Path]::GetFullPath((Join-Path $PWD $WorkDir))
$BatchScript = Join-Path $PSScriptRoot "djinni_openai_batch_ner.py"
$ManifestPath = Join-Path $ResolvedWorkDir "manifest.json"
$JobsPath = Join-Path $ResolvedWorkDir "jobs.json"

if (-not (Test-Path -LiteralPath $BatchScript)) {
    throw "Batch script not found: $BatchScript"
}
if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$Manifest = Read-JsonFile $ManifestPath
if ($null -eq $Manifest -or $null -eq $Manifest.shards) {
    throw "Invalid manifest: $ManifestPath"
}

while ($true) {
    $JobsFile = Read-JsonFile $JobsPath
    Show-JobSummary $Manifest $JobsFile

    $FailedJobs = @(Get-FailedJobs $JobsFile)
    if ($FailedJobs.Count -gt 0) {
        Write-Host ""
        Write-Host "Detected failed batch job(s):" -ForegroundColor Red
        foreach ($job in $FailedJobs) {
            Write-Host ("- {0}: {1}" -f $job.shard_id, $job.batch.status) -ForegroundColor Red
            if ($job.batch.errors -and $job.batch.errors.data) {
                foreach ($err in @($job.batch.errors.data)) {
                    Write-Host ("  code={0} message={1}" -f $err.code, $err.message) -ForegroundColor Red
                }
            }
        }
        throw "Stopping because at least one shard failed."
    }

    $CompletedUndownloadedJobs = @(Get-CompletedUndownloadedJobs $JobsFile)
    if ($CompletedUndownloadedJobs.Count -gt 0) {
        Write-Host "Downloading completed shard outputs..." -ForegroundColor Cyan
        Invoke-BatchPython @("download", "--work-dir", $ResolvedWorkDir)
        continue
    }

    $ActiveJobs = @(Get-ActiveJobs $JobsFile)
    if ($ActiveJobs.Count -ge $MaxConcurrentJobs) {
        Write-Host ("Waiting {0} seconds before polling status again..." -f $PollSeconds) -ForegroundColor Yellow
        Start-Sleep -Seconds $PollSeconds
        Invoke-BatchPython @("status", "--work-dir", $ResolvedWorkDir)
        continue
    }

    $PendingShardIds = @(Get-PendingShardIds $Manifest $JobsFile)
    if ($PendingShardIds.Count -gt 0) {
        $NextShard = $PendingShardIds[0]
        Write-Host ("Submitting next shard: {0}" -f $NextShard) -ForegroundColor Green
        Invoke-BatchPython @("submit", "--work-dir", $ResolvedWorkDir, "--shard-id", $NextShard, "--max-shards", "1")
        continue
    }

    Write-Host "All shards have been submitted, completed, and downloaded." -ForegroundColor Green

    if ($Finalize) {
        $FinalizeArgs = @("finalize", "--work-dir", $ResolvedWorkDir, "--output", $Output)
        if ($RequireEnglish) {
            $FinalizeArgs += "--require-english"
        }
        if ($RequireIt) {
            $FinalizeArgs += "--require-it"
        }
        Write-Host ("Finalizing annotations into {0}" -f $Output) -ForegroundColor Cyan
        Invoke-BatchPython $FinalizeArgs
    }
    break
}
