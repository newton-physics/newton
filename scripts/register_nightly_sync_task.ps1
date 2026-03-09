param(
    [string]$TaskName = "Newton Nightly Research Sync",
    [string]$Time = "02:00"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Run this script from inside the Newton repository."
}

$nightlyScript = Join-Path $repoRoot "scripts\nightly_sync_report.ps1"
if (-not (Test-Path $nightlyScript)) {
    throw "Missing nightly sync script: $nightlyScript"
}

$triggerTime = [datetime]::ParseExact($Time, "HH:mm", $null)
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$nightlyScript`""
$trigger = New-ScheduledTaskTrigger -Daily -At $triggerTime
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Nightly upstream sync and report for the Newton pressure-field research branch." -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName' at $Time."
