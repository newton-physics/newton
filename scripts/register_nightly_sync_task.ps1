param(
    [Alias("TaskName")]
    [string]$MergeTaskName = "Newton Nightly Research Sync",
    [string]$RepairTaskName = "Newton Nightly Research Repair",
    [Alias("Time")]
    [string]$MergeTime = "06:00",
    [string[]]$RepairTimes = @("06:15", "07:15", "08:15", "09:15", "10:15", "11:15"),
    [switch]$NoRepair,
    [switch]$DirectScripts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Convert-ToBashSingleQuoted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    return "'" + ($Text -replace "'", "'\''") + "'"
}

function Convert-ToPowerShellSingleQuoted {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    return "'" + ($Text -replace "'", "''") + "'"
}

function Convert-FromWslUncPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $providerPrefix = "Microsoft.PowerShell.Core\FileSystem::"
    if ($Path.StartsWith($providerPrefix)) {
        $Path = $Path.Substring($providerPrefix.Length)
    }

    if ($Path -match "^\\\\(?:wsl\.localhost|wsl[$])\\[^\\]+\\(?<LinuxPath>.*)$") {
        return Normalize-PosixPath -Path ("/" + ($Matches["LinuxPath"] -replace "\\", "/"))
    }

    return $Path
}

function Normalize-PosixPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($part in ($Path -split "/")) {
        if ([string]::IsNullOrWhiteSpace($part) -or $part -eq ".") {
            continue
        }
        if ($part -eq "..") {
            if ($parts.Count -gt 0) {
                $parts.RemoveAt($parts.Count - 1)
            }
            continue
        }
        $parts.Add($part)
    }

    return "/" + ($parts -join "/")
}

function Join-TaskPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    if ($Root.StartsWith("/")) {
        return $Root.TrimEnd("/") + "/" + ($RelativePath -replace "\\", "/").TrimStart("/")
    }

    return Join-Path $Root $RelativePath
}

function Get-TaskPathParent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ($Path.StartsWith("/")) {
        $lastSlash = $Path.LastIndexOf("/")
        if ($lastSlash -le 0) {
            return "/"
        }
        return $Path.Substring(0, $lastSlash)
    }

    return Split-Path -Parent $Path
}

function New-NightlyAction {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    if ($RepoRoot.StartsWith("/")) {
        $quotedRepo = Convert-ToBashSingleQuoted -Text $RepoRoot
        $quotedLog = Convert-ToBashSingleQuoted -Text $LogPath
        $quotedLogDir = Convert-ToBashSingleQuoted -Text (Get-TaskPathParent -Path $LogPath)
        $bashCommand = "set -euo pipefail; mkdir -p $quotedLogDir; cd $quotedRepo; date '+%Y-%m-%d %H:%M:%S %Z'; $Command >> $quotedLog 2>&1"
        $escapedBashCommand = $bashCommand -replace "\\", "\\" -replace '"', '\"'
        return New-ScheduledTaskAction -Execute "wsl.exe" -Argument "bash -lc `"$escapedBashCommand`""
    }

    $quotedRepo = Convert-ToPowerShellSingleQuoted -Text $RepoRoot
    $quotedLog = Convert-ToPowerShellSingleQuoted -Text $LogPath
    $quotedLogDir = Convert-ToPowerShellSingleQuoted -Text (Split-Path -Parent $LogPath)
    $psCommand = "`$ErrorActionPreference = 'Stop'; New-Item -ItemType Directory -Path $quotedLogDir -Force | Out-Null; Set-Location $quotedRepo; Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz' | Add-Content -Path $quotedLog; $Command *>> $quotedLog"
    $escapedPsCommand = $psCommand -replace '"', '\"'
    return New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$escapedPsCommand`""
}

function New-DailyTrigger {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Time
    )

    $triggerTime = [datetime]::ParseExact($Time, "HH:mm", $null)
    return New-ScheduledTaskTrigger -Daily -At $triggerTime
}

$repoRootForWindows = $null
$repoRootForTask = $null
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $gitRepoRoot = (& git rev-parse --show-toplevel 2>$null)
    $gitExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($gitExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($gitRepoRoot)) {
    $repoRootForWindows = ($gitRepoRoot | Out-String).Trim()
    $repoRootForTask = Convert-FromWslUncPath -Path $repoRootForWindows
} elseif ($PSScriptRoot) {
    $repoRootForWindows = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $repoRootForTask = Convert-FromWslUncPath -Path $repoRootForWindows
} else {
    throw "Run this script from inside the Newton repository."
}

$mergeScript = Join-Path $repoRootForWindows "scripts/nightly_upstream_merge.py"
$repairScript = Join-Path $repoRootForWindows "scripts/nightly_upstream_repair.py"
$codexAutomationScript = Join-Path $repoRootForWindows "scripts/run_codex_automation.py"
if (-not (Test-Path $mergeScript)) {
    throw "Missing nightly merge script: $mergeScript"
}
if (-not (Test-Path $repairScript)) {
    throw "Missing nightly repair script: $repairScript"
}
if (-not (Test-Path $codexAutomationScript)) {
    throw "Missing Codex automation runner: $codexAutomationScript"
}

$mergeLog = Join-TaskPath -Root $repoRootForTask -RelativePath ".codex/automations/nightly-upstream-merge/task.log"
$repairLog = Join-TaskPath -Root $repoRootForTask -RelativePath ".codex/automations/nightly-upstream-repair/task.log"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

if ($DirectScripts) {
    $mergeCommand = "uv run --script scripts/nightly_upstream_merge.py --push"
    $repairCommand = "uv run --script scripts/nightly_upstream_repair.py prepare && uv run --script scripts/nightly_upstream_repair.py finalize --success --push"
} else {
    $mergeCommand = "python3 scripts/run_codex_automation.py .codex/automations/nightly-upstream-merge/automation.toml"
    $repairCommand = "python3 scripts/run_codex_automation.py .codex/automations/nightly-upstream-repair/automation.toml"
}

$mergeAction = New-NightlyAction -RepoRoot $repoRootForTask -Command $mergeCommand -LogPath $mergeLog
$mergeTrigger = New-DailyTrigger -Time $MergeTime
Register-ScheduledTask `
    -TaskName $MergeTaskName `
    -Action $mergeAction `
    -Trigger $mergeTrigger `
    -Settings $settings `
    -Description "Nightly upstream merge for Newton research branches. Writes reports under .codex/automations/nightly-upstream-merge." `
    -Force | Out-Null

Write-Host "Registered scheduled task '$MergeTaskName' at $MergeTime."

if (-not $NoRepair) {
    $repairAction = New-NightlyAction -RepoRoot $repoRootForTask -Command $repairCommand -LogPath $repairLog
    $repairTriggers = foreach ($repairTime in $RepairTimes) {
        New-DailyTrigger -Time $repairTime
    }
    Register-ScheduledTask `
        -TaskName $RepairTaskName `
        -Action $repairAction `
        -Trigger $repairTriggers `
        -Settings $settings `
        -Description "Bounded nightly repair for queued Newton upstream merge failures. Writes logs under .codex/automations/nightly-upstream-repair." `
        -Force | Out-Null

    Write-Host "Registered scheduled task '$RepairTaskName' at $($RepairTimes -join ', ')."
}
