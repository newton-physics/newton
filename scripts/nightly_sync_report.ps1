param(
    [string]$UpstreamRemote = "upstream",
    [string]$OriginRemote = "origin",
    [string]$StableBranch = "main",
    [string]$ResearchBranch = "research/pressure-field",
    [ValidateSet("rebase", "merge")]
    [string]$Strategy = "rebase"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-GitOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowFailure
    )

    $output = & git @Arguments 2>&1
    if (($LASTEXITCODE -ne 0) -and (-not $AllowFailure)) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE.`n$output"
    }
    return ($output | Out-String).Trim()
}

function Get-GitHubRepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteName
    )

    $remoteUrl = Get-GitOutput -Arguments @("remote", "get-url", $RemoteName)

    if ($remoteUrl -match "^https://github\.com/(?<path>.+?)(?:\.git)?$") {
        return $Matches["path"]
    }

    if ($remoteUrl -match "^git@github\.com:(?<path>.+?)(?:\.git)?$") {
        return $Matches["path"]
    }

    throw "Remote $RemoteName is not a GitHub remote: $remoteUrl"
}

function Test-GitRef {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Ref
    )

    & git show-ref --verify --quiet $Ref
    return $LASTEXITCODE -eq 0
}

$repoRoot = Get-GitOutput -Arguments @("rev-parse", "--show-toplevel")
Set-Location $repoRoot

$reportDir = Join-Path $repoRoot "reports\upstream-sync"
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

$timestamp = Get-Date
$reportStamp = $timestamp.ToString("yyyy-MM-dd_HHmmss")
$reportPath = Join-Path $reportDir "$reportStamp.md"
$currentBranch = Get-GitOutput -Arguments @("branch", "--show-current")
$worktreeStatus = Get-GitOutput -Arguments @("status", "--short", "--branch")
$dirtyStatus = Get-GitOutput -Arguments @("status", "--porcelain")
$isDirty = -not [string]::IsNullOrWhiteSpace($dirtyStatus)

$syncResult = "not-run"
$syncOutput = ""
$syncError = ""

if ($isDirty) {
    $syncResult = "skipped"
    $syncError = "Working tree is dirty. Nightly sync skipped to avoid rebasing or switching branches over uncommitted work."
} else {
    try {
        $syncScript = Join-Path $repoRoot "scripts\sync_upstream.ps1"
        $syncOutput = (& $syncScript -UpstreamRemote $UpstreamRemote -StableBranch $StableBranch -ResearchBranch $ResearchBranch -Strategy $Strategy -ReturnToOriginalBranch 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw $syncOutput
        }
        $syncResult = "success"
    } catch {
        $syncResult = "failed"
        $syncError = $_.Exception.Message
    }
}

$compareUrl = $null
try {
    $upstreamRepo = Get-GitHubRepoPath -RemoteName $UpstreamRemote
    $originRepo = Get-GitHubRepoPath -RemoteName $OriginRemote
    $compareUrl = "https://github.com/$upstreamRepo/compare/$StableBranch...$($originRepo -replace '/', ':'):$ResearchBranch"
} catch {
    $compareUrl = "Unavailable: $($_.Exception.Message)"
}

$refsToDescribe = @(
    @{ Label = "Upstream stable"; Ref = "refs/remotes/$UpstreamRemote/$StableBranch" },
    @{ Label = "Local stable"; Ref = "refs/heads/$StableBranch" },
    @{ Label = "Research"; Ref = "refs/heads/$ResearchBranch" }
)

$refSummaries = foreach ($refInfo in $refsToDescribe) {
    if (Test-GitRef -Ref $refInfo.Ref) {
        $sha = Get-GitOutput -Arguments @("rev-parse", "--short", $refInfo.Ref)
        $subject = Get-GitOutput -Arguments @("log", "-1", "--pretty=format:%s", $refInfo.Ref)
        "- $($refInfo.Label): `$sha` $subject"
    } else {
        "- $($refInfo.Label): missing"
    }
}

$divergence = "Unavailable"
if ((Test-GitRef -Ref "refs/remotes/$UpstreamRemote/$StableBranch") -and (Test-GitRef -Ref "refs/heads/$ResearchBranch")) {
    $counts = Get-GitOutput -Arguments @("rev-list", "--left-right", "--count", "$UpstreamRemote/$StableBranch...$ResearchBranch")
    $parts = $counts -split "\s+"
    if ($parts.Length -ge 2) {
        $divergence = "Behind upstream by $($parts[0]) commit(s), ahead by $($parts[1]) commit(s)."
    }
}

$recentUpstream = Get-GitOutput -Arguments @("log", "--oneline", "--decorate", "-5", "$UpstreamRemote/$StableBranch") -AllowFailure
$recentResearch = Get-GitOutput -Arguments @("log", "--oneline", "--decorate", "-5", $ResearchBranch) -AllowFailure

$reportLines = New-Object System.Collections.Generic.List[string]
$reportLines.Add("# Newton Nightly Sync Report")
$reportLines.Add("")
$reportLines.Add("- Timestamp: $($timestamp.ToString('yyyy-MM-dd HH:mm:ss zzz'))")
$reportLines.Add("- Repository: $repoRoot")
$reportLines.Add("- Started on branch: $currentBranch")
$reportLines.Add("- Sync result: $syncResult")
$reportLines.Add("- Strategy: $Strategy")
$reportLines.Add("- Compare URL: $compareUrl")
$reportLines.Add("")
$reportLines.Add("## Worktree")
$reportLines.Add("")
$reportLines.Add('```text')
$reportLines.Add($worktreeStatus)
$reportLines.Add('```')
$reportLines.Add("")
$reportLines.Add("## Branch Summary")
$reportLines.Add("")

foreach ($line in $refSummaries) {
    $reportLines.Add($line)
}

$reportLines.Add("")
$reportLines.Add("- Divergence: $divergence")
$reportLines.Add("")
$reportLines.Add("## Recent Upstream Commits")
$reportLines.Add("")
$reportLines.Add('```text')
$reportLines.Add($recentUpstream)
$reportLines.Add('```')
$reportLines.Add("")
$reportLines.Add("## Recent Research Commits")
$reportLines.Add("")
$reportLines.Add('```text')
$reportLines.Add($recentResearch)
$reportLines.Add('```')

if ($syncOutput) {
    $reportLines.Add("")
    $reportLines.Add("## Sync Output")
    $reportLines.Add("")
    $reportLines.Add('```text')
    $reportLines.Add($syncOutput)
    $reportLines.Add('```')
}

if ($syncError) {
    $reportLines.Add("")
    $reportLines.Add("## Sync Error")
    $reportLines.Add("")
    $reportLines.Add('```text')
    $reportLines.Add($syncError)
    $reportLines.Add('```')
}

Set-Content -Path $reportPath -Value $reportLines
Write-Host $reportPath
