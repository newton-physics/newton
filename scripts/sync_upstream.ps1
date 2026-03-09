param(
    [string]$UpstreamRemote = "upstream",
    [string]$StableBranch = "main",
    [string]$ResearchBranch = "research/pressure-field",
    [ValidateSet("rebase", "merge")]
    [string]$Strategy = "rebase",
    [switch]$CreateResearchBranch,
    [switch]$AllowDirty,
    [switch]$ReturnToOriginalBranch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}

function Test-GitRef {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Ref
    )

    & git show-ref --verify --quiet $Ref
    return $LASTEXITCODE -eq 0
}

$repoRoot = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Run this script from inside a git working tree."
}

Set-Location $repoRoot

$status = (& git status --porcelain)
if (($LASTEXITCODE -ne 0) -or (-not $AllowDirty -and $status)) {
    throw "Working tree is not clean. Commit or stash changes first, or rerun with -AllowDirty if you understand the risk."
}

$originalBranch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine the current branch."
}

$remoteTrackingRef = "refs/remotes/$UpstreamRemote/$StableBranch"
$stableBranchRef = "refs/heads/$StableBranch"
$researchBranchRef = "refs/heads/$ResearchBranch"

Write-Host "Fetching $UpstreamRemote/$StableBranch..."
try {
    Invoke-Git -Arguments @("fetch", $UpstreamRemote, "--prune")

    if (-not (Test-GitRef -Ref $remoteTrackingRef)) {
        throw "Missing $UpstreamRemote/$StableBranch. Check the remote and branch names."
    }

    Write-Host "Updating $StableBranch with a fast-forward merge..."
    if (Test-GitRef -Ref $stableBranchRef) {
        Invoke-Git -Arguments @("switch", $StableBranch)
    } else {
        Invoke-Git -Arguments @("switch", "-c", $StableBranch, "$UpstreamRemote/$StableBranch")
    }
    Invoke-Git -Arguments @("merge", "--ff-only", "$UpstreamRemote/$StableBranch")

    if (Test-GitRef -Ref $researchBranchRef) {
        Write-Host "Switching to $ResearchBranch..."
        Invoke-Git -Arguments @("switch", $ResearchBranch)
    } elseif ($CreateResearchBranch) {
        Write-Host "Creating $ResearchBranch from $StableBranch..."
        Invoke-Git -Arguments @("switch", "-c", $ResearchBranch, $StableBranch)
    } else {
        throw "Research branch $ResearchBranch does not exist. Rerun with -CreateResearchBranch to create it from $StableBranch."
    }

    if ($Strategy -eq "rebase") {
        Write-Host "Rebasing $ResearchBranch onto $StableBranch..."
        Invoke-Git -Arguments @("rebase", $StableBranch)
    } else {
        Write-Host "Merging $StableBranch into $ResearchBranch..."
        Invoke-Git -Arguments @("merge", "--no-edit", $StableBranch)
    }

    Write-Host ""
    Write-Host "Sync complete."
    Write-Host "Stable branch:   $StableBranch -> $UpstreamRemote/$StableBranch"
    Write-Host "Research branch: $ResearchBranch ($Strategy)"
    Write-Host "Started on:      $originalBranch"
}
finally {
    if ($ReturnToOriginalBranch -and $originalBranch) {
        Write-Host "Returning to $originalBranch..."
        & git switch $originalBranch | Out-Null
    }
}
