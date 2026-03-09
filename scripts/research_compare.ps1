param(
    [string]$UpstreamRemote = "upstream",
    [string]$OriginRemote = "origin",
    [string]$BaseBranch = "main",
    [string]$ResearchBranch = "research/pressure-field"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-GitOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
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

$repoRoot = Get-GitOutput -Arguments @("rev-parse", "--show-toplevel")
Set-Location $repoRoot

$upstreamRepo = Get-GitHubRepoPath -RemoteName $UpstreamRemote
$originRepo = Get-GitHubRepoPath -RemoteName $OriginRemote
$compareUrl = "https://github.com/$upstreamRepo/compare/$BaseBranch...$($originRepo -replace '/', ':'):$ResearchBranch"

Write-Host $compareUrl
