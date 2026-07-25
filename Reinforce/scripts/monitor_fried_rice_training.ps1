param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId,
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [Parameter(Mandatory = $true)]
    [string]$CheckpointDirectory,
    [int]$ExpectedEpisodes = 450,
    [int]$PollSeconds = 60,
    [int]$StallMinutes = 45
)

$ErrorActionPreference = "Stop"
$resolvedRunDirectory = [System.IO.Path]::GetFullPath($RunDirectory)
$resolvedCheckpointDirectory = [System.IO.Path]::GetFullPath($CheckpointDirectory)
$episodesPath = Join-Path $resolvedRunDirectory "episodes.csv"
$statusPath = Join-Path $resolvedRunDirectory "training_monitor_status.json"
$temporaryStatusPath = Join-Path $resolvedRunDirectory ".training_monitor_status.json.tmp"
$eventsPath = Join-Path $resolvedRunDirectory "training_monitor_events.log"
$lastEpisodeCount = -1
$lastProgressAt = Get-Date

while ($true) {
    $now = Get-Date
    $process = Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
    $episodeCount = 0
    $lastEpisodeId = $null
    if (Test-Path -LiteralPath $episodesPath) {
        $rows = @(Import-Csv -LiteralPath $episodesPath)
        $episodeCount = $rows.Count
        if ($episodeCount -gt 0) {
            $lastEpisodeId = [int]$rows[-1].episode_id
        }
    }
    if ($episodeCount -ne $lastEpisodeCount) {
        $lastEpisodeCount = $episodeCount
        $lastProgressAt = $now
        Add-Content -LiteralPath $eventsPath -Encoding utf8 -Value (
            "{0:o} episodes={1}/{2}" -f $now, $episodeCount, $ExpectedEpisodes
        )
    }

    $checkpointCount = 0
    if (Test-Path -LiteralPath $resolvedCheckpointDirectory) {
        $checkpointCount = @(
            Get-ChildItem -LiteralPath $resolvedCheckpointDirectory -Filter "*.zip" -File
        ).Count
    }
    $stalledMinutes = ($now - $lastProgressAt).TotalMinutes
    $state = if ($episodeCount -ge $ExpectedEpisodes) {
        "completed"
    }
    elseif ($null -eq $process) {
        "process_exited_early"
    }
    elseif ($stalledMinutes -ge $StallMinutes) {
        "stalled"
    }
    else {
        "running"
    }
    $status = [ordered]@{
        state = $state
        checked_at = $now.ToString("o")
        training_process_id = $TrainingProcessId
        process_alive = $null -ne $process
        episodes_completed = $episodeCount
        expected_episodes = $ExpectedEpisodes
        progress_fraction = $episodeCount / [double]$ExpectedEpisodes
        last_episode_id = $lastEpisodeId
        last_progress_at = $lastProgressAt.ToString("o")
        stalled_minutes = [math]::Round($stalledMinutes, 2)
        checkpoint_count = $checkpointCount
        run_directory = $resolvedRunDirectory
        checkpoint_directory = $resolvedCheckpointDirectory
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $temporaryStatusPath -Encoding utf8
    Move-Item -LiteralPath $temporaryStatusPath -Destination $statusPath -Force

    if ($state -in @("completed", "process_exited_early")) {
        Add-Content -LiteralPath $eventsPath -Encoding utf8 -Value (
            "{0:o} state={1} episodes={2}/{3}" -f $now, $state, $episodeCount, $ExpectedEpisodes
        )
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
