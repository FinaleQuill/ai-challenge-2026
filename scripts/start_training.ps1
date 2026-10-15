param(
    [string]$Config = 'configs/convnext_tiny.json',
    [switch]$Resume,
    [switch]$SkipPrepare
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$configPath = (Resolve-Path -LiteralPath $configPath).Path
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$runPath = if ([System.IO.Path]::IsPathRooted($settings.run_dir)) { $settings.run_dir } else { Join-Path $projectRoot $settings.run_dir }
$runPath = [System.IO.Path]::GetFullPath($runPath)
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$processPath = Join-Path $runPath 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previousRun = Get-Content -LiteralPath $processPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $previousProcess = Get-Process -Id $previousRun.pid -ErrorAction SilentlyContinue
    if ($previousProcess) {
        $expectedTicks = if ($previousRun.process_start_ticks) {
            [long]$previousRun.process_start_ticks
        } elseif ($previousRun.process_started_utc) {
            ([datetime]$previousRun.process_started_utc).ToUniversalTime().Ticks
        } else { 0 }
        if (($expectedTicks -eq 0) -or ($previousProcess.StartTime.ToUniversalTime().Ticks -eq $expectedTicks)) {
            throw "Training process $($previousRun.pid) is already running."
        }
    }
}
if (Test-Path -LiteralPath (Join-Path $runPath 'STOP')) {
    throw "Remove the STOP file in $runPath before starting or resuming."
}
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $runPath 'last.pt'))) {
    throw "No last.pt checkpoint to resume in $runPath."
}
$arguments = '-X utf8 -u -m cxr.pipeline --config "' + $configPath + '"'
if ($Resume) { $arguments += ' --resume' }
if ($SkipPrepare) { $arguments += ' --skip-prepare' }
$logSuffix = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $runPath "console_$logSuffix.log"
$stderrPath = Join-Path $runPath "console_$logSuffix.err.log"
$trainingProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$startedTicks = $trainingProcess.StartTime.ToUniversalTime().Ticks
Start-Sleep -Milliseconds 300
$trainingProcess.Refresh()
if ($trainingProcess.HasExited) {
    throw "Pipeline exited during startup. See $stderrPath and $stdoutPath."
}
@{ pid = $trainingProcess.Id; process_start_ticks = $startedTicks; config = $configPath; stdout = $stdoutPath; stderr = $stderrPath } | ConvertTo-Json | Set-Content -LiteralPath ($processPath + '.tmp') -Encoding UTF8
Move-Item -LiteralPath ($processPath + '.tmp') -Destination $processPath -Force
Write-Output "Training pipeline started. PID: $($trainingProcess.Id)"
Write-Output "Log: $stdoutPath"
Write-Output "Stop safely: create $(Join-Path $runPath 'STOP')"
