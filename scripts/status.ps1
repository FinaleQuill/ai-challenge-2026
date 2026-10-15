param([string]$Config = 'configs/convnext_tiny.json')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$runPath = if ([System.IO.Path]::IsPathRooted($settings.run_dir)) { $settings.run_dir } else { Join-Path $projectRoot $settings.run_dir }
$pipelinePath = Join-Path $runPath 'pipeline_status.json'
if (Test-Path -LiteralPath $pipelinePath) {
    $pipelineState = Get-Content -LiteralPath $pipelinePath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Output "Pipeline: $($pipelineState.status), stage: $($pipelineState.stage)"
    if ($pipelineState.error) { Write-Output "Error: $($pipelineState.error)" }
}
$summaryPath = Join-Path $runPath 'summary.json'
if (Test-Path -LiteralPath $summaryPath) {
    $trainingState = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Output "Training: $($trainingState.status), completed epochs: $($trainingState.completed_epochs), best epoch: $($trainingState.best_epoch), best Study AUC: $($trainingState.best_score)"
}
$logPath = Join-Path $runPath 'train.log'
if (Test-Path -LiteralPath $logPath) { Get-Content -LiteralPath $logPath -Tail 2 -Encoding UTF8 }
& nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
