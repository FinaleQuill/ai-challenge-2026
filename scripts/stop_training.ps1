param([string]$Config = 'configs/convnext_tiny.json')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$runPath = if ([System.IO.Path]::IsPathRooted($settings.run_dir)) { $settings.run_dir } else { Join-Path $projectRoot $settings.run_dir }
New-Item -ItemType Directory -Path $runPath -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $runPath 'STOP') -Force | Out-Null
Write-Output 'Stop requested. Training saves at the next optimizer boundary; other stages are interrupted.'
Write-Output "Before gaming, wait for $(Join-Path $runPath 'pipeline_status.json') status=stopped (or failed) and confirm the pipeline process has exited."
