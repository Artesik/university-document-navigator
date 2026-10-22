param([string]$Python = 'python', [switch]$Cpu)
$ErrorActionPreference = 'Stop'
$appFolder = $PSScriptRoot
$environmentFolder = Join-Path (Split-Path $appFolder -Parent) '.venv'
$environmentPython = Join-Path $environmentFolder 'Scripts/python.exe'
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}
if (-not (Test-Path -LiteralPath $environmentPython)) {
    Invoke-Checked $Python @('-c', 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"')
    Invoke-Checked $Python @('-m', 'venv', $environmentFolder)
}
Invoke-Checked $environmentPython @('-c', 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"')
$wheelIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
$torchVersion = if ($Cpu) { 'torch==2.8.0+cpu' } else { 'torch==2.8.0+cu128' }
Invoke-Checked $environmentPython @('-m', 'pip', 'install', $torchVersion, '--index-url', $wheelIndex)
Invoke-Checked $environmentPython @('-m', 'pip', 'install', '-r', (Join-Path $appFolder 'requirements-dev.txt'), '-c', (Join-Path $appFolder 'constraints-windows-py312.txt'))
Invoke-Checked $environmentPython @('-X', 'utf8', (Join-Path $appFolder 'scripts/download_model.py'))
Invoke-Checked $environmentPython @('-X', 'utf8', (Join-Path $appFolder 'cli.py'), 'doctor')
Write-Host 'Installation complete. Run app/run.bat'
