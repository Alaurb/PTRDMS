$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Python 3.12 virtual environment is missing."
    Write-Host "Run: py -3.12 -m venv .venv"
    Write-Host "Then: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    Write-Host "And: .\.venv\Scripts\python.exe -m pip install -r optional-requirements.txt"
    exit 1
}

& $python (Join-Path $repoRoot "app.py")
