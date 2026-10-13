# =============================================================================
# setup.ps1  --  One-shot bootstrap for Federated Tamil Court ASR (Windows)
# Usage:  Right-click -> "Run with PowerShell"
#         OR in a terminal: powershell -ExecutionPolicy Bypass -File setup.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

function Info    { Write-Host "[INFO]  $args" -ForegroundColor Cyan }
function Success { Write-Host "[OK]    $args" -ForegroundColor Green }
function Warn    { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function Err     { Write-Host "[ERROR] $args" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=================================================================" -ForegroundColor White
Write-Host "  Federated Privacy-Preserving Tamil Court ASR" -ForegroundColor White
Write-Host "  One-Shot Setup Script  (Windows PowerShell)" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor White
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# -- 1. Python check ----------------------------------------------------------
Info "Checking Python version..."
$PythonBin = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -match "^3\.(\d+)$" -and [int]$matches[1] -ge 9) {
            $PythonBin = $cmd
            Success "Found Python $ver -> $cmd"
            break
        }
    } catch { }
}
if (-not $PythonBin) { Err "Python 3.9+ not found. Install from https://python.org and retry." }

# -- 2. .env setup ------------------------------------------------------------
if (Test-Path ".env") {
    Warn ".env already exists -- skipping (delete it to reset)."
} else {
    if (-not (Test-Path ".env.example")) { Err ".env.example not found. Run from repo root." }
    Copy-Item ".env.example" ".env"
    Success ".env created from .env.example"
    Write-Host ""
    Write-Host "  IMPORTANT: Open .env and fill in your HF_TOKEN and other secrets." -ForegroundColor Yellow
    Write-Host "  Simulation mode works without any token." -ForegroundColor Yellow
    Write-Host ""
}

# -- 3. Virtual environment ---------------------------------------------------
if (Test-Path ".venv") {
    Info "Virtual environment .venv already exists -- skipping creation."
} else {
    Info "Creating virtual environment (.venv)..."
    & $PythonBin -m venv .venv
    Success "Virtual environment created."
}

# Activate
$ActivateScript = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) { Err "Could not find $ActivateScript" }
. $ActivateScript
Success "Virtual environment activated."

# -- 4. Pip + requirements ----------------------------------------------------
Info "Upgrading pip..."
pip install --quiet --upgrade pip

# Auto-detect CUDA for PyTorch
$TorchInstalled = python -c "import torch; print(torch.__version__)" 2>$null
if (-not $TorchInstalled) {
    Info "Detecting GPU/CUDA for PyTorch installation..."
    try {
        $nvOut = nvidia-smi 2>$null | Out-String
        if ($nvOut -match "CUDA Version: 12") {
            Info "CUDA 12.x detected -- installing PyTorch cu121..."
            pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
        } elseif ($nvOut -match "CUDA Version: 11") {
            Info "CUDA 11.x detected -- installing PyTorch cu118..."
            pip install --quiet torch --index-url https://download.pytorch.org/whl/cu118
        } else {
            Warn "GPU found but CUDA version unclear -- installing CPU PyTorch."
            pip install --quiet torch
        }
    } catch {
        Warn "No NVIDIA GPU detected -- installing CPU-only PyTorch."
        pip install --quiet torch
    }
} else {
    Info "PyTorch already installed ($TorchInstalled) -- skipping."
}

Info "Installing project requirements..."
pip install --quiet -r requirements.txt
Success "All dependencies installed."

# -- 5. Verify core imports ---------------------------------------------------
Info "Verifying key imports..."
python -c "import fastapi, uvicorn, transformers, peft, librosa, safetensors"
if ($LASTEXITCODE -ne 0) { Err "Import check failed -- see output above." }
Success "All imports OK."

# -- 6. Install Frontend Dependencies ------------------------------------------
Info "Installing Next.js frontend dependencies..."
Push-Location frontend
npm install
Pop-Location
Success "Frontend dependencies ready."

# -- 7. Launch Instructions ---------------------------------------------------
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "  1. Backend API:  uvicorn backend.api:app --port 8080 --reload" -ForegroundColor Yellow
Write-Host "  2. Frontend UI:  cd frontend; npm run dev" -ForegroundColor Yellow
Write-Host "  3. Open:         http://localhost:3000 in your browser" -ForegroundColor Cyan
Write-Host ""
