#!/usr/bin/env bash
# =============================================================================
# setup.sh  --  One-shot bootstrap for Federated Tamil Court ASR
# Usage:  bash setup.sh          (or: chmod +x setup.sh && ./setup.sh)
# Works on: Linux, macOS, Git Bash / WSL on Windows
# =============================================================================
set -euo pipefail

# -- Colours ------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

echo -e "${BOLD}"
echo "=================================================================="
echo "   Federated Privacy-Preserving Tamil Court ASR"
echo "   One-Shot Setup Script"
echo "=================================================================="
echo -e "${RESET}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- 1. Python check ----------------------------------------------------------
info "Checking Python version..."
PYTHON_BIN=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_BIN="$cmd"
            success "Found Python $VER -> $cmd"
            break
        fi
    fi
done
[ -z "$PYTHON_BIN" ] && error "Python 3.9+ is required. Install from https://python.org and retry."

# -- 2. .env setup ------------------------------------------------------------
if [ -f ".env" ]; then
    warn ".env already exists -- skipping copy (delete it manually to reset)."
else
    if [ ! -f ".env.example" ]; then
        error ".env.example not found. Run this script from the repo root."
    fi
    cp .env.example .env
    success ".env created from .env.example"
    echo ""
    echo "  IMPORTANT: Open .env and fill in your HF_TOKEN and other secrets"
    echo "  before using hosted federation mode."
    echo "  Simulation mode works fine without any token."
    echo ""
fi

# -- 3. Virtual environment ---------------------------------------------------
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at .venv -- skipping creation."
else
    info "Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created."
fi

# Activate (works on both Unix and Git Bash on Windows)
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    error "Could not find venv activate script."
fi
success "Virtual environment activated."

# -- 4. Pip + requirements ----------------------------------------------------
info "Upgrading pip..."
pip install --quiet --upgrade pip

# Auto-detect CUDA and install the right PyTorch build
TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || true)
if [ -z "$TORCH_INSTALLED" ]; then
    info "Detecting GPU/CUDA for PyTorch installation..."
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" | head -1 || echo "")
        if [ "$CUDA_VER" = "12" ]; then
            info "CUDA 12.x detected -- installing PyTorch cu121..."
            pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
        elif [ "$CUDA_VER" = "11" ]; then
            info "CUDA 11.x detected -- installing PyTorch cu118..."
            pip install --quiet torch --index-url https://download.pytorch.org/whl/cu118
        else
            warn "GPU found but CUDA version unclear -- installing CPU PyTorch."
            pip install --quiet torch
        fi
    else
        warn "No NVIDIA GPU detected -- installing CPU-only PyTorch."
        pip install --quiet torch
    fi
else
    info "PyTorch already installed ($TORCH_INSTALLED) -- skipping."
fi

info "Installing project requirements..."
pip install --quiet -r requirements.txt
success "All dependencies installed."

# -- 5. Verify core imports ---------------------------------------------------
info "Verifying key imports..."
python -c "import fastapi, uvicorn, transformers, peft, librosa, safetensors" \
    && success "All imports OK." \
    || error "Import check failed -- see output above for details."

# -- 6. Install Frontend Dependencies ------------------------------------------
info "Installing Next.js frontend dependencies..."
cd frontend && npm install && cd ..
success "Frontend dependencies ready."

# -- 7. Launch Instructions ---------------------------------------------------
echo ""
echo "Setup complete!"
echo "To start the application:"
echo "  1. Backend API:  uvicorn backend.api:app --port 8080 --reload"
echo "  2. Frontend UI:  cd frontend && npm run dev"
echo "  3. Open:         http://localhost:3000 in your browser"
echo ""
