#!/bin/bash
# Setup script for FewShot3DBrain reproduction
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=== FewShot3DBrain Setup ==="

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Install Python 3.9+."
    exit 1
fi
PYTHON=$(command -v python3 2>/dev/null || command -v python)
echo "Using: $($PYTHON --version)"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    $PYTHON -m venv venv
fi
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null || true

# Install deps
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Download weights
if [ ! -f "weights/fomo25_mmunetvae_pretrained.ckpt" ]; then
    echo "Downloading pretrained weights..."
    chmod +x scripts/download_champion_weights.sh
    ./scripts/download_champion_weights.sh
else
    echo "Weights already present: weights/fomo25_mmunetvae_pretrained.ckpt"
fi

echo ""
echo "Setup complete. Activate with: source venv/bin/activate"
echo "Quick test: python run_continual_lora.py --create_dummy --epochs 2 --lora_decoder"
echo "Full reproduction: see REPRODUCTION.md"
