#!/bin/bash

# Setup script for Sync-Or-Sink project using uv
set -e  # Exit on error

echo "==> Setting up environment with uv..."

# 1. Install uv if not found
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "==> Creating .venv (Python 3.10)..."
    uv venv .venv --python 3.10
else
    echo "==> .venv already exists."
fi

# 3. Install packages
echo "==> Installing dependencies..."
source .venv/bin/activate

# Install Pytorch first (specify index for stability in cluster)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
uv pip install "transformers==4.46.3" datasets accelerate "trl==0.12.2" peft matplotlib seaborn evaluate bitsandbytes

# 4. Verify GPU
echo "==> Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "==> Setup complete. Run 'source .venv/bin/activate' to start."