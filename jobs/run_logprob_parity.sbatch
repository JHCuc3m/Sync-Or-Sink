#!/bin/bash
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -J logprob_parity
#SBATCH -o outputs/logs/logprob_parity_%j.out
#SBATCH -e outputs/logs/logprob_parity_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
unset TRANSFORMERS_CACHE
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
if [ -x "/storage/ice1/2/9/eliu354/conda_envs/sync-or-sink/bin/python" ]; then
  PYTHON_BIN="/storage/ice1/2/9/eliu354/conda_envs/sync-or-sink/bin/python"
elif [ -x "$REPO_ROOT/.conda-run/bin/python" ]; then
  PYTHON_BIN="$REPO_ROOT/.conda-run/bin/python"
else
  source .venv/bin/activate
  PYTHON_BIN="python"
fi

echo "==> Job started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

"$PYTHON_BIN" scripts/logprob_parity.py

echo "==> Job finished: $(date)"
