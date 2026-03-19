#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -J logprob_parity
#SBATCH -o outputs/logs/logprob_parity_%j.out
#SBATCH -e outputs/logs/logprob_parity_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Job started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python scripts/logprob_parity.py

echo "==> Job finished: $(date)"
