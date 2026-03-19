#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 0:45:00
#SBATCH -J staleness_test
#SBATCH -o outputs/logs/staleness_test_%j.out
#SBATCH -e outputs/logs/staleness_test_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Job started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python scripts/staleness_test.py

echo "==> Job finished: $(date)"
