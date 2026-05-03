#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J dpo_baseline
#SBATCH -o outputs/logs/dpo_baseline_%j.out
#SBATCH -e outputs/logs/dpo_baseline_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Job started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python scripts/dpo_baseline.py --num_samples 2000 --max_steps 200

echo "==> Job finished: $(date)"
