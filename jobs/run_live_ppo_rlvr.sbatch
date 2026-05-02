#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J live_ppo_rlvr
#SBATCH -o outputs/logs/live_ppo_rlvr_%j.out
#SBATCH -e outputs/logs/live_ppo_rlvr_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Job started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python scripts/live_ppo_rlvr.py \
  --condition sync \
  --task easy_json \
  --lag 0 \
  --seed 0 \
  --updates 20 \
  --batch_size 2 \
  --ppo_epochs 1 \
  --eval_every 5 \
  --eval_batch_size 32 \
  --output_dir outputs/live_ppo

echo "==> Job finished: $(date)"
