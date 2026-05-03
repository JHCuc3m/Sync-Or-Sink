#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -J ppo_wikisql
#SBATCH -o outputs/logs/ppo_wikisql_%j.out
#SBATCH -e outputs/logs/ppo_wikisql_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Job started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# WikiSQL PPO with SFT checkpoint
# Requires: sbatch jobs/run_sft_wikisql.sbatch to have completed first

python scripts/live_ppo_rlvr.py \
  --task wikisql \
  --checkpoint outputs/checkpoints/wikisql_sft/checkpoint.pt \
  --condition sync \
  --lag 0 \
  --seed 0 \
  --updates 100 \
  --batch_size 4 \
  --ppo_epochs 1 \
  --lr 1e-6 \
  --eval_every 10 \
  --eval_batch_size 50 \
  --wikisql_train_examples 2000 \
  --wikisql_eval_examples 200 \
  --output_dir outputs/live_ppo

echo "==> Job finished: $(date)"
