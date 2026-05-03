#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -J sft_wikisql
#SBATCH -o outputs/logs/sft_wikisql_%j.out
#SBATCH -e outputs/logs/sft_wikisql_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Job started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python scripts/sft_wikisql.py \
  --max_steps 500 \
  --batch_size 4 \
  --lr 5e-5 \
  --num_train_examples 2000 \
  --num_eval_examples 200 \
  --eval_every 100 \
  --eval_samples 50 \
  --seed 42 \
  --verbose

echo "==> Job finished: $(date)"
