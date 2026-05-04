#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -J sync_replication
#SBATCH -o outputs/logs/sync_replication_%j.out
#SBATCH -e outputs/logs/sync_replication_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Sync Replication started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Common args
COMMON="--task wikisql \
  --checkpoint outputs/checkpoints/wikisql_sft/checkpoint.pt \
  --condition sync \
  --lag 0 \
  --updates 100 \
  --batch_size 4 \
  --ppo_epochs 1 \
  --lr 1e-6 \
  --eval_every 10 \
  --eval_batch_size 50 \
  --wikisql_train_examples 2000 \
  --wikisql_eval_examples 200 \
  --output_dir outputs/live_ppo"

# Sync baseline - seeds 1,2 (seed 0 already done)
for seed in 1 2; do
  echo "==> Running: sync, seed=$seed ($(date))"
  python scripts/live_ppo_rlvr.py $COMMON --seed $seed
done

echo "==> Sync Replication finished: $(date)"
