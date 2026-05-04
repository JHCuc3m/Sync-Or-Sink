#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100
#SBATCH --mem=24G
#SBATCH -t 0:30:00
#SBATCH -J sync_replication
#SBATCH -o outputs/logs/sync_replication_%j.out
#SBATCH -e outputs/logs/sync_replication_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Sync Replication started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Common args - higher LR (1e-5) and more updates (200) for stronger signal
COMMON="--task wikisql \
  --checkpoint outputs/checkpoints/wikisql_sft/checkpoint.pt \
  --condition sync \
  --lag 0 \
  --updates 200 \
  --batch_size 4 \
  --ppo_epochs 1 \
  --lr 1e-5 \
  --eval_every 10 \
  --eval_batch_size 50 \
  --wikisql_train_examples 2000 \
  --wikisql_eval_examples 200 \
  --output_dir outputs/live_ppo"

# Sync baseline - seeds 0,1,2 (re-run all with new LR)
for seed in 0 1 2; do
  echo "==> Running: sync, seed=$seed ($(date))"
  python scripts/live_ppo_rlvr.py $COMMON --seed $seed
done

echo "==> Sync Replication finished: $(date)"
