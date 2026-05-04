#!/bin/bash
#SBATCH -A cse
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100
#SBATCH --mem=24G
#SBATCH -t 0:30:00
#SBATCH -J staleness_sweep
#SBATCH -o outputs/logs/staleness_sweep_%j.out
#SBATCH -e outputs/logs/staleness_sweep_%j.err

cd /home/hice1/jchen3392/Sync-Or-Sink
source .venv/bin/activate

echo "==> Staleness Sweep started: $(date)"
echo "==> Node: $(hostname)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Common args - higher LR (1e-5) and more updates (200) for stronger signal
COMMON="--task wikisql \
  --checkpoint outputs/checkpoints/wikisql_sft/checkpoint.pt \
  --condition stale \
  --updates 200 \
  --batch_size 4 \
  --ppo_epochs 1 \
  --lr 1e-5 \
  --eval_every 10 \
  --eval_batch_size 50 \
  --wikisql_train_examples 2000 \
  --wikisql_eval_examples 200 \
  --output_dir outputs/live_ppo"

# L=1, seed 0
echo "==> Running: stale L=1, seed=0 ($(date))"
python scripts/live_ppo_rlvr.py $COMMON --lag 1 --seed 0

# L=2, seeds 0,1,2
for seed in 0 1 2; do
  echo "==> Running: stale L=2, seed=$seed ($(date))"
  python scripts/live_ppo_rlvr.py $COMMON --lag 2 --seed $seed
done

# L=4, seeds 0,1,2
for seed in 0 1 2; do
  echo "==> Running: stale L=4, seed=$seed ($(date))"
  python scripts/live_ppo_rlvr.py $COMMON --lag 4 --seed $seed
done

# L=8, seeds 0,1,2 (to potentially trigger clipping)
for seed in 0 1 2; do
  echo "==> Running: stale L=8, seed=$seed ($(date))"
  python scripts/live_ppo_rlvr.py $COMMON --lag 8 --seed $seed
done

echo "==> Staleness Sweep finished: $(date)"
