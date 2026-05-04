#!/bin/bash
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J ppo_exp
#SBATCH -o outputs/logs/ppo_exp_%j.out
#SBATCH -e outputs/logs/ppo_exp_%j.err

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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-ppo_final_experiment}"
TASK_NAME="${TASK_NAME:-structured_json}"
STEPS="${STEPS:-100}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-256}"
EVAL_SAMPLES="${EVAL_SAMPLES:-128}"
EVAL_EVERY="${EVAL_EVERY:-20}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
ACTOR_REFRESH_INTERVAL="${ACTOR_REFRESH_INTERVAL:-1}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-1}"
BUFFER_SAMPLING="${BUFFER_SAMPLING:-latest}"
ACTOR_DTYPE="${ACTOR_DTYPE:-fp32}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

"$PYTHON_BIN" scripts/ppo_sync_baseline.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --task_name "$TASK_NAME" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --ppo_epochs "$PPO_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --train_samples "$TRAIN_SAMPLES" \
  --eval_samples "$EVAL_SAMPLES" \
  --eval_every "$EVAL_EVERY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --actor_refresh_interval "$ACTOR_REFRESH_INTERVAL" \
  --replay_buffer_size "$REPLAY_BUFFER_SIZE" \
  --buffer_sampling "$BUFFER_SAMPLING" \
  --actor_dtype "$ACTOR_DTYPE" \
  $EXTRA_ARGS

echo "==> Job finished: $(date)"
