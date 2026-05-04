# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sync or Sink** is an ML research project studying PPO training stability in actor–learner RLHF pipelines under two sources of drift:
1. **Policy staleness** — actors generating rollouts from stale checkpoints (lag L)
2. **Log-probability mismatch** — numerical drift between fp16 inference and fp32 training backends (e.g., vLLM actor vs PyTorch/HF learner)

PPO is particularly sensitive to both because it depends on token-level probability ratios that amplify small numerical differences. All experiments use GPT-2 as the base model. Runs on PACE COC-ICE cluster (SLURM).

## Research Hypotheses

**H1 — Logprob mismatch increases PPO ratio noise**: fp16/fp32 backend drift causes higher ratio variance, more clipping, worse reward curves.

**H2 — Staleness introduces off-policy bias**: larger KL drift, noisier learning, lower correlation between advantages and ratios as lag L increases.

**H3 — Drift reduces task performance**: lower success rate / pass@1 under combined staleness + backend mismatch.

## Environment Setup

```bash
bash setup_env.sh          # installs uv, creates .venv (Python 3.10), installs deps
source .venv/bin/activate  # activate before any script
```

Key pinned versions: `transformers==4.46.3`, `trl==0.12.2`, torch with CUDA 12.1.

**WikiSQL data (for text-to-SQL experiments):**
```bash
bash scripts/download_wikisql.sh
```
This downloads WikiSQL from GitHub to `data/data/` (~25MB compressed, ~185MB extracted).

## Running Experiments

**Interactive GPU session (debugging):**
```bash
salloc -A cse -q coc-ice -N 1 --gpus=1 --constraint=gpu-a100 -t 1:00:00
source .venv/bin/activate
python scripts/<script>.py
```

Note: A100 GPUs are required for compatibility with PyTorch 2.5.1+cu121. Newer GPUs (e.g., RTX 6000 Blackwell) require CUDA 13+.

**Batch jobs (experiment sweeps):**
```bash
# Main experiment sweeps (LR=1e-5, 200 updates)
sbatch jobs/run_sync_replication.sbatch   # Sync baseline (seeds 0,1,2) — 45min
sbatch jobs/run_mismatch_sweep.sbatch     # Mismatch + rescored (seeds 0,1,2) — 1h30
sbatch jobs/run_staleness_sweep.sbatch    # Staleness L=1,2,4,8 (seeds 0,1,2) — 2h
sbatch jobs/run_stale_mismatch.sbatch     # Combined stale+mismatch L=2,4,8 — 2h

# Individual jobs (milestone / setup)
sbatch jobs/run_dpo_baseline.sbatch       # DPO baseline (milestone)
sbatch jobs/run_logprob_parity.sbatch     # fp32/fp16 logprob comparison (milestone)
sbatch jobs/run_staleness_test.sbatch     # Simulated staleness test (milestone)
sbatch jobs/run_sft_wikisql.sbatch        # SFT warm-start on WikiSQL
sbatch jobs/run_ppo_wikisql.sbatch        # Single PPO run (WikiSQL)
```

**Monitor jobs:**
```bash
squeue -u $USER
cat outputs/logs/<jobname>_<JOBID>.out
```

**Submitting jobs:** All `.sbatch` files use `cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."` to resolve the repo root from the script's own location — submit from any directory or run directly as a shell script. Set `HF_HOME=/storage/home/hcoda1/6/jchen3392/scratch/.cache/huggingface` to avoid home-directory quota issues (already set in all sbatch files).

## Output Locations

All outputs go under `outputs/` (gitignored):
- `outputs/figures/` — PNG plots
- `outputs/logs/` — JSON metric logs + SLURM stdout/stderr
- `outputs/checkpoints/` — model checkpoints
  - `outputs/checkpoints/wikisql_sft/` — WikiSQL SFT warm-start checkpoint
- `outputs/live_ppo/<run_name>/` — live PPO logs and metrics plots
  - Naming: `live_ppo_{condition}_{task}_seed{S}` for sync/mismatch/rescored
  - Naming: `live_ppo_{condition}_lag{L}_{task}_seed{S}` for stale/stale_mismatch

## Architecture

Each script is self-contained (no shared library). All scripts resolve output paths relative to their own `__file__`:
```python
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
```

### Scripts

| Script | Purpose |
|---|---|
| `scripts/dpo_baseline.py` | GPT-2 + LoRA DPO on HH-RLHF, 200 steps. Args: `--num_samples`, `--max_steps`. |
| `scripts/logprob_parity.py` | Per-token Δlog-prob between fp32 and fp16 GPT-2 (H1 diagnostic). |
| `scripts/staleness_test.py` | SFT-trains GPT-2, saves checkpoints at {0,25,50,75,100} steps, computes PPO ratios vs lag L (H2 diagnostic). Caches checkpoints. |
| `scripts/plot_pipeline.py` | Actor–learner pipeline diagram figure (CPU-only). |
| `scripts/live_ppo_rlvr.py` | Live PPO harness with staleness/mismatch conditions. Supports JSON and WikiSQL tasks. |
| `scripts/wikisql_utils.py` | WikiSQL dataset loading, prompt construction, SQLite execution for reward. |
| `scripts/sft_wikisql.py` | SFT warm-start on WikiSQL (prompt → gold SQL). Saves checkpoint for PPO. |
| `scripts/download_wikisql.sh` | Downloads WikiSQL data from GitHub to `data/data/`. |

### live_ppo_rlvr.py Conditions

The live PPO harness supports these experimental conditions via `--condition`:
- `sync` — actor and learner use same fp32 policy (baseline, L=0)
- `stale` — actor uses lagged policy snapshot (use with `--lag L`)
- `mismatch` — actor computes old logprobs in fp16, learner updates in fp32
- `rescored` — fp16 actor generates rollouts, but learner recomputes old logprobs in fp32 (mitigation)
- `stale_mismatch` — combined stale + fp16 actor logprobs

**Recommended parameters (based on experiments):**
```bash
python scripts/live_ppo_rlvr.py \
  --task wikisql \
  --checkpoint outputs/checkpoints/wikisql_sft/checkpoint.pt \
  --condition sync \
  --lag 0 \
  --seed 0 \
  --updates 200 \
  --batch_size 4 \
  --ppo_epochs 1 \
  --lr 1e-5 \
  --eval_every 10 \
  --eval_batch_size 50 \
  --wikisql_train_examples 2000 \
  --wikisql_eval_examples 200 \
  --output_dir outputs/live_ppo
```

**Debug mode** (fixed good/bad responses to verify PPO signal):
```bash
python scripts/live_ppo_rlvr.py --debug_fixed_responses
```

### WikiSQL Task

The WikiSQL task uses text-to-SQL generation with execution-based reward:
- **Prompt**: Question + table schema with symbolic columns (col0, col1, ...)
- **Output**: SQL query (e.g., `SELECT col2 FROM t WHERE col0 = 'value'`)
- **Reward**: Based on SQL execution against SQLite
  - 1.0 = correct result
  - 0.5 = valid SQL, wrong result
  - 0.1 = valid syntax, execution error
  - 0.0 = invalid SQL

**SFT warm-start workflow:**
```bash
# 1. Download WikiSQL data
bash scripts/download_wikisql.sh

# 2. Run SFT to train on gold SQL
sbatch jobs/run_sft_wikisql.sbatch
# → outputs/checkpoints/wikisql_sft/checkpoint.pt

# 3. Run PPO experiment sweeps
sbatch jobs/run_sync_replication.sbatch
sbatch jobs/run_staleness_sweep.sbatch
sbatch jobs/run_mismatch_sweep.sbatch
sbatch jobs/run_stale_mismatch.sbatch
```

## Key Metrics

PPO stability is measured by: `approx_kl`, `clip_fraction`, `entropy`, reward mean/variance, `pass_at_1`. The PPO clip threshold is `ε=0.2`. Importance ratio is `r_t = exp(log π_current - log π_old)`.

## Current Progress

**Completed (LR=1e-5, 200 updates — primary results):**
- ✅ Sync baseline — seeds 0,1,2; eval pass@1 ~48%, post-update clip ~1.6%
- ✅ Rescored — seeds 0,1,2; pre-update ratio var = 0 exactly (mitigation holds at LR=1e-5)
- ✅ Staleness sweep — L=1,2,4,8 × seeds 0–2; pre-update clip fraction monotonically 0%→5.3% (L=8), strongly supports H2
- ✅ Stale+mismatch — L=2,4,8 × seeds 0–2; only condition to fall below sync baseline (L=4,8), supports H3

**Completed (LR=1e-6, 100 updates — earlier runs, superseded):**
- ✅ SFT warm-start → `scripts/sft_wikisql.py` (loss: 5.05 → 0.37, valid SQL: 100%)
- ✅ Sync/mismatch/rescored/stale(L=4)/stale_mismatch(L=2) — all three seeds

**Pending:**
- ⏳ Mismatch condition re-run at LR=1e-5 (rescored done, mismatch not yet)

**Next steps:**
1. Run mismatch sweep at LR=1e-5 (`sbatch jobs/run_mismatch_sweep.sbatch`)
2. Generate comparison plots (lag vs pre-clip, lag vs ratio-var, training curves)
3. Final report (ICLR template, ≤8 pages)

**Hypothesis status:**
- H1 (logprob mismatch → ratio noise): ✅ Supported at LR=1e-6; rescoring mitigation confirmed at LR=1e-5
- H2 (staleness → off-policy bias): ✅ Strongly supported — pre-clip 0%→5.3% monotonic with lag
- H3 (combined drift → task degradation): ⚠️ Partially supported — stale+mismatch L=4,8 below sync; stale-only does not degrade

See `docs/FINDINGS.md` for detailed results and `docs/todos_next.md` for task tracking.

## Report

Milestone report source: [Overleaf](https://www.overleaf.com/project/6984ea9012dc6ef4b4563f9b).
