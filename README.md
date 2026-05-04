# Sync or Sink — Stability Tradeoffs in Synchronous and Asynchronous RLHF

Study of PPO training stability under two sources of drift in actor-learner RLHF pipelines:

1. **Policy staleness** — actors generating rollouts from stale checkpoints (lag L)
2. **Log-probability mismatch** — numerical drift between fp16 inference and fp32 training backends

## Research Hypotheses

- **H1** — fp16/fp32 backend drift causes higher ratio variance, more clipping, worse reward curves.
- **H2** — Larger lag L introduces off-policy bias: noisier learning, higher KL drift.
- **H3** — Combined staleness + mismatch reduces task performance below the sync baseline.

---

## Environment Setup

Requires Python 3.10 and CUDA. Uses [uv](https://github.com/astral-sh/uv) for fast package management.

```bash
git clone <repo>
cd Sync-Or-Sink
bash setup_env.sh
```

This will:

- Install `uv` if not present
- Create `.venv` with Python 3.10
- Install all dependencies (torch cu121, transformers 4.46.3, trl 0.12.2, peft, datasets, etc.)
- Verify GPU availability

Activate the environment before running any script:

```bash
source .venv/bin/activate
```

Download WikiSQL data (required for main experiments):

```bash
bash scripts/download_wikisql.sh
```

---

## Getting a GPU on PACE (COC-ICE Cluster)

**Important:** A100 GPUs are required for compatibility with PyTorch 2.5.1+cu121. Newer GPUs (e.g., RTX 6000 Blackwell) require CUDA 13+ which is not yet supported.

**Option A — Interactive session** (for debugging):

```bash
salloc -A cse -q coc-ice -N 1 --gpus=1 --constraint=gpu-a100 -t 1:00:00
source .venv/bin/activate
python scripts/<script>.py
```

**Option B — Batch jobs** (recommended for full runs):

```bash
sbatch jobs/<jobscript>.sbatch
```

Check job status and logs:

```bash
squeue -u $USER
cat outputs/logs/<jobname>_<JOBID>.out
```

---

## Reproducing Results

All outputs are written under `outputs/` (gitignored). Set `HF_HOME` to a scratch path to avoid home-directory quota issues — all sbatch files already do this.

### Main Experiment Sweeps (primary results, LR=1e-5, 200 updates)

Run in order — the PPO sweeps depend on the SFT checkpoint.

```bash
# 1. SFT warm-start on WikiSQL
sbatch jobs/run_sft_wikisql.sbatch
# → outputs/checkpoints/wikisql_sft/checkpoint.pt

# 2. Sync baseline (seeds 0,1,2) — ~45 min
sbatch jobs/run_sync_replication.sbatch

# 3. Mismatch + rescored (seeds 0,1,2) — ~1h30
sbatch jobs/run_mismatch_sweep.sbatch

# 4. Staleness sweep: L=1,2,4,8 × seeds 0,1,2 — ~2h
sbatch jobs/run_staleness_sweep.sbatch

# 5. Stale+mismatch: L=2,4,8 × seeds 0,1,2 — ~2h
sbatch jobs/run_stale_mismatch.sbatch
```

Live PPO outputs are written to `outputs/live_ppo/<run_name>/`:
- `live_ppo_{condition}_{task}_seed{S}` — for sync / mismatch / rescored
- `live_ppo_{condition}_lag{L}_{task}_seed{S}` — for stale / stale_mismatch

### Milestone / Diagnostic Scripts

```bash
sbatch jobs/run_dpo_baseline.sbatch       # GPT-2 + LoRA DPO on HH-RLHF
sbatch jobs/run_logprob_parity.sbatch     # fp32 vs fp16 per-token logprob comparison
sbatch jobs/run_staleness_test.sbatch     # Simulated staleness: PPO ratios vs lag L
python scripts/plot_pipeline.py           # Actor–learner pipeline diagram (CPU-only)
```

### Conditions (`--condition` flag in `live_ppo_rlvr.py`)

| Condition | Description |
|---|---|
| `sync` | Actor and learner share the same fp32 policy (baseline, L=0) |
| `mismatch` | Actor computes old logprobs in fp16; learner updates in fp32 |
| `rescored` | fp16 actor generates rollouts; learner recomputes old logprobs in fp32 (mitigation) |
| `stale` | Actor uses a lagged policy snapshot (`--lag L`) |
| `stale_mismatch` | Combined staleness + fp16 actor logprobs |

### WikiSQL Reward

| Output | Reward |
|---|---|
| Correct result | 1.0 |
| Valid SQL, wrong result | 0.5 |
| Valid syntax, execution error | 0.1 |
| Invalid SQL | 0.0 |

---

## Project Structure

```
Sync-Or-Sink/
├── setup_env.sh               # Environment setup
├── scripts/
│   ├── live_ppo_rlvr.py       # Main PPO harness (all staleness/mismatch conditions)
│   ├── wikisql_utils.py       # WikiSQL dataset loading + SQLite execution reward
│   ├── sft_wikisql.py         # SFT warm-start on WikiSQL
│   ├── download_wikisql.sh    # Download WikiSQL data to data/data/
│   ├── dpo_baseline.py        # GPT-2 + LoRA DPO on HH-RLHF (milestone)
│   ├── logprob_parity.py      # fp32 vs fp16 logprob comparison (milestone)
│   ├── staleness_test.py      # Simulated staleness experiment (milestone)
│   └── plot_pipeline.py       # Actor–learner pipeline diagram figure
├── jobs/
│   ├── run_sync_replication.sbatch
│   ├── run_mismatch_sweep.sbatch
│   ├── run_staleness_sweep.sbatch
│   ├── run_stale_mismatch.sbatch
│   ├── run_sft_wikisql.sbatch
│   ├── run_dpo_baseline.sbatch
│   ├── run_logprob_parity.sbatch
│   └── run_staleness_test.sbatch
├── data/
│   └── data/                  # WikiSQL dataset (downloaded, gitignored)
├── outputs/                   # All outputs (gitignored)
│   ├── figures/               # Generated plots
│   ├── logs/                  # JSON metric logs + SLURM output
│   ├── checkpoints/           # Model checkpoints
│   └── live_ppo/              # Live PPO experiment outputs
└── docs/
    ├── OVERVIEW.md            # Project guidance for Claude Code
    ├── FINDINGS.md            # Detailed results and analysis
    └── todos_next.md          # Task tracking
```

---

## Report

Milestone report source: [Overleaf](https://www.overleaf.com/project/6984ea9012dc6ef4b4563f9b)