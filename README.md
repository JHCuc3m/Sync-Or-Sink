# Sync or Sink — Stability Tradeoffs in Synchronous and Asynchronous RLHF

Study of PPO training stability under two sources of drift in actor-learner RLHF pipelines:

1. **Policy staleness** — actors generating rollouts from stale checkpoints (lag L)
2. **Log-probability mismatch** — numerical drift between fp16 inference and fp32 training backends

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

---

## Getting a GPU on PACE (COC-ICE Cluster)

**Option A — Interactive session** (for debugging):

```bash
salloc -A cse -q coc-ice -N 1 --gpus=1 -t 1:00:00
# Requests 1 GPU for 1 hour on coc-ice partition and billed to the cse account
salloc -A coc -q coc-ice -N 1 --gpus=1 --constraint=gpu-l40s -t 1:00:00
# Requests 1 L40S GPU for 1 hour on coc-ice partition and billed to the coc account
```

Once allocated, activate the environment and run scripts directly:

```bash
source .venv/bin/activate
python scripts/<script>.py
```

**Option B — Batch jobs** (recommended for full runs):

```bash
sbatch jobs/<jobscript>.sbatch
```

Check job status:

```bash
squeue -u $USER
```

View output logs (replace `<JOBID>` with the actual job ID):

```bash
cat outputs/logs/<jobname>_<JOBID>.out
cat outputs/logs/<jobname>_<JOBID>.err
```

Check available GPU partitions:

```bash
sinfo -o "%20P %20G %5D %N"
```

---

## Reproducing Results

All outputs are written to `outputs/figures/` (plots) and `outputs/logs/` (JSON logs).

### 1. DPO Baseline

Trains GPT-2 + LoRA on Anthropic HH-RLHF with DPO for 200 steps.
Produces a training loss curve and reward accuracy metrics.

```bash
sbatch jobs/run_dpo_baseline.sbatch
```

Outputs:

- `outputs/figures/dpo_loss_curve.png`
- `outputs/logs/dpo_baseline_logs.json`

### 2. Logprob Parity Test

Compares per-token log-probabilities between fp32 and fp16 instances of GPT-2
to quantify backend numerical mismatch.

```bash
sbatch jobs/run_logprob_parity.sbatch
```

Outputs:

- `outputs/figures/logprob_drift_histogram.png`
- `outputs/logs/logprob_parity_stats.json`

### 3. Simulated Staleness Test

Fine-tunes GPT-2 with SFT for 100 steps, saving checkpoints at steps 0, 25, 50, 75, 100.
Computes PPO importance ratios between the final policy and each stale checkpoint
to measure KL divergence, ratio variance, and clip fraction vs. lag L.

```bash
sbatch jobs/run_staleness_test.sbatch
```

Checkpoints are saved to `outputs/checkpoints/`. Subsequent runs skip SFT training
and load from disk automatically.

Outputs:

- `outputs/figures/staleness_kl_clip.png`
- `outputs/figures/staleness_ratio_variance.png`
- `outputs/logs/staleness_stats.json`

### 4. Pipeline Diagram

Generates the actor-learner pipeline figure used in the report.
Runs on the login node (no GPU needed).

```bash
source .venv/bin/activate
python scripts/plot_pipeline.py
```

Output:

- `outputs/figures/pipeline_diagram.png`

---

## Project Structure

```
Sync-Or-Sink/
├── setup_env.sh               # Environment setup
├── scripts/
│   ├── dpo_baseline.py        # DPO training baseline
│   ├── logprob_parity.py      # fp32 vs fp16 logprob comparison
│   ├── staleness_test.py      # Simulated staleness experiment
│   └── plot_pipeline.py       # Pipeline diagram figure
├── jobs/
│   ├── run_dpo_baseline.sbatch
│   ├── run_logprob_parity.sbatch
│   └── run_staleness_test.sbatch
├── outputs/
│   ├── figures/               # Generated plots
│   ├── logs/                  # JSON metric logs + SLURM output
│   └── checkpoints/           # SFT model checkpoints (staleness test)
└── docs/
    ├── proposal.md
    ├── todos.md
```

---

## Milestone Report

The milestone report source is at [Overleaf](https://www.overleaf.com/project/6984ea9012dc6ef4b4563f9b).