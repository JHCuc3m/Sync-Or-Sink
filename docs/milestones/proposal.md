# Sync or Sink: Stability Tradeoffs in Synchronous and Asynchronous RLHF

## Overview

This project studies stability issues in Reinforcement Learning from Human Feedback (RLHF)
when training is performed using an actor–learner architecture. In modern RLHF pipelines,
rollout generation and policy optimization are often decoupled to improve throughput.
However, this introduces two sources of drift:

1. Policy staleness caused by asynchronous actors generating rollouts from older policies.
2. Log-probability mismatch between different inference and training backends
   (e.g., vLLM for serving vs PyTorch / HuggingFace for training).

These effects can destabilize PPO training because PPO depends on token-level
probability ratios, which amplify small numerical differences.

The goal of this project is to build a reproducible RLHF pipeline and measure how
staleness and log-probability mismatch affect training stability and final task performance.

---

## Goals

We aim to:

- Implement a minimal RLHF actor–learner pipeline
- Measure stability of PPO under controlled drift
- Compare synchronous vs asynchronous rollouts
- Measure cross-backend log-probability mismatch
- Evaluate mitigation strategies

Key questions:

- Does staleness increase PPO instability?
- Does backend mismatch increase clipping / KL spikes?
- Do these effects hurt downstream task success?

---

## Project Scope

This project focuses on controlled, reproducible experiments rather than large-scale RLHF.

We use:

- Small language models (≤ 1–3B parameters)
- LoRA / QLoRA when needed
- Deterministic reward tasks (RLVR-style)
- Preference datasets for evaluation

We measure:

- KL divergence
- Clipping fraction
- Entropy
- Reward statistics
- Task success rate
- Preference win-rate

---

## Proposed Pipeline

Actor–Learner PPO setup:


Actor (vLLM or HF)
↓ rollouts
Replay buffer (with version tags)
↓
Learner (PyTorch / TRL PPO)
↓ update policy
Actors refreshed periodically


We control:

- policy lag L ∈ {0,1,2,4}
- backend consistency vs mismatch
- mitigation methods

---

## Hypotheses

H1 — Logprob mismatch increases PPO ratio noise  
Prediction: higher variance of ratios, more clipping, worse reward curves.

H2 — Staleness introduces off-policy bias  
Prediction: larger KL drift, noisier learning, lower correlation between advantages and ratios.

H3 — Drift reduces task performance  
Prediction: lower success rate / pass@1 / preference win-rate.

---

## Baselines

We will implement:

- DPO baseline on preference data
- PPO synchronous baseline
- PPO with controlled drift

Baselines provide a reference for stability and performance.

---

## Datasets / Tasks

Preference datasets:

- HH-RLHF
- UltraFeedback (optional)

Verifiable reward tasks:

- JSON format validation
- Regex / schema checks
- Deterministic transformation tasks

These allow low-variance reward signals.

---

## Metrics

Training stability:

- approx_KL
- clip_fraction
- entropy
- reward mean / variance

System metrics:

- staleness histogram
- logprob drift histogram
- throughput

Task metrics:

- success rate
- pass@1
- preference win-rate

---

## Environment Setup

Recommended:


conda create -n rlhf python=3.10
conda activate rlhf

pip install torch torchvision torchaudio
pip install transformers datasets accelerate trl peft
pip install matplotlib seaborn evaluate
pip install bitsandbytes


Optional:


pip install vllm

---

## Current Status

Milestone goal:

- Baseline training working
- Logprob comparison implemented
- Initial plots generated
- Pipeline partially implemented
- Milestone report written

Final goal:

- Full actor–learner PPO
- Drift experiments
- Mitigation experiments
- Final report