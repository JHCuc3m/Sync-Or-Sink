# Milestone TODO List

Goal: Complete milestone report with baseline results,
implementation progress, and a concrete plan.

Target requirements:

- Baseline experiment
- Some results / plots
- Partial implementation of method
- Clear technical description
- Next-step plan

---

## 1. Environment Setup

- [x] Create venv (uv)
- [x] Install torch
- [x] Install transformers
- [x] Install datasets
- [x] Install accelerate
- [x] Install trl
- [x] Install peft
- [x] Install matplotlib / seaborn
- [x] Verify GPU works

Test:

- [x] Load model
- [x] Run forward pass

---

## 2. Minimal Training Pipeline

- [x] Load small model (gpt2 / tiny model)
- [x] Load tokenizer
- [x] Load dataset
- [ ] Implement reward function
- [ ] Run simple training loop

Goal:

✔ model trains without crash

---

## 3. Baseline Experiment (REQUIRED)

Choose one:

- [x] DPO baseline
- [ ] PPO synchronous baseline
- [ ] SFT baseline

Recommended:

DPO baseline

Tasks:

- [x] load preference dataset
- [x] run training (job submitted)
- [x] save logs
- [x] plot loss curve

Outputs:

- [x] training curve
- [x] metric plot

Needed for milestone grading

---

## 4. Logprob Parity Test

Implement:

logp1 = model A
logp2 = model B

Compute:

delta = logp1 - logp2

Tasks:

- [x] compute logprobs
- [x] compare
- [x] plot histogram

Outputs:

- [x] drift histogram
- [x] mean / std

Counts as method progress

---

## 5. Simulated Staleness Test (optional but good)

- [x] use older checkpoint (steps 0, 25, 50, 100)
- [x] compute ratios
- [x] measure KL / clipping

Output:

- [x] ratio variance plot
- [x] KL plot

Good for milestone but optional

---

## 6. Figures Needed for Report

- [x] pipeline diagram
- [x] training curve
- [x] drift histogram
- [x] metric plot

Figures give points even if method incomplete

---

## 7. Write Milestone Report

### Introduction

- [x] problem description
- [x] motivation
- [x] pipeline figure

### Related Work

- [x] PPO
- [x] DPO
- [x] IMPALA
- [x] SEED RL
- [x] RLHF pipeline
- [x] vLLM

Need ≥5 citations (7 included)

### Technical Approach

- [x] define input/output
- [x] define pipeline
- [x] define metrics
- [x] define hypotheses
- [x] describe training

### Experiments

- [x] dataset description
- [x] baseline description
- [x] results plots
- [x] progress on method

### Next Steps

- [x] timeline
- [x] risks
- [x] plan

### Formatting

- [x] ICLR template (compile on Overleaf or local LaTeX)
- [x] ≤ 5 pages
- [x] references

---

## 8. Final Check

- [x] baseline results included
- [x] at least 1 plot
- [x] method progress shown
- [x] plan included
- [x] hypotheses included
- [x] technical details clear

If all checked → milestone should score well