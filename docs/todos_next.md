# Final Project TODO List

Goal: Complete full actor–learner PPO pipeline, run drift and mitigation
experiments, evaluate on verifiable tasks, and write the final report.

Builds on milestone results: DPO baseline, logprob parity test, and
simulated staleness test are already done.

---

## 1. Synchronous PPO Baseline

- [ ] Implement PPO training loop (TRL PPOTrainer or custom)
- [ ] Define reward function (verifiable task or preference score)
- [ ] Train on chosen task, log KL / clip fraction / entropy / reward
- [ ] Compare stability metrics against DPO baseline

Goal: clean reference point with L=0 and no backend mismatch.

---

## 2. Verifiable Reward Task Setup

- [ ] Choose task (JSON format validation, regex check, or similar)
- [ ] Implement deterministic reward function (pass/fail → scalar)
- [ ] Build prompt/dataset for the task
- [ ] Verify reward signal is low-variance and reproducible

Goal: enables H3 evaluation (task success rate / pass@1).

---

## 3. Full Actor–Learner Pipeline

- [ ] Implement replay buffer with version tags
- [ ] Decouple rollout generation from policy update
- [ ] Implement actor refresh logic (reload weights every K learner steps)
- [ ] Run pipeline end-to-end and confirm training is stable at L=0

Goal: working async pipeline as the foundation for drift experiments.

---

## 4. Staleness Drift Experiments (H2)

- [ ] Run PPO with lag L ∈ {0, 1, 2, 4} learner updates
- [ ] Collect KL, clip fraction, entropy, reward curves per L
- [ ] Compare reward curves and stability metrics across L values
- [ ] Plot: stability metrics vs L, reward curve vs L

Goal: quantify how staleness degrades live PPO training.

---

## 5. Backend Mismatch Experiments (H1)

- [ ] Run PPO with fp16 actor logprobs vs fp32 learner logprobs
- [ ] Run PPO with consistent fp32 logprobs (control)
- [ ] Compare clip fraction, KL, and reward curves between conditions
- [ ] Plot: drift histogram and training curves side by side

Goal: show backend mismatch causes spurious clipping in live training.

---

## 6. Combined Drift + Task Evaluation (H3)

- [ ] Run experiments combining staleness + backend mismatch
- [ ] Measure task success rate (pass@1) under each condition
- [ ] Compare: sync vs async × consistent vs mismatched backends
- [ ] Plot: 2×2 grid of task success rate vs training step

Goal: link stability metrics to downstream task performance.

---

## 7. Mitigation Experiments

- [ ] Mitigation A: recompute logprobs in fp32 at training time
- [ ] Mitigation B: V-trace importance-weight clipping (IMPALA-style)
- [ ] Mitigation C: KL-adaptive clip threshold
- [ ] Evaluate each mitigation against unmitigated baseline
- [ ] Plot: reward curves and clip fractions with vs without mitigation

Goal: show at least one mitigation recovers stability.

---

## 8. Final Report

- [ ] Expand technical approach with full pipeline description
- [ ] Add PPO experiment results (H1, H2, H3 sections)
- [ ] Add mitigation results section
- [ ] Update related work if needed
- [ ] Write conclusion summarizing findings
- [ ] Format to ICLR template, ≤ 8 pages + references
- [ ] Proofread and submit

---

## Priority Order

1. Synchronous PPO baseline (needed before everything else)
2. Verifiable task setup (needed for H3)
3. Full pipeline + staleness experiments (H2, core contribution)
4. Backend mismatch in live PPO (H1, core contribution)
5. Combined evaluation + task success (H3)
6. Mitigation experiments
7. Final report
