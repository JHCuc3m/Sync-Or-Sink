# Final Project TODO List

Goal: Build a clean, reproducible live PPO/RLVR harness, run small staleness and
mismatch sweeps, validate fp32 rescoring as the primary mitigation, and write the
final report. Scoped to single-process simulation at GPT-2 scale — no distributed
actor/learner or vLLM integration required for the core results.

Builds on milestone results: DPO baseline, logprob parity test, and
simulated staleness test are already done.

---

## 0. Pre-Sweep Infrastructure

- [ ] Add per-condition/seed output subdirectories so runs do not overwrite `outputs/logs/` and `outputs/figures/`
- [ ] Add a new PACE job template under `jobs/` for the live PPO harness
- [ ] Replace hardcoded cluster paths in `.sbatch` files with `$SLURM_SUBMIT_DIR` or document the expected path

Goal: reproducible multi-run sweeps without manual output management.

---

## 1. Live PPO Harness + Verifiable Task (Phase 1)

- [ ] Create `scripts/live_ppo_rlvr.py` with CLI flags: `--condition`, `--lag`, `--seed`, `--updates`, `--task`, `--output_dir`
- [ ] Implement deterministic verifiable reward task — JSON extraction or regex-constrained formatting (not arithmetic)
- [ ] Log at every update step: reward mean, pass@1, clip fraction, KL proxy, entropy, ratio mean/variance, advantage mean/std
- [ ] Run sync baseline (L=0, fp32 consistent) and confirm stable training

Goal: clean reference point with L=0 and no backend mismatch; foundation for all sweeps.

---

## 2. Staleness Sweep (Phase 2 — H2)

- [ ] Simulate actor staleness in single-process: use rollout logprobs from a delayed policy snapshot
- [ ] Run live PPO with L ∈ {0, 1, 2, 4}; optionally L=8 if runs are stable and cheap
- [ ] Collect KL proxy, clip fraction, ratio variance, reward/pass@1 curves per L
- [ ] Plot: stability metrics vs L, reward curve vs L

Goal: show larger lag increases clip fraction, KL proxy, and ratio variance in live training.

---

## 3. Mismatch and Rescoring (Phase 3 — H1)

Run three conditions against the sync baseline:

- [ ] **control**: consistent fp32 logprobs throughout
- [ ] **mismatch**: actor-side fp16 logprobs used directly in PPO ratios
- [ ] **rescored**: actor generates responses with fp16, but learner recomputes old logprobs in fp32 before PPO update
- [ ] Compare clip fraction, KL proxy, and reward/pass@1 curves across all three
- [ ] Plot: training curves and clip fractions for control vs mismatch vs rescored

Goal: show fp16/fp32 mismatch causes spurious clipping, and learner-side rescoring recovers stability.

---

## 4. Replication (Phase 4)

- [ ] Re-run the core conditions (sync, L=2, L=4, mismatch, mismatch+rescored) over seeds 0, 1, 2
- [ ] Report mean ± std for key metrics across seeds

Goal: prefer replication over additional one-off conditions.

---

## 5. Final Report

- [ ] Frame narrative: static proxy evidence (milestone) → live PPO validation → rescoring mitigation
- [ ] Key result: PPO stability metrics degrade predictably with lag/mismatch; rescoring recovers stability (pass@1 gains are secondary)
- [ ] Expand technical approach with harness description and single-process staleness simulation design
- [ ] Add Phase 2 staleness results (H2) and Phase 3 mismatch/rescoring results (H1)
- [ ] Add replication summary (seed variance)
- [ ] Update related work if needed; write conclusion
- [ ] Format to ICLR template, ≤ 8 pages + references
- [ ] Proofread and submit

---

## Stretch Goals (only after core results exist)

- Combined `stale + mismatch` condition for H3 (2×2 grid: sync/async × matched/mismatched)
- Second verifiable task if JSON is too easy or too noisy
- Static vLLM-vs-HF logprob parity experiment before attempting full vLLM actor rollouts
- V-trace importance-weight clipping or KL-adaptive clip threshold as additional mitigations

---

## Priority Order

1. Pre-sweep infrastructure (blocks reproducible sweeps)
2. Live PPO harness + verifiable task (blocks everything else)
3. Staleness sweep — L ∈ {0, 1, 2, 4} (H2, core contribution)
4. Mismatch and rescoring (H1, core contribution)
5. Replication over seeds
6. Final report
7. Stretch: combined H3 evaluation, vLLM parity, additional mitigations
