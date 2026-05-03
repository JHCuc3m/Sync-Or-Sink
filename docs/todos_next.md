# Final Project TODO List

Goal: Build a clean, reproducible live PPO/RLVR harness on WikiSQL text-to-SQL with
execution reward, run staleness sweeps, and validate the stability degradation hypothesis.
Scoped to single-process simulation at GPT-2 scale.

Builds on milestone results: DPO baseline, logprob parity test, and simulated staleness
test are already done. Live PPO harness infrastructure exists but needs WikiSQL task.

---

## 0. Pre-Sweep Infrastructure (DONE)

- [x] Add per-condition/seed output subdirectories
- [x] Add PACE job template for live PPO harness (`jobs/run_live_ppo_rlvr.sbatch`)
- [x] Implement all experimental conditions: sync, stale, mismatch, rescored, stale_mismatch
- [ ] Replace hardcoded cluster paths in `.sbatch` files (low priority, documented workaround)

---

## 1. WikiSQL Task + SFT Warm-Start (Phase 1)

### Why WikiSQL over JSON

The synthetic JSON task lacks grounding. WikiSQL provides:
- Real dataset with RL precedent (Seq2SQL used policy gradient + execution reward)
- Verifiable execution reward (run SQL against SQLite, compare to gold result)
- Appropriate difficulty for GPT-2 (single-table queries, simple grammar)
- Better publication credibility

### Implementation

- [ ] Add WikiSQL data loading (HuggingFace `wikisql` dataset)
- [ ] Implement prompt template: `Question: {nl} | Table: {schema} | SQL:`
- [ ] Implement SQLite execution for reward computation:
  - 1.0 if generated SQL executes to same answer as gold SQL
  - 0.5 if SQL parses and executes but wrong answer
  - 0.1 if valid SQL syntax but execution error/empty result
  - 0.0 if invalid SQL (parse error)
- [ ] Create `scripts/sft_wikisql.py` for supervised warm-start on gold SQL
- [ ] Update `scripts/live_ppo_rlvr.py` to support WikiSQL task alongside JSON
- [ ] Run SFT warm-start: ~500-1000 steps on train split subset

### Data splits

- Train prompts: 500–2,000 examples (subset of WikiSQL train)
- Eval prompts: 100–200 examples (fixed eval set)

Goal: SFT model that can produce valid SQL most of the time, providing a foundation for PPO.

---

## 2. Sync PPO Baseline (Phase 2)

- [ ] Run sync PPO (L=0, fp32 consistent) from SFT checkpoint
- [ ] Confirm stable training: reward improves, execution accuracy increases
- [ ] Verify metrics are logged correctly: execution accuracy, valid SQL rate, clip fraction, entropy

Goal: clean reference point before introducing drift.

---

## 3. Staleness Sweep (Phase 3 — H2)

- [ ] Run live PPO with L ∈ {0, 1, 2, 4}; optionally L=8 if stable
- [ ] Collect per-lag metrics:
  - Execution accuracy curve
  - Valid SQL rate
  - Post-update clip fraction
  - Post-update ratio variance
  - Post-update abs logprob movement
  - Entropy
- [ ] Plot: stability metrics vs L, execution accuracy vs L

Goal: show larger lag increases clip fraction, ratio variance, and degrades execution accuracy.

---

## 4. Mismatch and Rescoring (Phase 4 — H1)

Run three conditions against sync baseline:

- [ ] **control**: consistent fp32 logprobs throughout
- [ ] **mismatch**: actor-side fp16 logprobs used in PPO ratios
- [ ] **rescored**: fp16 actor generates SQL, learner recomputes old logprobs in fp32
- [ ] Compare clip fraction, ratio variance, and execution accuracy across all three
- [ ] Plot: training curves for control vs mismatch vs rescored

Goal: show fp16/fp32 mismatch causes spurious clipping, rescoring recovers stability.

---

## 5. Replication (Phase 5)

- [ ] Re-run core conditions (sync, L=2, L=4, mismatch, rescored) over seeds 0, 1, 2
- [ ] Report mean ± std for key metrics

Goal: statistical credibility over breadth.

---

## 6. Final Report

- [ ] Frame narrative: static proxy evidence (milestone) → WikiSQL RLVR → stability analysis
- [ ] Describe SFT warm-start methodology
- [ ] Present staleness results (H2): clip fraction and execution accuracy vs lag
- [ ] Present mismatch/rescoring results (H1): rescoring as mitigation
- [ ] Add replication summary (seed variance)
- [ ] Cite Seq2SQL and WikiSQL as task precedent
- [ ] Format to ICLR template, ≤ 8 pages + references

---

## Stretch Goals

- Combined `stale + mismatch` condition for H3 (2×2 grid)
- Exact SQL match metric (stricter than execution accuracy)
- Spider subset as harder benchmark (future work mention)
- V-trace or KL-adaptive clip as additional mitigations

---

## Priority Order

1. WikiSQL task + SFT warm-start (blocks everything)
2. Sync PPO baseline (validates harness works)
3. Staleness sweep L ∈ {0, 1, 2, 4} (H2, core contribution)
4. Mismatch and rescoring (H1, core contribution)
5. Replication over seeds
6. Final report
7. Stretch goals
