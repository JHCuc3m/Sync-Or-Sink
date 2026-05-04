# Final Project TODO List

**Goal:** Build a clean, reproducible live PPO/RLVR harness on WikiSQL text-to-SQL with
execution reward, run staleness sweeps, and validate the stability degradation hypothesis.
Scoped to single-process simulation at GPT-2 scale.

**Current status:** Initial experiments complete with LR=1e-6 (see FINDINGS.md).
Re-running all experiments with LR=1e-5 and 200 updates for stronger signal.

**Milestone results (complete):** DPO baseline, logprob parity test, simulated staleness test.

---

## 0. Pre-Sweep Infrastructure (DONE)

- [x] Add per-condition/seed output subdirectories → `scripts/live_ppo_rlvr.py` outputs to `outputs/live_ppo/<run_name>/`
- [x] Add PACE job template for live PPO harness → `jobs/run_live_ppo_rlvr.sbatch`
- [x] Implement all experimental conditions: sync, stale, mismatch, rescored, stale_mismatch → `scripts/live_ppo_rlvr.py`
- [x] Fix output directory naming to include lag for staleness conditions → `live_ppo_stale_lag{L}_wikisql_seed{S}`
- [x] Create consolidated sbatch job files for experiment sweeps
- [ ] Replace hardcoded cluster paths in `.sbatch` files (low priority, documented workaround)

---

## 1. WikiSQL Task + SFT Warm-Start (Phase 1) — COMPLETE

### Why WikiSQL over JSON

The synthetic JSON task lacks grounding. WikiSQL provides:
- Real dataset with RL precedent (Seq2SQL used policy gradient + execution reward)
- Verifiable execution reward (run SQL against SQLite, compare to gold result)
- Appropriate difficulty for GPT-2 (single-table queries, simple grammar)
- Better publication credibility

### Implementation

| Status | Task | File |
|--------|------|------|
| ✅ | Download WikiSQL data | `scripts/download_wikisql.sh` → `data/data/` |
| ✅ | WikiSQL data loading | `scripts/wikisql_utils.py` (`WikiSQLDataset` class) |
| ✅ | Prompt template | `scripts/wikisql_utils.py` (`build_prompt()`) |
| ✅ | SQLite execution reward | `scripts/wikisql_utils.py` (`score_sql_execution()`) |
| ✅ | SFT warm-start script | `scripts/sft_wikisql.py`, `jobs/run_sft_wikisql.sbatch` |
| ✅ | Integrate WikiSQL into live PPO | `scripts/live_ppo_rlvr.py` (`--task wikisql --checkpoint`) |
| ✅ | Run SFT warm-start | `sbatch jobs/run_sft_wikisql.sbatch` |
| ✅ | Run PPO with WikiSQL | `sbatch jobs/run_ppo_wikisql.sbatch` |

**Reward structure:**
- 1.0 = generated SQL executes to same answer as gold
- 0.5 = valid SQL, executes, wrong answer
- 0.1 = valid SQL syntax, execution error
- 0.0 = invalid SQL (parse error)

### Data splits

- Train: 2,000 examples (subset of WikiSQL train, 56K total)
- Eval: 200 examples (from WikiSQL dev, 8K total)

**Goal:** SFT model that produces valid SQL most of the time, providing foundation for PPO.

---

## 2. Sync PPO Baseline (Phase 2) — COMPLETE (re-running with higher LR)

- [x] Run sync PPO (L=0, fp32 consistent) from SFT checkpoint
- [x] Confirm stable training: clip fraction = 0, low ratio variance
- [x] Verify metrics are logged correctly: execution accuracy, valid SQL rate, clip fraction, entropy
- [x] Replicate with seeds 0, 1, 2

**Initial results (LR=1e-6):** Stable but flat eval metrics (~0.72 reward, ~0.46 pass@1). See FINDINGS.md.

**Re-run:** `sbatch jobs/run_sync_replication.sbatch` (LR=1e-5, 200 updates, seeds 0,1,2)

---

## 3. Staleness Sweep (Phase 3 — H2) — COMPLETE (re-running with higher LR)

- [x] Run live PPO with L ∈ {1, 2, 4} (initial run, LR=1e-6)
- [x] Collect per-lag metrics (ratio variance, logprob movement increased with lag)
- [ ] Re-run with L ∈ {1, 2, 4, 8} and LR=1e-5 for stronger signal
- [ ] Plot: stability metrics vs L, execution accuracy vs L

**Initial results (LR=1e-6, L=4 only):**
- Ratio variance increased +75% vs sync
- Abs logprob movement increased +147% vs sync
- But no clipping triggered, task performance unchanged

**Re-run:** `sbatch jobs/run_staleness_sweep.sbatch` (LR=1e-5, 200 updates, L=1,2,4,8)

---

## 4. Mismatch and Rescoring (Phase 4 — H1) — COMPLETE (re-running with higher LR)

- [x] **sync**: consistent fp32 logprobs throughout (baseline)
- [x] **mismatch**: actor-side fp16 logprobs used in PPO ratios
- [x] **rescored**: fp16 actor generates SQL, learner recomputes old logprobs in fp32
- [x] Compare clip fraction, ratio variance, and execution accuracy across all three
- [ ] Re-run with LR=1e-5 for stronger signal
- [ ] Plot: training curves for sync vs mismatch vs rescored

**Initial results (LR=1e-6):**
- Mismatch introduced pre-update ratio variance (0.000021 vs 0 for sync)
- Rescoring eliminated ratio variance (back to 0)
- Task performance unchanged across conditions

**Re-run:** `sbatch jobs/run_mismatch_sweep.sbatch` (LR=1e-5, 200 updates, seeds 0,1,2)

---

## 5. Stale + Mismatch (Phase 5 — H3) — COMPLETE (re-running with higher LR)

Combined staleness and fp16/fp32 mismatch to test H3 (combined drift reduces performance).

- [x] Run stale_mismatch L=2 seed 0 (initial run, LR=1e-6)
- [ ] Re-run with L ∈ {2, 4, 8} and seeds 0,1,2 for full coverage
- [ ] Compare against individual conditions

**Initial results (LR=1e-6, L=2, seed 0):**
- Intermediate ratio variance between stale-only and mismatch-only
- Pass@1 dropped 3% (0.463 → 0.449)

**Re-run:** `sbatch jobs/run_stale_mismatch.sbatch` (LR=1e-5, 200 updates, L=2,4,8, seeds 0,1,2)

---

## 6. Final Report

- [ ] Frame narrative: static proxy evidence (milestone) → WikiSQL RLVR → stability analysis
- [ ] Describe SFT warm-start methodology
- [ ] Present staleness results (H2): clip fraction and execution accuracy vs lag
- [ ] Present mismatch/rescoring results (H1): rescoring as mitigation
- [ ] Present combined stale+mismatch results (H3)
- [ ] Add replication summary (seed variance)
- [ ] Cite Seq2SQL and WikiSQL as task precedent
- [ ] Format to ICLR template, ≤ 8 pages + references

---

## Job Scripts

| Script | Experiments | Time | Status |
|--------|-------------|------|--------|
| `jobs/run_sync_replication.sbatch` | sync × seeds 0,1,2 | 10h | Ready |
| `jobs/run_mismatch_sweep.sbatch` | mismatch × 3, rescored × 3 | 16h | Ready |
| `jobs/run_staleness_sweep.sbatch` | L=1,2,4,8 × seeds | 30h | Ready |
| `jobs/run_stale_mismatch.sbatch` | L=2,4,8 × seeds 0,1,2 | 24h | Ready |

**Submit all:**
```bash
sbatch jobs/run_sync_replication.sbatch
sbatch jobs/run_mismatch_sweep.sbatch
sbatch jobs/run_staleness_sweep.sbatch
sbatch jobs/run_stale_mismatch.sbatch
```

---

## Stretch Goals

- [ ] Exact SQL match metric (stricter than execution accuracy)
- [ ] Spider subset as harder benchmark (future work mention)
- [ ] V-trace or KL-adaptive clip as additional mitigations

---

## Priority Order

1. ~~WikiSQL data + utilities~~ ✅ `scripts/wikisql_utils.py`, `scripts/download_wikisql.sh`
2. ~~SFT warm-start script~~ ✅ `scripts/sft_wikisql.py`, `jobs/run_sft_wikisql.sbatch`
3. ~~Integrate WikiSQL into live PPO~~ ✅ `scripts/live_ppo_rlvr.py`, `jobs/run_ppo_wikisql.sbatch`
4. ~~Run SFT warm-start~~ ✅
5. ~~Run sync PPO baseline (LR=1e-6)~~ ✅ → re-running with LR=1e-5
6. ~~Staleness sweep (LR=1e-6)~~ ✅ → re-running with LR=1e-5, L=1,2,4,8
7. ~~Mismatch and rescoring (LR=1e-6)~~ ✅ → re-running with LR=1e-5
8. ~~Stale+mismatch (LR=1e-6)~~ ✅ → re-running with LR=1e-5
9. **Analyze re-run results (LR=1e-5)**
10. Final report
