# Final Project TODO List

**Goal:** Build a clean, reproducible live PPO/RLVR harness on WikiSQL text-to-SQL with
execution reward, run staleness sweeps, and validate the stability degradation hypothesis.
Scoped to single-process simulation at GPT-2 scale.

**Current status:** All experiments complete at LR=1e-5, 200 updates (see FINDINGS.md).
Moving to figures and final report.

**Milestone results (complete):** DPO baseline, logprob parity test, simulated staleness test.

---

## 0. Infrastructure — COMPLETE

- [x] Per-condition/seed output subdirectories → `outputs/live_ppo/<run_name>/`
- [x] All experimental conditions implemented → `scripts/live_ppo_rlvr.py`
- [x] Lag included in output directory name for stale/stale_mismatch → `live_ppo_stale_lag{L}_wikisql_seed{S}`
- [x] Consolidated sbatch job files for all sweeps
- [x] sbatch files use `realpath`-based `cd` — work from any submission path or run directly
- [x] `HF_HOME` set in all sbatch files → avoids home-directory quota errors

---

## 1. WikiSQL Task + SFT Warm-Start — COMPLETE

| Status | Task |
|--------|------|
| ✅ | WikiSQL data download → `data/data/` |
| ✅ | Data loading, prompt template, SQLite execution reward → `scripts/wikisql_utils.py` |
| ✅ | SFT warm-start (loss 5.05 → 0.37, valid SQL 100%) → `scripts/sft_wikisql.py` |
| ✅ | SFT checkpoint → `outputs/checkpoints/wikisql_sft/checkpoint.pt` |
| ✅ | WikiSQL integrated into live PPO harness |

---

## 2. Sync PPO Baseline (H2 control) — COMPLETE

- [x] LR=1e-6, 100 updates, seeds 0,1,2 (superseded)
- [x] LR=1e-5, 200 updates, seeds 0,1,2
  - Eval pass@1 ~48%, post-update clip ~1.6%, pre-update ratio var = 0
  - Clipping now active at this LR — better signal for drift comparisons

---

## 3. Staleness Sweep (H2) — COMPLETE

- [x] LR=1e-6, L=4 only, seeds 0,1,2 (superseded — naming bug lost L=1,2)
- [x] LR=1e-5, L ∈ {1,2,4,8}, seeds 0–2 (L=1: seed 0 only)
  - Pre-update clip fraction: 0% (sync) → 0.16% (L=1) → 0.60% (L=2) → 2.0% (L=4) → **5.3% (L=8)**
  - Pre-update ratio variance grows monotonically; logprob movement 3× sync at L=8
  - **H2 strongly supported**

---

## 4. Mismatch and Rescoring (H1) — COMPLETE

- [x] LR=1e-6, 100 updates, seeds 0,1,2 (superseded)
- [x] LR=1e-5, 200 updates, seeds 0,1,2
  - Mismatch pre-update ratio var = 0.0000376 (nonzero, consistent across seeds)
  - Rescored pre-update ratio var = 0.0 exactly (mitigation holds at LR=1e-5)
  - Pre-update clip = 0 for both — fp16/fp32 gap too small to trigger clip directly
  - **H1 supported**; rescoring confirmed as robust mitigation

---

## 5. Stale + Mismatch (H3) — COMPLETE

- [x] LR=1e-6, L=2, seed 0 only (superseded)
- [x] LR=1e-5, L ∈ {2,4,8}, seeds 0–2 (9 runs)
  - At L=2: pre-clip 70% higher than stale-only; task metrics similar to sync
  - At L=4: eval pass@1 0.447 vs sync 0.480 — **below sync baseline**
  - At L=8: eval pass@1 0.453 vs sync 0.480 — **below sync baseline**
  - Stale-only at same lags does not degrade — mismatch is the differentiating factor
  - **H3 partially supported**

---

## 6. Figures — TODO

- [ ] **H2 figure**: lag L (x-axis) vs pre-update clip fraction — line plot with error bars across seeds
- [ ] **H2 figure**: lag L (x-axis) vs pre-update ratio variance — same format
- [ ] **H1 figure**: training curves (sync / mismatch / rescored) — ratio variance panel + logprob movement panel
- [ ] **H3 figure**: grouped bar — eval pass@1 across conditions at matched lags (stale vs stale+mismatch vs sync)
- [ ] **Overview figure**: actor-learner pipeline diagram showing where staleness and mismatch enter (`scripts/plot_pipeline.py`)

---

## 7. Final Report — TODO

- [ ] Frame narrative: static proxy evidence (milestone) → WikiSQL RLVR → stability analysis
- [ ] Describe SFT warm-start methodology and WikiSQL task
- [ ] Present H2 results: pre-clip fraction and ratio variance monotonically increasing with lag
- [ ] Present H1 results: mismatch ratio inflation, rescoring as mitigation
- [ ] Present H3 results: stale+mismatch as the only condition below sync baseline
- [ ] Seed variance / replication summary table
- [ ] Cite Seq2SQL, WikiSQL, PPO, and RLHF literature
- [ ] Format to ICLR template, ≤8 pages + references

---

## Job Scripts (all complete)

| Script | Experiments | Wall time | Status |
|--------|-------------|-----------|--------|
| `jobs/run_sync_replication.sbatch` | sync × seeds 0,1,2 | 0:45 | ✅ Done |
| `jobs/run_mismatch_sweep.sbatch` | mismatch × 3, rescored × 3 | 1:30 | ✅ Done |
| `jobs/run_staleness_sweep.sbatch` | L=1,2,4,8 × seeds 0–2 | 2:00 | ✅ Done |
| `jobs/run_stale_mismatch.sbatch` | L=2,4,8 × seeds 0–2 | 2:00 | ✅ Done |

---

## Stretch Goals

- [ ] Exact SQL match metric (stricter than execution accuracy)
- [ ] Spider subset as harder benchmark (future work mention)
- [ ] V-trace or KL-adaptive clip as additional mitigations

---

## Priority Order

1. ~~WikiSQL data + utilities~~ ✅
2. ~~SFT warm-start~~ ✅
3. ~~Integrate WikiSQL into live PPO~~ ✅
4. ~~Sync PPO baseline~~ ✅ LR=1e-5, seeds 0,1,2
5. ~~Staleness sweep~~ ✅ LR=1e-5, L=1,2,4,8
6. ~~Mismatch and rescoring~~ ✅ LR=1e-5, seeds 0,1,2
7. ~~Stale+mismatch~~ ✅ LR=1e-5, L=2,4,8, seeds 0,1,2
8. **Generate figures** ← current priority
9. **Write final report**
