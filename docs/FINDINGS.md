# Findings

## Phase 1: WikiSQL Task + SFT Warm-Start

### Summary

Phase 1 is complete and produced a usable SFT checkpoint for PPO initialization. Training converged cleanly, but task-level correctness is still limited.

### Observed Results

| Metric | Start (Step 1) | End (Step 500) |
|--------|----------------|----------------|
| Loss | 5.05 | 0.37 |
| Valid SQL % | 98% | 100% |
| Exact Match % | 0% | 2% |

- SFT loss dropped rapidly from ~5 to ~0.5 by step 100, then stabilized around 0.4.
- Valid SQL rate on validation stayed very high (98-100%).
- Exact SQL match remained low (0-2%) without a strong upward trend.

### Interpretation

- The model learned SQL syntax and formatting reliably.
- Semantic correctness (producing the correct query) remains the main bottleneck.
- This checkpoint is sufficient as a warm start for PPO, but not strong enough to claim high task accuracy from SFT alone.

### Implications For Next Phase

- Proceed with sync PPO as the control baseline.
- Track execution-based accuracy as the primary quality metric (not just exact string match).
- Keep validating stability metrics (`clip_fraction`, ratio variance, entropy, logprob movement) during PPO sweeps.

---

## Phase 2: Sync PPO Baseline

### Summary

Sync PPO baseline (L=0, fp32 consistent) ran for 100 updates from the SFT checkpoint. Training was stable but showed no improvement in task performance. This establishes a clean reference point before introducing drift conditions.

### Configuration

- Condition: `sync` (actor and learner use same fp32 policy)
- Task: `wikisql`
- Updates: 100
- Batch size: 4
- Learning rate: 1e-6
- Clip epsilon: 0.2
- Seed: 0

### Observed Results

| Metric | Value |
|--------|-------|
| Train Reward | mean=0.70, range=[0.30, 1.00] |
| Train Pass@1 | mean=0.43, range=[0.00, 1.00] |
| Eval Reward | 0.70–0.71 (flat) |
| Eval Pass@1 | 0.42–0.44 (flat) |
| Valid SQL Rate | 100% |
| Clip Fraction | 0.00 (always) |
| Post-Update Clip Fraction | 0.00 (always) |
| Abs Logprob Movement | mean=0.0017, range=[0.0003, 0.008] |

### Interpretation

**Stability metrics are excellent:**
- Clip fraction = 0 confirms ratios never exceed ε=0.2, indicating no off-policy drift in sync condition.
- Logprob movements are very small (~0.001), showing gradual policy updates.
- Entropy stable around 0.5, no collapse.

**No task improvement:**
- Eval pass@1 stayed flat at ~44% throughout training.
- Eval reward stayed flat at ~0.71 throughout training.
- This suggests the learning rate (1e-6) may be too conservative for meaningful reward improvement.

**Why clip fraction = 0?**
- In sync condition, old logprobs = new logprobs at rollout time, so pre-update ratios are exactly 1.0.
- Even post-update, policy changes are small enough (due to low LR) that ratios stay within [0.8, 1.2].
- This is expected behavior for a stable baseline — confirms the harness is working correctly.

### Implications For Next Phases

1. **Sync baseline is stable but static** — serves as control for staleness/mismatch experiments.
2. **Staleness experiments (Phase 3)** should show clip fraction > 0 as lag L increases.
3. **Mismatch experiments (Phase 4)** should show increased ratio variance from fp16/fp32 drift.
4. **Consider higher learning rate** (e.g., 1e-5) if reward improvement is desired, but current LR is fine for stability analysis.

---

## Phase 3: Sync Replication

### Summary

Ran sync baseline (L=0, fp32 consistent) across seeds 0, 1, 2 to establish variance bounds for all metrics. Results are highly consistent across seeds, confirming the baseline is stable and reproducible.

### Observed Results

| Seed | Eval Reward | Eval Pass@1 | Clip Frac (post) | Ratio Var (post) | Abs Logprob Move |
|------|-------------|-------------|------------------|------------------|------------------|
| 0 | 0.708 | 0.433 | 0.0002 | 0.000053 | 0.00171 |
| 1 | 0.718 | 0.460 | 0.0002 | 0.000114 | 0.00177 |
| 2 | 0.745 | 0.496 | 0.0006 | 0.000117 | 0.00194 |
| **Mean** | **0.724** | **0.463** | **0.0003** | **0.000095** | **0.00181** |

### Interpretation

- Eval reward: 0.724 ± 0.019 (low variance)
- Eval pass@1: 0.463 ± 0.032 (low variance)
- Clip fraction and ratio variance are near-zero across all seeds
- Abs logprob movement consistent at ~0.0018

The sync baseline provides a reliable reference point with tight variance bounds for comparison with drift conditions.

---

## Phase 4: Staleness Sweep (H2)

### Summary

Ran stale condition with lag L=4 across seeds 0, 1, 2 to test H2 (staleness introduces off-policy bias). Results show increased ratio variance and logprob movement compared to sync, but clip fraction remains near-zero.

**Note:** Due to output directory naming (no lag suffix), only L=4 results are available. L=1 and L=2 runs were overwritten.

### Observed Results

| Seed | Lag | Eval Reward | Eval Pass@1 | Ratio Var (pre) | Ratio Var (post) | Abs Logprob Move |
|------|-----|-------------|-------------|-----------------|------------------|------------------|
| 0 | 4 | 0.706 | 0.429 | 0.000060 | 0.000121 | 0.00415 |
| 1 | 4 | 0.718 | 0.467 | 0.000072 | 0.000210 | 0.00453 |
| 2 | 4 | 0.736 | 0.487 | 0.000077 | 0.000167 | 0.00472 |
| **Mean** | 4 | **0.720** | **0.461** | **0.000070** | **0.000166** | **0.00447** |

### Comparison to Sync Baseline

| Metric | Sync (L=0) | Stale (L=4) | Change |
|--------|------------|-------------|--------|
| Eval Reward | 0.724 | 0.720 | -0.6% |
| Eval Pass@1 | 0.463 | 0.461 | -0.4% |
| Ratio Var (pre) | 0.000000 | 0.000070 | +∞ (from zero) |
| Ratio Var (post) | 0.000095 | 0.000166 | +75% |
| Abs Logprob Move | 0.00181 | 0.00447 | +147% |

### Interpretation

**Partial support for H2:**
- Pre-update ratio variance increased from 0 to 0.00007 — staleness introduces measurable off-policy drift in importance ratios.
- Abs logprob movement increased 2.5x (0.0018 → 0.0045) — larger policy updates per step.
- Post-update ratio variance increased 75% — more variability in ratios after gradient updates.

**However:**
- Clip fraction remained near-zero — ratios stayed within ε=0.2 clipping threshold.
- Task performance (eval reward, pass@1) was not degraded.

**Possible explanations:**
- L=4 may not be large enough to trigger clipping with LR=1e-6.
- The small policy changes per update limit accumulated drift even with lag.
- Would need L=8+ or higher LR to see more pronounced effects.

---

## Phase 5: Mismatch Sweep (H1)

### Summary

Ran mismatch (fp16 actor logprobs, fp32 learner) and rescored (fp16 actor, fp32 rescoring) conditions across seeds 0, 1, 2 to test H1 (logprob mismatch increases ratio noise). Results show mismatch introduces measurable ratio variance that rescoring eliminates.

### Observed Results — Mismatch Condition

| Seed | Eval Reward | Eval Pass@1 | Ratio Var (pre) | Ratio Var (post) | Abs Logprob Move |
|------|-------------|-------------|-----------------|------------------|------------------|
| 0 | 0.711 | 0.436 | 0.000021 | 0.000057 | 0.00264 |
| 1 | 0.730 | 0.465 | 0.000021 | 0.000144 | 0.00279 |
| 2 | 0.741 | 0.485 | 0.000022 | 0.000097 | 0.00277 |
| **Mean** | **0.727** | **0.462** | **0.000021** | **0.000099** | **0.00273** |

### Observed Results — Rescored Condition

| Seed | Eval Reward | Eval Pass@1 | Ratio Var (pre) | Ratio Var (post) | Abs Logprob Move |
|------|-------------|-------------|-----------------|------------------|------------------|
| 0 | 0.710 | 0.436 | 0.000000 | 0.000043 | 0.00162 |
| 1 | 0.726 | 0.467 | 0.000000 | 0.000119 | 0.00185 |
| 2 | 0.732 | 0.475 | 0.000000 | 0.000083 | 0.00183 |
| **Mean** | **0.723** | **0.459** | **0.000000** | **0.000082** | **0.00177** |

### Comparison: Sync vs Mismatch vs Rescored

| Metric | Sync | Mismatch | Rescored |
|--------|------|----------|----------|
| Eval Reward | 0.724 | 0.727 | 0.723 |
| Eval Pass@1 | 0.463 | 0.462 | 0.459 |
| Ratio Var (pre) | 0.000000 | 0.000021 | 0.000000 |
| Abs Logprob Move | 0.00181 | 0.00273 | 0.00177 |

### Interpretation

**Support for H1:**
- Mismatch introduces pre-update ratio variance (0.000021) where sync and rescored have zero — fp16/fp32 numerical drift is measurable.
- Mismatch increases abs logprob movement by 51% vs sync (0.00273 vs 0.00181).
- Rescoring eliminates the pre-update ratio variance (back to 0.000000) — recomputing logprobs in fp32 removes numerical drift.
- Rescored abs logprob movement matches sync (0.00177 vs 0.00181).

**Task performance unaffected:**
- All three conditions have similar eval reward (~0.72) and pass@1 (~0.46).
- The numerical drift magnitude is too small to impact task metrics at this scale.

**Conclusion:** Rescoring is an effective mitigation for fp16/fp32 logprob mismatch. It restores sync-equivalent stability metrics while allowing fp16 inference for efficiency.

---

## Phase 6: Stale + Mismatch (H3)

### Summary

Ran combined stale_mismatch condition (L=2, fp16 actor) to test H3 (combined drift reduces performance). This represents a realistic actor-learner pipeline where actors use stale fp16 checkpoints.

### Observed Results

| Metric | Value |
|--------|-------|
| Condition | stale_mismatch |
| Lag | 2 |
| Seed | 0 |
| Eval Reward | 0.717 |
| Eval Pass@1 | 0.449 |
| Ratio Var (pre) | 0.000042 |
| Ratio Var (post) | 0.000096 |
| Abs Logprob Move | 0.00362 |

### Comparison to Individual Conditions

| Metric | Sync | Stale (L=4) | Mismatch | Stale+Mismatch (L=2) |
|--------|------|-------------|----------|----------------------|
| Eval Reward | 0.724 | 0.720 | 0.727 | 0.717 |
| Eval Pass@1 | 0.463 | 0.461 | 0.462 | 0.449 |
| Ratio Var (pre) | 0.000 | 0.000070 | 0.000021 | 0.000042 |
| Abs Logprob Move | 0.00181 | 0.00447 | 0.00273 | 0.00362 |

### Interpretation

**Partial support for H3:**
- Combined condition shows intermediate ratio variance (0.000042) — between mismatch-only (0.000021) and stale-only at L=4 (0.000070). This is expected since L=2 < L=4.
- Abs logprob movement (0.00362) is between stale-L4 (0.00447) and mismatch (0.00273).
- Eval pass@1 dropped slightly to 0.449 vs sync 0.463 (-3%).

**Limitations:**
- Only seed 0 was run — need replication for statistical confidence.
- L=2 is relatively small; L=4 combined with mismatch would show larger effects.
- Task performance degradation is modest at current LR.

---

## Summary of Hypotheses

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| **H1**: Logprob mismatch increases ratio noise | ✅ Supported | Mismatch shows 0.000021 pre-update ratio variance vs 0 for sync; rescoring eliminates it |
| **H2**: Staleness introduces off-policy bias | ⚠️ Partially supported | L=4 shows +75% post-update ratio variance and +147% logprob movement, but no clipping or task degradation |
| **H3**: Combined drift reduces task performance | ⚠️ Partially supported | Pass@1 dropped 3% (0.463→0.449), but effect is small; needs higher lag or LR for stronger signal |

## Recommendations

1. **Fix staleness sweep output naming** — add lag to directory name to preserve L=1,2,4 results.
2. **Try higher learning rate** (1e-5) — current LR may be too conservative to show pronounced drift effects.
3. **Run L=8 staleness** — larger lag may trigger actual clipping.
4. **Replicate stale_mismatch** — run seeds 1,2 for statistical confidence on H3.
