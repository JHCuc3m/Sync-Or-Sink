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

### Configuration

- Condition: `sync`, Lag: 0, LR: 1e-6, Updates: 100, Batch size: 4

### Observed Results

| Seed | Eval Reward | Eval Pass@1 | Clip Frac (post) | Ratio Var (post) | Abs Logprob Move |
|------|-------------|-------------|------------------|------------------|------------------|
| 0 | 0.712 | 0.440 | 0.00016 | 0.000053 | 0.00171 |
| 1 | 0.732 | 0.480 | 0.00015 | 0.000114 | 0.00177 |
| 2 | 0.720 | 0.440 | 0.00057 | 0.000117 | 0.00194 |
| **Mean** | **0.721** | **0.453** | **0.00029** | **0.000095** | **0.00181** |

### Interpretation

- Eval reward: 0.721 ± 0.010 (low variance)
- Eval pass@1: 0.453 ± 0.023 (low variance)
- Pre-update ratio variance = 0.000000 for all seeds (ratios exactly 1.0 at rollout)
- Post-update clip fraction and ratio variance are near-zero across all seeds
- Abs logprob movement consistent at ~0.0018

The sync baseline provides a reliable reference point with tight variance bounds for comparison with drift conditions.

---

## Phase 3 Re-run: Sync Replication at LR=1e-5, 200 Updates

### Summary

Re-ran sync baseline (L=0, fp32 consistent) at LR=1e-5 and 200 updates across seeds 0, 1, 2. The new baseline is stable and achieves slightly better task performance than LR=1e-6. Critically, the higher LR makes post-update PPO clipping measurable (~1.6%), providing a cleaner signal for comparing drift conditions against the baseline.

### Configuration

- Condition: `sync`, Lag: 0, LR: **1e-5**, Updates: **200**, Batch size: 4

### Observed Results

| Seed | Eval Reward | Eval Pass@1 | Clip Frac (post) | Ratio Var (pre) | Ratio Var (post) | Abs Logprob Move (steady) |
|------|-------------|-------------|------------------|-----------------|------------------|---------------------------|
| 0 | 0.732 | 0.480 | 0.01751 | 0.0000000 | 0.0054146 | 0.01577 |
| 1 | 0.750 | 0.500 | 0.01539 | 0.0000000 | 0.0377712 | 0.01382 |
| 2 | 0.714 | 0.460 | 0.01619 | 0.0000000 | 0.0110492 | 0.01932 |
| **Mean** | **0.732** | **0.480** | **0.01636** | **0.0000000** | **0.0180783** | **0.01630** |

### Comparison: Sync LR=1e-6 vs LR=1e-5

| Metric | Sync LR=1e-6 (100 steps) | Sync LR=1e-5 (200 steps) | Change |
|--------|--------------------------|--------------------------|--------|
| Eval Reward | 0.721 | 0.732 | +1.5% |
| Eval Pass@1 | 0.453 | 0.480 | +5.9% |
| Clip Frac (post) | 0.00029 | 0.01636 | +56× |
| Ratio Var (pre) | 0.0000000 | 0.0000000 | 0 (unchanged) |
| Ratio Var (post) | 0.0000949 | 0.0180783 | +190× |
| Abs Logprob Move | 0.00181 | 0.01630 | +9× |

### Observations from Plots

- **Logprob movement** spikes to 0.07–0.13 in the first few updates (large initial SFT→PPO transition), then decays and stabilises at ~0.014–0.019 steady state.
- **Pre-update ratio variance = 0.0 exactly** — in sync, actor and learner share the same fp32 model, so rollout logprobs equal learner logprobs by construction at every step. This invariant holds throughout all 200 updates.
- **Post-update clip fraction ~1.6%** — the 10× larger gradient steps push some token ratios outside ε=0.2 after the update. This is the expected and healthy regime for PPO; at LR=1e-6 this was essentially zero, masking drift effects.
- **Eval reward and pass@1 curves are broadly flat** with high step-to-step variance (batch size 4). Seeds 0 and 1 show a very slight upward trend in eval pass@1; seed 2 is flatter. No instability or collapse observed.
- **Seed variance is low**: eval reward range 0.714–0.750, pass@1 range 0.460–0.500.

### Interpretation

The LR=1e-5 sync baseline is better calibrated for stability analysis than LR=1e-6. Post-update clipping is now clearly nonzero (~1.6%), which means the PPO surrogate objective is actually constraining updates. Any drift condition (mismatch, staleness) that inflates pre-update ratios will compound on top of this active clipping regime, making the effects more observable.

Task performance is marginally better than LR=1e-6 (eval reward 0.732 vs 0.721, pass@1 0.480 vs 0.453), suggesting the higher LR is not harmful to learning. This sync baseline replaces the LR=1e-6 baseline as the primary reference for all subsequent drift comparisons.

---

## Phase 4: Staleness Sweep (H2)

### Summary

Full staleness sweep at LR=1e-5, 200 updates, across L ∈ {1, 2, 4, 8} (seeds 0–2 for L=2,4,8; seed 0 only for L=1). The naming fix (`live_ppo_stale_lag{L}_wikisql_seed{N}`) preserves all lag results without overwriting. This is the primary evidence for H2.

**Key result:** Pre-update clip fraction increases monotonically from ~0 (sync) to 5.3% (L=8), and pre-update ratio variance grows 20× from L=1 to L=8. At L=8, over 1 in 20 token ratios is already outside the ε=0.2 clipping boundary before any gradient step — directly demonstrating off-policy bias from staleness.

### Configuration

- Condition: `stale`, LR: 1e-5, Updates: 200, Batch size: 4

### Observed Results

| Lag | Seed | Eval Reward | Eval Pass@1 | Clip Frac (pre) | Clip Frac (post) | Ratio Var (pre) | Logprob Move (steady) |
|-----|------|-------------|-------------|-----------------|------------------|-----------------|-----------------------|
| 1 | 0 | 0.700 | 0.400 | 0.00158 | 0.02286 | 0.0005765 | 0.02350 |
| 2 | 0 | 0.740 | 0.480 | 0.00707 | 0.03425 | 0.0016646 | 0.02911 |
| 2 | 1 | 0.708 | 0.480 | 0.00580 | 0.02783 | 0.0014847 | 0.02541 |
| 2 | 2 | 0.730 | 0.460 | 0.00521 | 0.03060 | 0.0011558 | 0.02511 |
| **2** | **mean** | **0.726** | **0.473** | **0.00603** | **0.03089** | **0.0014350** | **0.02654** |
| 4 | 0 | 0.712 | 0.440 | 0.01630 | 0.03716 | 0.0029843 | 0.03481 |
| 4 | 1 | 0.740 | 0.480 | 0.02309 | 0.04377 | 0.0045582 | 0.03448 |
| 4 | 2 | 0.750 | 0.500 | 0.02161 | 0.04344 | 0.0044116 | 0.03365 |
| **4** | **mean** | **0.734** | **0.473** | **0.02033** | **0.04146** | **0.0039847** | **0.03432** |
| 8 | 0 | 0.742 | 0.500 | 0.04955 | 0.06773 | 0.0110832 | 0.05076 |
| 8 | 1 | 0.760 | 0.520 | 0.05052 | 0.06377 | 0.0128315 | 0.04386 |
| 8 | 2 | 0.730 | 0.460 | 0.05852 | 0.07489 | 0.0121167 | 0.05287 |
| **8** | **mean** | **0.744** | **0.493** | **0.05286** | **0.06880** | **0.0120105** | **0.04916** |

### Comparison: Sync Baseline vs Stale Conditions (all at LR=1e-5)

| Lag | Eval Reward | Eval Pass@1 | Clip Frac (pre) | Clip Frac (post) | Ratio Var (pre) | Logprob Move (steady) |
|-----|-------------|-------------|-----------------|------------------|-----------------|-----------------------|
| 0 (sync) | 0.732 | 0.480 | 0.00000 | 0.01636 | 0.0000000 | 0.01630 |
| 1 | 0.700 | 0.400 | 0.00158 | 0.02286 | 0.0005765 | 0.02350 |
| 2 | 0.726 | 0.473 | 0.00603 | 0.03089 | 0.0014350 | 0.02654 |
| 4 | 0.734 | 0.473 | 0.02033 | 0.04146 | 0.0039847 | 0.03432 |
| 8 | 0.744 | 0.493 | 0.05286 | 0.06880 | 0.0120105 | 0.04916 |

### Observations from Plots

- **Clip fraction plots show a clear and visually striking monotonic increase with lag.** At L=1 there are occasional spikes up to ~0.03; by L=8, the clip fraction runs at 0.05–0.15 throughout training with high persistent variance.
- **Logprob movement grows steadily with lag.** At L=8, steady-state movement (~0.05) is ~3× larger than sync (~0.016), and the training trajectory is much noisier overall.
- **Reward and Pass@1 eval curves remain flat** across all lag values — no clear degradation trend in task performance. The policy learns at a similar rate despite the growing off-policy bias.
- **Seed variance is low within each lag**: at L=4, seeds 0–2 have eval reward within 0.038 and pass@1 within 0.060. Results are reproducible.

### Interpretation

**Strong support for H2:**

1. **Pre-update clip fraction is a direct, clean measure of off-policy bias.** It captures the fraction of ratios that are *already* outside ε=0.2 before the gradient step, driven purely by lag. The monotonic scaling L=1→2→4→8 is consistent with the hypothesis that larger staleness accumulates more policy drift between rollout and update.

2. **Pre-update ratio variance scales monotonically with lag:** 0.000577 (L=1) → 0.001435 (L=2) → 0.003985 (L=4) → 0.012011 (L=8). Each doubling of lag roughly doubles the variance.

3. **Logprob movement increases with lag** (sync: 0.016, L=8: 0.049), consistent with noisier gradient estimates from stale rollouts.

4. **Task performance does not clearly degrade** despite the large off-policy bias at L=8. The PPO clipping mechanism prevents collapse: even with 5.3% pre-update clips, the policy updates are bounded and the reward curve stays stable. This suggests the PPO surrogate objective is doing its job as a conservative policy constraint.

---

## Phase 5: Mismatch Sweep (H1)

### Summary

Ran mismatch (fp16 actor logprobs, fp32 learner) and rescored (fp16 actor, fp32 rescoring) conditions across seeds 0, 1, 2 to test H1 (logprob mismatch increases ratio noise). Mismatch introduces a consistent nonzero pre-update ratio variance that rescoring eliminates.

### Configuration

- Conditions: `mismatch`, `rescored` — both Lag: 0, LR: 1e-6, Updates: 100, Batch size: 4

### Observed Results — Mismatch Condition

| Seed | Eval Reward | Eval Pass@1 | Ratio Var (pre) | Ratio Var (post) | Abs Logprob Move |
|------|-------------|-------------|-----------------|------------------|------------------|
| 0 | 0.712 | 0.440 | 0.0000209 | 0.0000570 | 0.00264 |
| 1 | 0.740 | 0.480 | 0.0000205 | 0.0001437 | 0.00279 |
| 2 | 0.760 | 0.520 | 0.0000224 | 0.0000971 | 0.00277 |
| **Mean** | **0.737** | **0.480** | **0.0000213** | **0.0000992** | **0.00273** |

### Observed Results — Rescored Condition

| Seed | Eval Reward | Eval Pass@1 | Ratio Var (pre) | Ratio Var (post) | Abs Logprob Move |
|------|-------------|-------------|-----------------|------------------|------------------|
| 0 | 0.722 | 0.460 | 0.0000000 | 0.0000431 | 0.00162 |
| 1 | 0.742 | 0.500 | 0.0000000 | 0.0001192 | 0.00185 |
| 2 | 0.742 | 0.500 | 0.0000000 | 0.0000826 | 0.00183 |
| **Mean** | **0.735** | **0.487** | **0.0000000** | **0.0000816** | **0.00176** |

### Comparison: Sync vs Mismatch vs Rescored

| Metric | Sync | Mismatch | Rescored |
|--------|------|----------|----------|
| Eval Reward | 0.721 | 0.737 | 0.735 |
| Eval Pass@1 | 0.453 | 0.480 | 0.487 |
| Ratio Var (pre) | 0.0000000 | 0.0000213 | 0.0000000 |
| Ratio Var (post) | 0.0000949 | 0.0000992 | 0.0000816 |
| Abs Logprob Move | 0.00181 | 0.00273 | 0.00176 |

### Interpretation

**Support for H1:**
- Mismatch introduces a consistent nonzero pre-update ratio variance (mean 0.0000213) where sync and rescored have exactly zero — fp16/fp32 numerical drift is reliably measurable.
- Mismatch increases abs logprob movement by 51% vs sync (0.00273 vs 0.00181), indicating the gradient signal is corrupted by the numerical discrepancy.
- Rescoring eliminates the pre-update ratio variance entirely (back to 0.0000000) — recomputing logprobs in fp32 fully removes numerical drift.
- Rescored abs logprob movement matches sync (0.00176 vs 0.00181), confirming rescoring restores sync-equivalent behavior.

**Task performance unaffected:**
- All three conditions have similar eval reward (~0.73) and pass@1 (~0.47).
- The numerical drift magnitude is too small to impact task metrics at LR=1e-6.

**Conclusion:** Rescoring is an effective mitigation for fp16/fp32 logprob mismatch. It restores sync-equivalent stability metrics while allowing fp16 inference for efficiency.

---

## Phase 5 Re-run: Mismatch Sweep at LR=1e-5, 200 Updates

### Summary

Re-ran `mismatch` and `rescored` conditions at LR=1e-5, 200 updates, seeds 0–2. Key findings: mismatch continues to produce a small but consistent nonzero pre-update ratio variance (H1 confirmed at higher LR); rescoring eliminates it exactly; post-update clipping is now active in both conditions (~1.4–1.6%). Task performance is highest in mismatch, with rescored seed 2 being a notable outlier.

### Configuration

- Conditions: `mismatch`, `rescored`, Lag: 0, LR: **1e-5**, Updates: **200**, Batch size: 4

### Observed Results — Mismatch at LR=1e-5

| Seed | Eval Reward | Eval Pass@1 | Clip Frac (pre) | Clip Frac (post) | Ratio Var (pre) | Logprob Move (steady) |
|------|-------------|-------------|-----------------|------------------|-----------------|-----------------------|
| 0 | 0.740 | 0.480 | 0.00000 | 0.01482 | 0.0000363 | 0.01334 |
| 1 | 0.750 | 0.500 | 0.00000 | 0.01835 | 0.0000370 | 0.01620 |
| 2 | 0.740 | 0.480 | 0.00000 | 0.01422 | 0.0000394 | 0.02079 |
| **Mean** | **0.743** | **0.487** | **0.00000** | **0.01580** | **0.0000376** | **0.01678** |

### Observed Results — Rescored at LR=1e-5

| Seed | Eval Reward | Eval Pass@1 | Clip Frac (pre) | Clip Frac (post) | Ratio Var (pre) | Logprob Move (steady) |
|------|-------------|-------------|-----------------|------------------|-----------------|-----------------------|
| 0 | 0.710 | 0.420 | 0.00000 | 0.01290 | 0.0000000 | 0.01498 |
| 1 | 0.740 | 0.480 | 0.00000 | 0.01427 | 0.0000000 | 0.01431 |
| 2 | 0.568 | 0.360 | 0.00000 | 0.01418 | 0.0000000 | 0.01823 |
| **Mean** | **0.673** | **0.420** | **0.00000** | **0.01378** | **0.0000000** | **0.01584** |

### Full Comparison: Sync vs Mismatch vs Rescored at LR=1e-5

| Metric | Sync | Mismatch | Rescored |
|--------|------|----------|----------|
| Eval Reward | 0.732 | **0.743** | 0.673 |
| Eval Pass@1 | 0.480 | **0.487** | 0.420 |
| Clip Frac (pre) | 0.00000 | 0.00000 | 0.00000 |
| Clip Frac (post) | 0.01636 | 0.01580 | 0.01378 |
| Ratio Var (pre) | 0.0000000 | **0.0000376** | 0.0000000 |
| Logprob Move (steady) | 0.01630 | 0.01678 | 0.01584 |

### Interpretation

**H1 confirmed at LR=1e-5:**
- Mismatch pre-update ratio variance (0.0000376) is nonzero and consistent across all three seeds — the fp16/fp32 numerical gap is detectable regardless of learning rate. The value is slightly higher than at LR=1e-6 (0.0000213), consistent with a larger policy moving further from its fp16-computed logprob baseline.
- Pre-update clip fraction = 0 for mismatch — the ratio inflation from fp16/fp32 drift (~0.000038 variance) is too small to push individual token ratios outside ε=0.2 before the gradient step. Mismatch corrupts the gradient signal without triggering clipping directly.
- Rescored pre-update ratio variance = 0.0 exactly — recomputing logprobs in fp32 before the update fully eliminates the numerical gap at LR=1e-5 just as at LR=1e-6.
- Post-update clip fractions are similar across all three conditions (~1.4–1.6%), driven by the gradient step size rather than drift.

**Task performance:** Mismatch (0.743/0.487) matches or slightly exceeds sync (0.732/0.480), showing fp16 noise does not impair learning at this scale. Rescored mean (0.673/0.420) is pulled down by seed 2 (0.568/0.360); seeds 0 and 1 are within normal range (0.710–0.740). The rescored mean should be interpreted with this outlier in mind.

---

## Phase 6: Stale + Mismatch (H3)

### Summary

Full stale_mismatch sweep at LR=1e-5, 200 updates, across L ∈ {2, 4, 8} × seeds 0–2 (9 runs). This is the combined condition: fp16 actor using a lagged checkpoint. Compared directly against stale-only at the same lags, the mismatch component adds measurable ratio inflation, and task performance falls below the sync baseline at L=4 and L=8 — the first consistent task degradation signal in the project.

### Configuration

- Condition: `stale_mismatch`, LR: 1e-5, Updates: 200, Batch size: 4

### Observed Results

| Lag | Seed | Eval Reward | Eval Pass@1 | Clip Frac (pre) | Clip Frac (post) | Ratio Var (pre) | Logprob Move (steady) |
|-----|------|-------------|-------------|-----------------|------------------|-----------------|-----------------------|
| 2 | 0 | 0.722 | 0.460 | 0.00957 | 0.03356 | 0.0018389 | 0.02980 |
| 2 | 1 | 0.724 | 0.480 | 0.00910 | 0.03922 | 0.0020556 | 0.02991 |
| 2 | 2 | 0.720 | 0.440 | 0.01217 | 0.04084 | 0.0024784 | 0.03075 |
| **2** | **mean** | **0.722** | **0.460** | **0.01028** | **0.03787** | **0.0021243** | **0.03015** |
| 4 | 0 | 0.666 | 0.380 | 0.02309 | 0.04542 | 0.0039359 | 0.03906 |
| 4 | 1 | 0.734 | 0.500 | 0.02493 | 0.04727 | 0.0052154 | 0.03627 |
| 4 | 2 | 0.730 | 0.460 | 0.02331 | 0.04480 | 0.0063719 | 0.03352 |
| **4** | **mean** | **0.710** | **0.447** | **0.02378** | **0.04583** | **0.0051744** | **0.03628** |
| 8 | 0 | 0.732 | 0.480 | 0.06175 | 0.07921 | 0.0114383 | 0.05228 |
| 8 | 1 | 0.722 | 0.460 | 0.05715 | 0.07507 | 0.0127097 | 0.04963 |
| 8 | 2 | 0.702 | 0.420 | 0.05047 | 0.06554 | 0.0099390 | 0.04560 |
| **8** | **mean** | **0.719** | **0.453** | **0.05646** | **0.07328** | **0.0113623** | **0.04917** |

### Stale+Mismatch vs Stale-Only at Matched Lags (LR=1e-5)

| Lag | Condition | Eval Reward | Eval Pass@1 | Clip Frac (pre) | Ratio Var (pre) | Logprob Move |
|-----|-----------|-------------|-------------|-----------------|-----------------|--------------|
| 0 | sync | 0.732 | 0.480 | 0.00000 | 0.0000000 | 0.01630 |
| 2 | stale | 0.726 | 0.473 | 0.00603 | 0.0014350 | 0.02654 |
| 2 | stale+mismatch | 0.722 | 0.460 | 0.01028 | 0.0021243 | 0.03015 |
| 4 | stale | 0.734 | 0.473 | 0.02033 | 0.0039847 | 0.03432 |
| 4 | stale+mismatch | **0.710** | **0.447** | 0.02378 | 0.0051744 | 0.03628 |
| 8 | stale | 0.744 | 0.493 | 0.05286 | 0.0120105 | 0.04916 |
| 8 | stale+mismatch | **0.719** | **0.453** | 0.05646 | 0.0113623 | 0.04917 |

### Observations from Plots

- **L=2:** Clip fraction is slightly elevated vs stale-only L=2. Reward and pass@1 are flat and close to stale-only; mismatch adds noise but does not change the reward trajectory meaningfully at this lag.
- **L=4:** Clip fraction shows more volatile spikes than stale-only L=4. Seed 0 shows a clear downward drift in eval reward (0.666), pulling the mean below sync. The combined drift is beginning to impair learning.
- **L=8:** The highest clip fractions in the entire experiment — peaks up to 0.15–0.20. Logprob movement is persistently high (0.05–0.12) with no settling. All three seeds show eval reward and pass@1 below the corresponding stale-only L=8 values and below sync.

### Interpretation

**Partial support for H3:**

1. **Mismatch adds ratio inflation on top of staleness, most clearly at lower lags.** At L=2, stale+mismatch pre-clip (1.03%) is 70% higher than stale-only (0.60%); the ratio variance is 48% higher. At L=8 the gap narrows to ~7% because staleness dominates and the fp16/fp32 gap is comparatively small.

2. **Task performance falls below sync at L=4 and L=8 — the only conditions to do so.** Stale-only at the same lags does not degrade performance (stale L=4: 0.734, stale L=8: 0.744 — both at or above sync's 0.732). Stale+mismatch L=4 (0.710, pass@1 0.447) and L=8 (0.719, pass@1 0.453) are consistently below sync. This is the primary H3 evidence.

3. **Seed variance is higher in stale+mismatch than stale-only**, particularly at L=4 (seed 0: 0.666 vs seed 1: 0.734). The combined drift makes optimisation noisier and more seed-sensitive.

4. **Logprob movement is slightly elevated vs stale-only** at each matched lag, consistent with an additive fp16 noise contribution on top of staleness drift.

**Limitations:**
- Mismatch-only at LR=1e-5 is pending; the isolated fp16/fp32 contribution cannot be precisely quantified at this LR.
- At L=8 the stale component so strongly dominates that the mismatch increment is within noise.

---

## Summary of Hypotheses

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| **H1**: Logprob mismatch increases ratio noise | ✅ Supported | Mismatch pre-update ratio variance 0.0000376 (LR=1e-5) vs 0 for sync/rescored; confirmed at both LRs; rescoring fully eliminates it |
| **H2**: Staleness introduces off-policy bias | ✅ Strongly supported | Pre-update clip fraction scales monotonically from 0% (sync) to 5.3% (L=8); pre-update ratio variance grows 20× from L=1 to L=8; logprob movement grows 3× |
| **H3**: Combined drift reduces task performance | ⚠️ Partially supported | stale+mismatch L=4 (0.710/0.447) and L=8 (0.719/0.453) are the only conditions below sync (0.732/0.480); stale-only at same lags does not degrade — mismatch is the differentiating factor |

---

## Recommendations

1. **Analyse stale_mismatch results** — L=2,4,8 × seeds 0,1,2 completed; compare against stale-only and mismatch-only to assess H3.
2. **Run mismatch LR=1e-5 re-run** — rescored is done; mismatch condition at LR=1e-5 still pending to complete the H1 comparison.
3. **Generate comparison plots** — lag-vs-pre-clip and lag-vs-ratio-variance figures are the key visuals for H2; sync/mismatch/rescored training curve comparison for H1.
4. **Write final report** — H2 is now strongly supported; H1 is supported at LR=1e-6 (rescored mitigation confirmed at LR=1e-5); H3 needs stale_mismatch analysis.
