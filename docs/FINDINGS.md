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
