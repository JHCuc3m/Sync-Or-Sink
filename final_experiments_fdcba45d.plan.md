---
name: Final Experiments
overview: Prioritize experiments that convert the current milestone’s static drift evidence into defensible live PPO results without overcommitting to a full distributed/vLLM system. The core deliverable should be a clean, reproducible PPO/RLVR harness plus small sweeps for staleness, mismatch, and one mitigation.
todos:
  - id: harness
    content: Build a single live PPO/RLVR harness with CLI flags for condition, lag, seed, updates, task, and output directory.
    status: pending
  - id: task
    content: Implement a deterministic structured-generation reward task, preferably JSON extraction or regex-constrained formatting.
    status: pending
  - id: staleness
    content: Run a small live PPO staleness sweep for L = 0, 1, 2, 4 and collect PPO stability metrics plus pass@1.
    status: pending
  - id: mismatch
    content: Run fp16 actor-logprob mismatch and fp32 learner-rescoring mitigation conditions against the sync baseline.
    status: pending
  - id: replicate
    content: Repeat the core conditions over at least three seeds if compute allows, prioritizing seeds over extra variants.
    status: pending
  - id: report
    content: Frame the final report around static proxy evidence leading into live PPO validation and rescoring mitigation.
    status: pending
isProject: false
---

# Final Experiment Plan

## Read Of The Current State

The project goal is clear and good: study whether actor-learner drift in RLHF breaks PPO stability, especially through policy staleness and actor/learner logprob mismatch.

Current implemented repo surface appears to be:

- `[scripts/dpo_baseline.py](scripts/dpo_baseline.py)`: DPO + LoRA baseline on HH-RLHF.
- `[scripts/logprob_parity.py](scripts/logprob_parity.py)`: static fp32 vs fp16 GPT-2 logprob mismatch.
- `[scripts/staleness_test.py](scripts/staleness_test.py)`: static staleness proxy using SFT checkpoints.
- `[jobs/](jobs/)`: three PACE jobs for those experiments.

Important caveat: `[extra_information/DL_Project.pdf](extra_information/DL_Project.pdf)` describes a live PPO prototype, but the checked-in repo currently does not show a live PPO script or actor-learner loop. So the next phase should be scoped as building and validating that missing bridge, not jumping straight to full vLLM integration.

## Recommended Core Direction

The highest-value next step is:

**Build one clean live PPO/RLVR experiment harness, then run a small staleness sweep and a mismatch-versus-rescoring comparison.**

This is realistic because it extends the current GPT-2-scale setup and PACE job pattern. It is experimentally meaningful because it tests whether the static ratio corruption observed in the milestone actually appears during training.

Avoid making vLLM the required core result. Treat true vLLM-vs-HF mismatch as a stretch goal if the live PPO harness stabilizes early.

## Proposed Experiment Scope

### Phase 1: Live PPO Baseline

Create a single configurable script, likely `[scripts/live_ppo_rlvr.py](scripts/live_ppo_rlvr.py)`, using GPT-2 and a deterministic verifiable reward task.

Recommended task: structured JSON extraction or regex-constrained generation, not arithmetic as the main final task. Arithmetic is easy to score but GPT-2 may not learn it cleanly, which can blur the stability story. JSON/format compliance is more language-model-like and gives interpretable pass@1.

Track at minimum:

- reward mean
- pass@1 or exact success rate
- PPO clip fraction
- KL proxy
- entropy
- ratio mean/variance
- advantage mean/std

### Phase 2: Staleness Sweep

Run live PPO with lag values:

- `L = 0, 1, 2, 4`
- optionally `L = 8` if runs are stable and cheap

Keep this single-process at first: simulate actor staleness by using older policy snapshots or rollout logprobs from delayed copies. A true distributed actor/learner implementation is not needed to answer the research question for this class project.

Main claim to test:

- Larger lag should increase clip fraction, KL proxy, and ratio variance.
- Reward/pass@1 may degrade, but the stability metrics are the cleaner primary result.

### Phase 3: Mismatch And Rescoring

Run three conditions:

- sync/control: consistent fp32 logprobs
- mismatch: actor-side fp16 logprobs used in PPO ratios
- rescored: actor generates responses, but learner recomputes old logprobs in fp32 before PPO update

This is the cleanest mitigation because it directly targets the fake ratio noise from actor/learner numerical mismatch. If rescoring reduces clip fraction or stabilizes reward, that gives a strong final-result section.

### Phase 4: Replication, Not Breadth

After the harness works, prefer a few seeds over more experimental variants.

Minimum realistic final matrix:

- sync, stale L=2, stale L=4
- mismatch, mismatch + rescoring
- seeds `0, 1, 2` if compute allows

This is better than running many one-off conditions with only one seed.

## Stretch Goals

Only pursue these after the core live PPO results exist:

- Add `stale + mismatch` combined condition for H3.
- Try a second verifiable task if JSON is too easy or too noisy.
- Add vLLM actor logprob comparison as a static Level-C parity experiment before attempting full vLLM actor rollouts.
- Add ratio capping or adaptive clipping after rescoring; these are less clean conceptually and should not displace rescoring.

## Repo And PACE Cleanup Needed

Before running sweeps, improve reproducibility just enough:

- Add unique output names or output directories per condition/seed so runs do not overwrite `[outputs/logs/](outputs/logs/)` and `[outputs/figures/](outputs/figures/)`.
- Add one new PACE job template under `[jobs/](jobs/)` for the live PPO harness.
- Avoid hardcoded cluster paths in job scripts if possible; use `$SLURM_SUBMIT_DIR` or document the expected path clearly.
- Add CLI flags for condition, task, lag, seed, updates, and output directory.

## Final Report Framing

The final report should be honest and strong:

- The milestone established static evidence for staleness and numerical logprob drift.
- The final project tests whether those drift sources matter in live PPO training.
- The key result is not necessarily huge pass@1 gains; it is whether PPO stability metrics degrade predictably and whether learner-side rescoring recovers stability.

This makes the project feel like a controlled ML-systems study rather than an unfinished attempt at a full production RLHF stack.