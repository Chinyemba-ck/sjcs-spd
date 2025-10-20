# Training Monitoring Guide
## SmolLM2 Attention C=2000 with Scaled Sparsity

This guide documents what to monitor during training to detect if the previous failure pattern is recurring.

## Previous Failure Pattern (lyric-lion-9)

**Timeline:**
- Steps 0-1500: Healthy training
- Step 1600: Phase transition - gradient spike (0.036 → 0.065)
- Step 4000: Catastrophic gradient spike (0.224)
- Steps 4000+: Sustained instability and performance degradation

**Root Cause:** Insufficient sparsity regularization (0.0003) for 24,000 components

**Fix Applied:** Scaled `importance_minimality_coeff` from 0.0003 to 0.0008

---

## Critical Metrics to Monitor

### 1. **Gradient Norm** (`train/misc/grad_norm`)

**Healthy Range:**
- Steps 0-1500: < 0.04
- Steps 1500+: < 0.05
- Throughout: Should NOT spike above 0.08

**Warning Signs:**
- Single spike > 0.08 → Monitor closely
- Sustained > 0.05 → Warning
- Any value > 0.10 → CRITICAL - Consider aborting

**Previous Failure:**
- Step 1600: 0.065 (1.79× jump)
- Step 4000: 0.224 (10× baseline)

**Action:** If gradient norm exceeds 0.10 for 3+ consecutive eval steps → ABORT

---

### 2. **KL Divergence Gap** (`eval/kl/unmasked - eval/kl/ci_masked`)

**Healthy Pattern:**
- Gap should be NEGATIVE (unmasked < ci_masked)
- This means components are HELPING

**Warning Signs:**
- Gap approaching zero → Components becoming less useful
- Gap becomes POSITIVE → Components are HARMFUL

**Previous Failure:**
- Step 3500: Gap reversed to +0.001 (first sign)
- Step 4000: +0.004 (confirmed harmful)
- Step 13000: +0.060 (56% degradation)

**Action:** If gap becomes positive → HIGH RISK, monitor other metrics

---

### 3. **Eval KL (Absolute)** (`eval/kl/unmasked`)

**Healthy Pattern:**
- Should DECREASE over time
- Target: < 0.06 by step 13000 (MLP achieved 0.061)

**Warning Signs:**
- Plateauing while still > 0.08
- Increasing after initial decrease
- Single-step jump > 0.02

**Previous Failure:**
- Best: 0.060 at step 1000
- Degraded to: 0.166 at step 13000 (2.76× worse)

**Comparison to MLP:**
- MLP at step 1000: 0.120
- MLP at step 13000: 0.061 (improved)
- Attention should follow similar trajectory

---

### 4. **Component Survival** (`eval/ci_l0/total` or `eval/l0_0.001/total`)

**Healthy Pattern:**
- Step 1000: > 100 components alive (> 0.4%)
- Step 5000: > 50 components alive (> 0.2%)
- Final: > 30 components alive (> 0.125%)

**Warning Signs:**
- Faster death than MLP (adjusted for 2.67× more components)
- < 2% alive before step 10000

**Previous Failure:**
- Step 1000: 114.5 (0.48%) - OK
- Step 5000: 50.9 (0.21%) - Borderline
- Step 13000: 30.0 (0.12%) - Low

**Expected with Fix:**
- Higher coefficient → More sparsity pressure
- May see MORE death initially, but healthier surviving components
- Quality > Quantity

---

### 5. **Training Loss Components**

**Monitor ratios:**
```
weighted_recon = stochastic_recon × 0.2 + stochastic_recon_layerwise × 2.0
weighted_sparse = importance_minimality × 0.0008
```

**Healthy Pattern:**
- `weighted_sparse` should be HIGHER than previous run (due to 0.0008 vs 0.0003)
- This increased pressure is the fix

**Previous Failure (step 5000):**
- weighted_sparse = 15.98 × 0.0003 = 0.00480

**Expected with Fix (step 5000):**
- weighted_sparse = ~16 × 0.0008 = 0.01280 (2.67× higher) ✓

---

## Monitoring Schedule

**Every 500 steps (via WandB):**
1. Check gradient norm (should be < 0.05)
2. Check KL gap (should be negative)
3. Check eval KL trend (should be decreasing)

**Critical Checkpoints:**

**Step 1600** (previous failure point):
- Gradient norm: MUST be < 0.05 (previous: 0.065)
- KL gap: MUST be negative (previous: still negative, but close)
- If healthy here → Good sign

**Step 4000** (previous catastrophic point):
- Gradient norm: MUST be < 0.05 (previous: 0.224)
- KL gap: MUST be negative (previous: +0.004)
- If healthy here → Very good sign

**Step 13000** (end of previous run):
- Eval KL: Should be < 0.08 (previous: 0.166)
- Compare to MLP: 0.061 at step 13000

---

## Early Abort Criteria

**ABORT IMMEDIATELY if:**
1. Gradient norm > 0.15 for any single step
2. KL gap becomes positive AND gradient norm > 0.08
3. Single-step KL jump > 0.03
4. Eval KL > 0.15 after step 5000

**CONSIDER ABORTING if:**
1. Gradient norm > 0.08 sustained for 5+ evals
2. KL gap positive for 3+ consecutive evals
3. Eval KL not improving after step 10000

---

## Success Criteria

**Training is SUCCESSFUL if:**
1. Gradient norms stay < 0.05 throughout
2. KL gap remains negative throughout
3. Final eval KL < 0.10 (ideally < 0.08)
4. Performance comparable to or better than MLP

**This would validate:**
- Our root cause diagnosis (insufficient regularization)
- Our fix (scaled coefficient)
- Our understanding of the failure mechanism

---

## WandB Dashboard

Project: `smollm2-spd`
Run prefix: `smollm2_135m_3layer_attn_fixed`

**Key plots to watch:**
- `train/misc/grad_norm` (line chart, check for spikes)
- `eval/kl/unmasked` vs `eval/kl/ci_masked` (overlay, gap should be negative)
- `eval/l0_0.001/total` (component survival over time)

**Compare to:**
- Failed run: lyric-lion-9 (C=2000, coeff=0.0003)
- Successful run: balmy-cloud-8 (MLP, C=1000, coeff=0.0003)

---

## Notes

- Training will take ~12-16 hours on 2× H100
- Cost: ~$65-85 total
- Check WandB every few hours during work hours
- Set up WandB alerts for gradient norm > 0.10

**Remember:** We're testing a hypothesis. If it fails, we learn something valuable about SPD optimization dynamics.
