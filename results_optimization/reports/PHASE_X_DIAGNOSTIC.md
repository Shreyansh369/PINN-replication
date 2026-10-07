# Phase X: conditioning and representation diagnostics (Mode 1, 5 000 steps)

**Outcome:**
- **X1/X2 → CASE 4.** The tanh²(ω₁t) time factor improved conditioning substantially but does **not** pass the oscillation gate.
- **X3/X4 → representation is adequate.** The same network fits the 20.6 Hz solution's frequency when optimization is made easy.
- Together: **the dominant remaining bottleneck is optimization of the physics loss, not Fourier representation.** STOPPED for approval.

## Exact configurations (`configs/phaseX/*.json`; diff against parent printed at creation)

| ID | Parent | Only change(s) | Role |
|---|---|---|---|
| X1 | E4 (hard, paper convention) | `hard_constraints: ff_tsq → ff_tanh2` | method candidate |
| X2 | E4b (hard, D4 convention) | `hard_constraints: ff_tsq → ff_tanh2` | method candidate |
| X3 | C0 network, paper convention | `grouping: data_only`, `weighting: fixed` (no PDE/IC/BC/NTK/RAD) | **diagnostic only** |
| X4 | C0 network, D4 convention | as X3, plus the D4 convention | **diagnostic only** |

ω₁ = (β₁l/L)²·√c² = 129.37 rad/s. It comes from the Eq. 49 coefficient and the fixed-fixed eigenproblem only (`Benchmark.fundamental_omega`). It was not tuned and not derived from the solution.

All other settings are identical to E: 6 × 200, m = 100, Adam lr 1e-4 constant, mini-batch 32, 640 points per epoch, 5 000 steps, seed 1234, 1 thread per run, 4 concurrent. Tests: 102 pass, including exact IC/BC satisfaction with tanh² for Modes 1 and 2, and unchanged E4/E4b run keys.

## Results (`tables/phaseX_oscillation.csv`, `figures/phaseX_midspan_traces.png`, `figures/phaseX_loss_history.png`)

| | X1 hard tanh² | X2 hard tanh², D4 conv. | X3 supervised (diag.) | X4 supervised (diag.) | *E4 (ref.)* | *E4b (ref.)* |
|---|---|---|---|---|---|---|
| L2_paper / L2_exact | 1.50 / 1.50 | **0.909 / 0.909** | 0.463 / 0.463 | 0.316 / 0.316 | 2.83 | 1.61 |
| best-so-far L2_exact | 1.50 | 0.907 | 0.114 | 0.146 | 2.82 | 1.60 |
| fitted ω [rad/s] (exact 129.32) | fit failed (no oscillation) | 112.2 | **129.25** | **129.37** | 0.07 | 10.8 |
| frequency ratio | — | 0.868 | 0.999 | 1.000 | 0.001 | 0.083 |
| fitted decay rate [1/s] (exact 3.54) | — | 31.1 | 3.50 | 3.64 | — | — |
| amplitude ratio | 0.19 | 0.38 | 1.01 | 1.03 | 1.26 | 1.14 |
| max \|u_t\| ratio | 0.54 | 0.71 | 1.06 | 0.97 | 0.02 | 0.11 |
| IC error, u (× A₀) | 4.8e-6 | 4.8e-6 | 0.16 | 0.19 | 4.8e-6 | 4.8e-6 |
| BC error | 1.4e-5 | 1.4e-5 | 2.96 | 1.21 | 1.4e-5 | 1.4e-5 |
| PDE residual, RMS(r)/RMS(u_tt) | 3.64 | **0.215** | 2.6e3 (not trained) | 13.8 (not trained) | 4.31 | 1.28 |
| training wall-clock | 297 s | 296 s | 23 s | 23 s | 295 s | 296 s |
| PDE evaluations | 1.6e5 | 1.6e5 | 0 | 0 | 1.6e5 | 1.6e5 |
| parameters / training peak RSS | 241 601 / 0.82 GB | 241 601 / 0.82 GB | 241 601 / 0.79 GB | 241 601 / 0.79 GB | same | same |
| **Oscillation verdict** | NO | **NO (improved)** | YES (diagnostic) | YES (diagnostic) | NO | NO |

Traces:
- **X2**: follows the exact solution through the first half-period (trough at ≈ 0.028 s vs 0.024 s), then **over-damps**, with the amplitude gone by ≈ 0.1 s.
- **X1**: drops within about 0.01 s to a constant of about 25 mm.
- **X3/X4**: oscillate at the correct frequency with a slowly drifting offset.

## Conditioning diagnostics (`tables/phaseX_conditioning.csv`)

| | time factor | required \|N*\| max / median | network \|N\| median | \|N\| near t = 0 (t < 0.01 s) | ‖∇θ L_pde‖, early (t < 0.02 s) | late (t > 0.5 s) | early/late |
|---|---|---|---|---|---|---|---|
| E4 | (t/T)² | **8 270** / 3.8 | 1.13 | 1.14 | 7.2e3 | 1.5e8 | **5e-5** |
| E4b | (t/T)² | 8 270 / 3.8 | 3.9 | 15.3 | 7.7e3 | 6.9e5 | 1e-2 |
| X1 | tanh²(ω₁t) | **1.89** / 0.97 | 0.69 | 0.69 | 9.1e7 | 1.6e8 | **0.6** |
| X2 | tanh²(ω₁t) | 1.89 / 0.97 | 0.96 | 0.53 | 8.4e5 | 9.8e4 | **8.5** |

**The conditioning hypothesis is supported, within these runs:**
- The required network output range shrinks from about 8 300 to below 2.
- The early-time PDE gradient goes from 2×10⁴ times weaker than the late-time gradient (E4) to comparable or dominant (X1, X2).
- Against the same-convention parents: L2 2.83 → 1.50 (X1) and 1.61 → 0.909 (X2); PDE residual 1.28 → 0.215 (X2).
- **This is not sufficient for the gate:**
  - X2's frequency is 13 % low and it is strongly over-damped.
  - X1's inner network collapsed to an almost constant output (|N| = 0.69 ± 0.01 everywhere), so it cannot represent the decaying oscillation at all.

## Representation-capacity interpretation (X3/X4)

When the solution is given as data, the identical network finds the correct frequency (ω ratio 0.999–1.000), damping (3.50–3.64 vs 3.54), amplitude and velocity within 5 000 steps, under **both** conventions.

**The Fourier representation is therefore not the limiting factor for Mode 1** (CASE 2 information).

The supervised L2, however, plateaus at 0.11–0.46 with a noisy loss (`phaseX_loss_history.png`). So even an easy regression is not driven to high accuracy by **Adam at lr 1e-4 with mini-batch 32** within 5k steps. That makes the optimizer/batch setting a measured, independent limiter on accuracy.

## Recommendation: next single controlled experiment (requires approval)

The bottleneck is **optimization of the physics loss**. The cleanest single change, supported by the evidence above:

**Y-batch: one change, the learning-rate schedule.** Use the reference code's schedule (lr₀ = 1e-3, decayed ×0.9 every 1 000 steps) instead of the paper's constant 1e-4. Everything else is unchanged. Four concurrent 5k-step runs, about 6 min:

| ID | Config | Purpose |
|---|---|---|
| Y1 | X2 + reference-code lr schedule | Does better optimization turn X2's correct early dynamics into a sustained 20.6 Hz oscillation? |
| Y2 | X1 + same schedule | Does it free the paper-convention network from its constant-output collapse? |
| Y3 | X4 + same schedule (supervised, **diagnostic**) | Control: does the schedule unlock accuracy where representation is known to be adequate? |
| Y4 | C0 (paper Fourier + NTK) + same schedule | Fairness: does the paper baseline benefit equally from the same optimizer change? |

Decision:
- If Y1 (or Y2) passes the oscillation gate, it becomes the Mode-1 candidate for a 20k confirmation, with Y4 as the matched paper-style baseline.
- If not, the next lever is mini-batch size, as its own single-change batch.

**Timeline (it is 21:58 UTC).**
- Y batch: about 10 min including analysis.
- If it succeeds, Mode-1 20k (2 seeds) and Mode-2 (frozen method vs C0) can run concurrently: about 30–35 min.
- Leaves ample time before the 60–90 min freeze window.
- If no candidate passes the gate after Y, the presentation should report the diagnostic findings (static attractor, ansatz conditioning, representation adequacy) as the validated contribution, and not claim an optimized method.
