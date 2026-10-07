# Phase E screen: new-method candidates on Mode 1 (FE-D-M1), 5 000 steps

**Outcome: CASE C. None of the four candidates recovers a genuine oscillation. STOPPED.**

This is recorded as a negative result, together with a mechanistic diagnosis and one proposed next experiment.

## Exact configurations (`configs/phaseE/*.json`)

Shared settings:
- Physics, references and evaluation are identical to D/C0: Eq. 49 coefficients, fixed-fixed BCs, exact-root mode-1 IC (A₀ = 0.08 m), t ∈ [0, 1] s, 201 × 2001 evaluation grid.
- 6 × 200 tanh trunk, m = 100 Fourier features, σ_x = (1), σ_t = (10, 1).
- Adam, constant lr 1e-4; mini-batch 32 from 640 points per term redrawn each epoch.
- 5 000 steps; seed 1234; 1 thread per run, 4 runs concurrent.

| ID | Method | Differs from its parent by |
|---|---|---|
| E3 | Fourier + RAD | B-B (Fourier, paper convention, λ = 1) + RAD: k = 1, c = 1, 5 000 candidates every 500 steps |
| E4 | Hard-constrained Fourier | C0's network + ansatz `u = u0 + (t/T)²·Φ·A0·N`; PDE loss only, no NTK (single term) |
| E5 | Hard-constrained Fourier + RAD | E4 + RAD (same RAD settings as E3) |
| E4b | Hard-constrained Fourier, D4 convention | E4 with no 2π, standardised inputs, zero biases |

## Results (`tables/phaseE_oscillation.csv`, `figures/phaseE_midspan_traces.png`)

| | E3 Fourier+RAD | E4 Hard | E5 Hard+RAD | E4b Hard (D4 conv.) |
|---|---|---|---|---|
| L2_paper / L2_exact | 2.21 / 2.21 | 2.83 / 2.83 | 2.84 / 2.84 | 1.61 / 1.61 |
| fitted ω [rad/s] (exact 129.3) / ratio | 6.9 / 0.053 | 0.07 / 0.001 | 0.07 / 0.001 | 10.8 / 0.083 |
| mid-span amplitude ratio | 0.19 | 1.26 | 1.24 | 1.14 |
| max \|u_t\| ratio | 0.06 | 0.02 | 0.02 | 0.11 |
| IC error, u (× A₀) | 0.66 | **4.8e-6** | **4.8e-6** | **4.8e-6** |
| BC error (× A₀ / L) | 0.46 | **1.4e-5** | **1.4e-5** | **1.4e-5** |
| PDE residual, RMS(r)/RMS(u_tt) | 52.6 | 4.31 | 3.78 | 1.28 |
| L2 slope (log-log, steps 2.5k–5k) | −1.97 (from 13 to 2.2) | −0.20 | −0.17 | −0.13 |
| PDE-loss trend | still falling | **plateau from 2.5k** | slowly falling | **plateau from 2.5k** |
| wall-clock (training) | 344 s | 295 s | 330 s | 296 s |
| of which RAD | 26 s (+8 %) | — | 32 s (+11 %) | — |
| ms/step | 69 | 59 | 66 | 59 |
| PDE evaluations | 1.6e5 | 1.6e5 | 1.6e5 | 1.6e5 |
| candidate evaluations | 4.5e4 | 0 | 4.5e4 | 0 |
| parameters | 241 601 | 241 601 | 241 601 | 241 601 |
| peak RSS (training) | 1.28 GB | 0.82 GB | 1.36 GB | 0.82 GB |
| **Verdict** | **STATIC/LOW-FREQ** | **LOW-FREQ (slow decay)** | **LOW-FREQ (slow decay)** | **LOW-FREQ** |

For reference, the paper baseline C0 (Fourier + NTK) runs at 80 ms/step (67 ms + 13 ms NTK) and had L2 = 3.26 at 5k steps, static.

Traces:
- **E4/E5**: a single slow decay from 80 mm to about 0 over 1 s (≈ 0.5 Hz).
- **E4b**: decays within about 0.2 s, then low-frequency wiggles.
- **E3**: a near-constant offset of about 25 mm with small ripples.

None oscillates at 20.6 Hz.

## Interpretation: which bottleneck?

**(iii) Hard-constraint formulation: identified as a numerical-conditioning problem (measured, no training).**
- The ansatz is exact: IC/BC errors are 1e-5 to 1e-6, and the static attractor of the soft formulation is gone (the field must leave u₀).
- But the network must represent `N* = (u_exact − u0)/((t/T)² Φ A0)`. At mid-span that is `(q(t) − 1)/t²`, which tends to **−ω²/2 = −8 369** as t → 0 and is about −1 for t > 0.1 s (`profiles/phaseE_ansatz_conditioning.txt`).
- So a 4-decade output range is required, concentrated exactly where the t² factor suppresses the network's gradient signal. A tanh network initialised at O(1) cannot reach it in 5k steps at lr 1e-4. It settles instead on the O(1) slow decay seen in the traces, and the PDE loss plateaus.
- With the time factor scaled to the beam's fundamental, `g(t) = tanh²(ω₁t)`, the required N* is **[−1.9, −0.16]** for Mode 1 and **[−3.8, −0.06]** for Mode 2. Here ω₁ = 129.4 rad/s comes from the PDE coefficients and the BC eigenproblem only, not from the solution, so the same frozen choice serves Mode 2. It is O(1) in both modes and is still exact for all six IC/BC conditions (tanh² ~ (ω₁t)² near 0, so u_t(x,0) = 0 holds).

**(iv) RAD behaviour: no effect at this stage (measured).**
- The selected points stay essentially uniform in t and x (`figures/phaseE_rad_distribution.png`): with a diffuse, large residual, p ∝ |r|/mean + 1 is at most about 2:1.
- E5 vs E4: L2 2.84 vs 2.83, at +11 % cost.
- RAD cannot help before the model is in the right regime. It is deprioritised until a candidate oscillates; RAD is **not** shown to help.

**(i) Temporal spectral representation: not yet separated.**
- E4b (no feature reaches 129 rad/s) moved faster than E4 (3 features do), so feature coverage alone does not explain the ranking.
- A supervised fit of the exact solution with the same network would separate representation from PDE optimization.

**(ii) Optimization:** lr 1e-4 at mini-batch 32 is the paper's setting. At 5k steps (0.6 % of the paper budget) slow progress is expected, but a PDE-loss **plateau** points to the conditioning in (iii) rather than budget alone.

**(v) Implementation:** none found. The ansatz, the residuals and RAD all pass the 96 tests; IC/BC are exact as derived.

## Recommendation for the NEXT experiment (requires approval)

One batch: four 5 000-step Mode-1 runs, 1 thread each, concurrent, about 6 min. No other changes to settings.

| ID | Change | Question answered |
|---|---|---|
| X1 | E4 with `g(t) = tanh²(ω₁ t)` (ω₁ from the PDE coefficients + BC eigenproblem; declared **problem-specific prior**) | Does fixing the ansatz conditioning (iii) produce a genuine 20.6 Hz oscillation? |
| X2 | X1 with the D4 convention | The same, under the convention that moved fastest in E4b |
| X3 | Supervised regression of the exact solution (data loss only, no PDE), paper convention, same network and budget | (i) Can this Fourier network *represent* 20.6 Hz within 5k steps? |
| X4 | X3 with the D4 convention | (i) for the other convention |

X3/X4 are **diagnostics only** (they use the solution as data) and are never presented as a method.

Decision rule:
- If X1 or X2 oscillates, it becomes the candidate for 20k Mode-1 confirmation (with your approval), and RAD is re-tested on top of it.
- If X3/X4 cannot fit the solution either, the bottleneck is representation (σ_t / feature scaling). That would be a separately declared experiment, not a silent change.

The X-series needs one small code addition: a `g(t)` option for the hard ansatz, plus a data-only diagnostic loss. Both come with tests and do not change any existing code path.
