# Phase D: baseline-repair diagnostics (D1–D4)

**These are diagnostics of choices the paper does not specify. They are not optimization methods.**

Setup:
- Each run is C0 (paper Fourier + NTK, 6 × 200, mini-batch 32, lr 1e-4) with exactly one change.
- 5 000 steps = 250 epochs = 1.6e5 PDE evaluations.
- Seed 1234; 1 thread per run, 4 concurrent; about 6.3 min of training each.
- Sources: `tables/phaseD_oscillation.csv` (from `experiments/oscillation_report.py`) and the run records.

**Verdict rule:** OSCILLATORY means |ω_fit/ω_d − 1| < 0.10 **and** a temporal-amplitude ratio in [0.5, 2] **and** a max|u_t| ratio in [0.5, 2]. Each is measured on the model's own mid-span trace and field, not via L2.

## Results (exact-physics reference; L2_paper is identical to 3 s.f. at this error level)

| Run | Change from C0 | L2_paper / L2_exact | ω_fit/ω_d | amp. ratio | max\|u_t\| ratio | PDE res. | IC u | BC | Final λ (u, ut, ux) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | unit-normalised inputs | 1.78 / 1.78 | 0.00 | 0.21 | 0.08 | 1.96 | 0.28 | 0.25 | 1e9, 9e8, 4e11 | STATIC |
| D2 | zero biases | 2.49 / 2.49 | 0.01 | 1.91 | 0.72 | 1.4e3 | 0.028 | 0.44 | 4e15, 4e12, 5e14 | LOW-FREQ (moves, wrong ω) |
| D3 | unit inputs + zero biases | 1.86 / 1.86 | 0.14 | 1.36 | 0.45 | 19.5 | 0.019 | 0.22 | 1e12, 1e9, 6e11 | LOW-FREQ |
| D4 | reference-code convention + zero biases | 1.11 / 1.11 | −0.28 | 0.53 | 0.24 | 1.35 | 0.029 | 0.14 | 9e9, 1e7, 1e10 | LOW-FREQ |

For comparison, the A1 baselines at **matched 5k steps** (validation grid) had L2_exact of C0 3.26, Fourier only 2.25, C0-rc 1.13. All of them were STATIC at 20k steps (amplitude ratios 0.02–0.09).

## Answers

**A. Did normalization recover oscillation?** **No.** D1 (unit inputs) stays static: amplitude 0.21, |u_t| 0.08, fitted ω = 0.

**B. Did bias initialization recover oscillation?** **Partly, but not genuinely.** Zero biases (D2, D3, D4) remove the static collapse:
- the temporal amplitude is restored (ratio 0.5–1.9) and max|u_t| reaches 0.24–0.72 of the exact value;
- the IC error falls 5–10× (0.14 → 0.02–0.03).

But the motion is at the **wrong frequency** (ω_fit/ω_d = −0.28 to 0.14). No variant reaches the 20.6 Hz mode.

**C. Did NTK weighting remain pathological?** **Yes.** λ for the boundary/IC terms is 1e7–4e15 in every variant (λ_f = 1). D2's λ_u = 4e15 coincides with its PDE residual blowing up to 1.4e3.

**D. Did Fourier-only differ from Fourier + NTK?** Fourier only (A1, no NTK) is *also* static at 20k (amplitude 0.02, IC 0.74). It is less wrong than C0 at matched steps (2.25 vs 3.26 at 5k; 1.81 vs 3.26 at 20k).

**E. Is NTK itself responsible for the static attractor?** **Not solely.** The attractor appears with and without NTK under the paper-faithful soft-constraint formulation, so the common factor is the soft-constrained, physical-units 4th-order problem at 20.6 Hz. NTK does not rescue it, and it makes C0 worse than Fourier alone at matched budget. NTK is **implicated as an aggravating factor**, not as the sole cause.

**F. Strongest baseline.** No configuration is a credible baseline yet:
- Lowest L2: D4 (1.11). Closest to oscillatory behaviour: D2/D3.
- **The official paper-style baseline remains C0 as specified** (the paper-faithful resolution). The repairs did not demonstrably reproduce the paper's dynamics, so swapping in one of them would not make the baseline more faithful.

## Case classification

**CASE 3: all configurations remain non-oscillatory**, though bias initialization changes the failure mode from static to wrong-frequency. Long training of any soft-constrained baseline is not justified.

The conditioning diagnosis points at the **soft IC/BC formulation**. A static or slowly varying field satisfies the soft IC/BC terms at almost no cost, and NTK then inflates exactly those terms. That makes hard constraints the mechanistically targeted next step:
- With u(x,0) = u0(x) and u_t(x,0) = 0 enforced exactly, a static field u ≡ u0(x) has PDE residual c²u0'''' = ω²u0 ≠ 0.
- The static attractor is therefore excluded by construction: to reduce the only remaining loss, the network must create u_tt ≈ −ω²u0 near t = 0, which is an oscillation.

This is a hypothesis, to be tested in Phase E.
