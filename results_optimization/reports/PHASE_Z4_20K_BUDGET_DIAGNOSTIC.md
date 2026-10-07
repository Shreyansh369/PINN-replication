# Z4-20K: final budget-scaling diagnostic (Mode 1)

**Outcome: CASE B.** More physics computation keeps improving persistence: from 3.2 to 8.1 cycles; L2_exact 0.526 → 0.263; PDE residual 0.110 → 0.041. The oscillation still collapses at **0.393 s (8.1 of 20.6 cycles)**, and the gain in the last block was about half that of the previous two blocks.

**Full-window reproduction is not achieved. There is no validated optimized PINN.** STOPPED.

## Exact configuration

`configs/phaseZ/Z4_20K.json`; run `Z4_20K__s1234__fa8fa7fe7e`.
- Z4 recipe: hard-constrained Fourier PINN, tanh²(ω₁t) with ω₁ = 129.37 rad/s from the PDE coefficient and the fixed-fixed eigenproblem, D4 convention, 6 × 200 tanh, m = 100, σ_x = (1), σ_t = (10, 1), 241 601 parameters.
- Training: Adam, lr = 1e-3 · 0.9^(step/1000); **mini-batch 128**; PDE loss only; no RAD, NTK or extra weights; seed 1234; float32; 1 thread.
- **Only change from Z4: `train.max_steps` 5 000 → 20 000.** The automatic diff against the recorded Z4 config shows no other result-affecting field. The recorded Z4 run ID reproduces.
- **Bit-level check:** the 5k checkpoint equals the original Z4 run (L2 0.526, PDE residual 0.110, collapse 0.153 s).

Collapse metric: unchanged from the Y1-20K report. Local amplitude in a one-period sliding window; collapse when the predicted/exact ratio is < 0.5; threshold not altered.

## Checkpoints (`tables/phaseZ4_20K_checkpoints.csv`)

| Checkpoint | L2_exact (= L2_paper) | PDE residual | Frequency [rad/s] (exact 129.32) | Decay [1/s] (exact 3.54) | Amp. ratio | max \|u_t\| ratio | IC / BC error | **Collapse time** | **Cycles** (displ. / vel.) | Train time (cum.) | PDE evals (cum.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5k | 0.526 | 0.110 | 128.1 (−0.9 %) | 8.98 | 0.69 | 0.93 | 4.8e-6 / 1.4e-5 | 0.153 s | 3.2 / 3.4 | 593 s | 6.4e5 |
| 10k | 0.389 | 0.109 | 129.3 (−0.02 %) | 6.71 | 0.76 | 0.96 | same | 0.249 s | 5.1 / 4.9 | 1 181 s | 1.28e6 |
| 15k | 0.290 | 0.043 | 129.1 (−0.2 %) | 5.73 | 0.87 | 1.01 | same | 0.346 s | 7.1 / 7.3 | 1 739 s | 1.92e6 |
| 20k | **0.263** | **0.041** | 129.2 (−0.1 %) | 5.40 | 0.89 | 1.01 | same | **0.393 s** | **8.1 / 7.9** | 2 290 s | 2.56e6 |

Collapse-time increments per 6.4e5 PDE evaluations: **+0.096 s, +0.097 s, +0.047 s**. The front keeps moving, but the last block advanced about half as far as the previous two.

The LR schedule had decayed to 1.2e-4 by 20k steps (from 1e-3). This is a possible contributor to the slowdown; it was not tested and is not claimed.

## Budget comparison

| Method | Steps | Batch | PDE evals | L2_exact | PDE residual | Collapse time | Cycles | Train time | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Y1 | 5k | 32 | 1.6e5 | 0.774 | 0.178 | 0.057 s | 1.2 | 300 s | FAIL |
| Y1 | 20k | 32 | 6.4e5 | 0.550 | 0.108 | 0.151 s | 3.1 | 1 078 s | FAIL |
| Z4 | 5k | 128 | 6.4e5 | 0.526 | 0.110 | 0.153 s | 3.2 | 611 s | FAIL |
| **Z4-20k** | 20k | 128 | **2.56e6** | **0.263** | **0.041** | **0.393 s** | **8.1** | 2 290 s | **FAIL** (full window not reproduced) |

The paper-style baseline at a comparable budget is C0 (paper Fourier+NTK) at 20k steps × 32 = 6.4e5 PDE evaluations: L2 3.26, static (0 cycles), 1 575 s.

## What this establishes, kept separate

**1. Evidence that compute helps: established within the tested range.**
- Across 1.6e5 → 2.56e6 PDE evaluations, all of the following improve monotonically along each recipe: collapse time, L2_exact, frequency accuracy (to −0.1 %), amplitude (0.89) and velocity (1.01) ratios, and the fitted decay rate (8.98 → 5.40).
- The rate of improvement in persistence is not constant, and the last block slowed.

**2. Evidence that batch size helps: NOT established as an effect on the solution.**
- At equal PDE evaluations (6.4e5), mini-batch 128 (Z4-5k) and mini-batch 32 (Y1-20k) reach practically the same state: 3.2 vs 3.1 cycles, L2 0.526 vs 0.550, PDE residual 0.110 vs 0.108.
- What is established is **wall-clock efficiency**: the large batch got there in 0.57× the time on this CPU.
- Z4-20k's further gains are attributable to more computation; there is no matched small-batch run at 2.56e6 evaluations.

**3. Evidence that the candidate reproduces the physical solution: NO.**
- It covers 8.1 of 20.6 cycles, then collapses.
- It remains over-damped (fitted decay 1.5× physical).
- L2_exact 0.263 is about 570× the paper's 4.64e-4.
- Frequency, amplitude and velocity in the reproduced cycles are physically correct (frequency within 0.1 %), but the gate requires the full window.

This experiment does not establish whether more computation would eventually cover the full window. The decelerating last increment argues against assuming it, and **no extrapolation is claimed**.

## Figures

- `phaseZ4_20K_disp_traces.png` and `phaseZ4_20K_vel_traces.png`: exact vs 5k/10k/15k/20k, collapse marked.
- `budget_collapse_vs_pde_evals.png`: collapse time vs cumulative PDE evaluations for both recipes.
- `budget_L2_vs_pde_evals.png`, `budget_pde_residual_vs_pde_evals.png`.
- `phaseZ4_20K_loss_and_residual.png`: training (PDE) loss and validation L2 history.

## Status for the presentation (per instruction: no automatic continuation)

The full-window oscillation is not reproduced at 20k. As instructed, the presentation should be built around the **validated diagnostic findings**, not an optimized method:
1. The faithful Fourier+NTK reproduction collapses to a static field (A1, D, Y4, Z3), with NTK weights reaching 1e14–1e15.
2. The hard-constraint ansatz with (t/T)² is ill-conditioned: it needs a network output of about −8 400 near t = 0. A tanh²(ω₁t) factor needs O(1) and restores the gradient balance (E → X).
3. Representation is adequate: supervised fits recover 20.6 Hz (X3/X4/Y3).
4. The remaining failure is a temporal collapse front. RAD does not fix it (Z1); seed does not change it (Z2). It advances monotonically with physics computation: **0.06 s → 0.39 s (1.2 → 8.1 cycles) over 1.6e5 → 2.56e6 PDE evaluations**, with correct frequency in the reproduced cycles. Larger batches reach the same state 1.8× faster in wall-clock.
5. Mode 2 was not run: no Mode-1 method passed the gate, so per protocol there is nothing frozen to transfer.
