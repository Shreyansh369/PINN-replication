# Y1-20K: optimization-budget diagnostic (Mode 1)

**Outcome: CASE B.** More optimization budget moves the collapse **progressively and monotonically later**: from 1.2 to 3.1 cycles over 5k → 20k steps. The oscillation still collapses at 0.151 s, i.e. after 3.1 of the 20.6 cycles in the window.

**This is not a successful vibration reproduction.** No optimized-PINN claim is made. STOPPED for approval.

## Exact configuration

`configs/phaseY/Y1_20K.json`; run `Y1_20K__s1234__36f2d370bf`.
- Y1 recipe: hard-constrained Fourier PINN with time factor tanh²(ω₁t) (ω₁ = 129.37 rad/s from the PDE coefficient + fixed-fixed eigenproblem), D4 Fourier convention (no 2π, standardised inputs, zero biases), 6 × 200 tanh, m = 100, σ_x = (1), σ_t = (10, 1); 241 601 parameters.
- Training: Adam, lr = 1e-3 · 0.9^(step/1000), mini-batch 32 from 640 points per epoch (redrawn each epoch); PDE loss only.
- No RAD, NTK, causal weighting or extra loss weights; seed 1234; float32; 1 thread.

**Only change from Y1: `train.max_steps` 5 000 → 20 000.** Verified by an automatic config diff. The other differences listed by the diff are logging cadence and a budget label, which are not part of the run key. Snapshots were saved every 5 000 steps with the new `train.snapshot_every`; it is cadence only, excluded from the key, and tested.

**Bit-level check:** the 5k checkpoint of Y1-20K reproduces the original 5k Y1 run exactly (L2 0.774, PDE residual 0.178, identical collapse time). Because of the LR schedule, the first 5k steps of the two runs are identical by construction.

## Persistence measure (identical for every model)

- Local amplitude A(t) is the half peak-to-peak of the mid-span displacement in a sliding window of one damped period (0.0486 s); R(t) = A_pred(t)/A_exact(t).
- **Collapse time** = first t ≥ P/2 with R(t) < 0.5. **Persistence** = collapse time × f_d (cycles). The full window is 20.6 cycles.
- The same is computed for the velocity.

## Checkpoint results (`tables/phaseY1_20K_checkpoints.csv`)

| Step | L2_exact | L2_paper | PDE residual (rel.) | Freq. fit [rad/s] (exact 129.32) | Decay fit [1/s] (exact 3.54) | Amp. ratio | max \|u_t\| ratio | IC / BC error | **Collapse time** | Persistence (displ. / vel.) | Train time (cum.) | PDE evals (cum.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5k | 0.774 | 0.774 | 0.178 | fit fails† | 18.4 | 0.47 | 0.77 | 4.8e-6 / 1.4e-5 | **0.057 s** | 1.2 / 1.4 cycles | 268 s | 1.6e5 |
| 10k | 0.665 | 0.665 | 0.129 | 127.6 (−1.3 %) | 13.3 | 0.57 | 0.89 | 4.8e-6 / 1.4e-5 | **0.082 s** | 1.7 / 1.9 | 536 s | 3.2e5 |
| 15k | 0.640 | 0.640 | 0.142 | 127.5 (−1.4 %) | 12.5 | 0.59 | 0.98 | 4.8e-6 / 1.4e-5 | **0.104 s** | 2.1 / 2.4 | 806 s | 4.8e5 |
| 20k | **0.550** | 0.550 | **0.108** | 127.9 (−1.1 %) | 9.75 | 0.65 | 0.97 | 4.8e-6 / 1.4e-5 | **0.151 s** | **3.1 / 2.9** | 1 078 s | 6.4e5 |

† The global damped-cosine fit is unreliable on an early-collapse trace. From 10k steps on, the fitted frequency is within 1.1–1.4 % of exact. The fitted decay rate falls monotonically (18.4 → 9.75 1/s) but stays 2.8× the physical 3.54 1/s, i.e. **over-damped**.

**Collapse time vs training: 0.057 → 0.082 → 0.104 → 0.151 s.** It is monotonic and does not plateau; the last 5k steps gave the largest advance (+0.047 s). L2_exact and the PDE residual improve overall; the PDE residual is non-monotonic between 10k and 15k (0.129 → 0.142).

## Compute-matched comparison (≈ 6.4e5 PDE evaluations)

| Method | Steps | Batch | Optimizer updates | PDE evaluations | L2_exact | PDE residual | Persistence (collapse time) | Training time | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Y1 | 5k | 32 | 5 000 | 1.6e5 | 0.774 | 0.178 | 1.2 cycles (0.057 s) | 300 s‡ | FAIL |
| Z4 | 5k | 128 | 5 000 | **6.4e5** | 0.526 | 0.110 | **3.2 cycles (0.153 s)** | **611 s** | FAIL |
| Y1-20K | 20k | 32 | 20 000 | **6.4e5** | 0.550 | 0.108 | **3.1 cycles (0.151 s)** | 1 078 s | FAIL |
| *Z1 (Y1 + RAD)* | 5k | 32 | 5 000 | 1.6e5 (+4.5e4 cand.) | 0.718 | 0.169 | 1.6 cycles (0.080 s) | 335 s | FAIL |
| *Z2 (Y1, seed 1235)* | 5k | 32 | 5 000 | 1.6e5 | 0.799 | 0.267 | 1.1 cycles (0.054 s) | 304 s | FAIL |

‡ Y1/Z-runs ran 4-concurrent; Y1-20K ran alone. Phase A measured no per-step slowdown from concurrency (66–70 ms/step either way). Y1-20K's 268 s at 5k (vs Y1's 300 s) is within the documented ±20 % timing jitter.

**This is a compute-matched comparison of many small-batch optimizer updates versus fewer large-batch updates. It is not a pure batch-size isolation.** At equal PDE evaluations, the two reach practically the same state:
- persistence 3.1 vs 3.2 cycles;
- L2 0.550 vs 0.526;
- PDE residual 0.108 vs 0.110.

The large-batch route got there in **0.57× the wall-clock** (611 s vs 1 078 s), using 4× fewer optimizer updates.

## Scientific interpretation

**What this experiment establishes:**
1. For the Y1 recipe, the collapse point is **not fixed**: it advances monotonically with optimization budget over 5k–20k steps.
2. Within this range, persistence tracks the **total number of PDE-residual evaluations** rather than the number of optimizer updates. Z4 and Y1-20K, with 4× different update counts but equal evaluations, reach the same collapse time, accuracy and residual. On this CPU, fewer large-batch updates are therefore the more wall-clock-efficient way to spend that budget.
3. The frequency is captured early (within ≈1–1.4 %), and the amplitude/velocity of the first cycles improve with training. The model remains over-damped (fitted decay 2.8× physical).
4. Combined with Phases D–Z: the remaining Mode-1 failure behaves like a **budget-limited, front-like propagation** of the correct dynamics from the exactly-enforced IC into later times, not like a representation limit (X3/X4 fit the solution) or a fixed attractor.

**What it does not establish:**
- That the front will keep advancing at this rate, or reach 1 s. A linear extrapolation of the 5k–20k points (≈ +0.03–0.05 s per 1.6e5 evaluations) would need on the order of 4e6 further PDE evaluations. That is an **indication only**: the rate may change, and the late, low-amplitude cycles may behave differently.
- Anything about L2 vs the paper target. At 0.55 the error is still three orders of magnitude above 4.64e-4, and full persistence would not by itself imply the target is reachable.
- Generality beyond seed 1234 at 20k (Z2 showed the same failure mode at 5k, but there is no 20k replicate).
- Anything about Mode 2, which remains blocked until a Mode-1 method passes the gate.

**Limitations:** a single seed at 20k; a 5k checkpoint spacing; the collapse threshold of 0.5 is a fixed but arbitrary choice (the same for all runs); the timing comparison mixes concurrent and solo execution (no measured effect on step time, within ±20 % jitter).

## Figures

- `phaseY1_20K_disp_traces.png`: exact vs 5k / 10k / 15k / 20k, collapse time marked.
- `phaseY1_20K_vel_traces.png`: the same for velocity.
- `phaseY1_20K_collapse_vs_steps.png`: collapse time vs cumulative PDE evaluations, with Y1, Z1, Z2, Z4.
- `phaseY1_20K_loss_and_residual.png`: training (PDE) loss, validation L2_exact, and checkpoint PDE residual vs steps.

## Recommendation: exactly ONE next experiment (requires approval)

**Z4 recipe (Y1 with mini-batch 128) extended 5k → 20k steps, nothing else changed.**
- 2.56e6 PDE evaluations (4× Z4, 4× Y1-20K); ≈ 41 min on one core (measured 611 s per 5k steps).
- Snapshots every 5k steps (6.4e5 evaluations apart), so the first point coincides with Z4/Y1-20K.

Rationale: point 2 above shows this recipe reaches the same state at 0.57× the wall-clock, so it is the cheapest way to test the one open question, **does the collapse front keep advancing with 4× more physics budget, and how fast?**
- If the front reaches the full window, that run becomes the strongest current candidate requiring confirmation.
- If it advances but stays short, extrapolation becomes better grounded.
- If it plateaus, budget is ruled out (CASE C).

Deadline: it is ≈23:05 UTC. A 41-min run plus analysis ends ≈23:55. If the presentation window cannot absorb that, the alternative is to freeze now and present the validated diagnostic chain:
- static attractor of the faithful Fourier+NTK;
- conditioning of the hard-constraint ansatz;
- representation adequacy;
- the measured temporal-collapse front and its budget dependence.
