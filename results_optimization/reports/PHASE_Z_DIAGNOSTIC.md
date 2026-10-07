# Phase Z: sampling, seed and batch diagnostics (Mode 1, 5 000 steps)

**Outcome: CASE D / CASE F. No run passes the oscillation gate with persistence.**
- RAD (Z1) and mini-batch 128 (Z4) do not resolve the failure at this budget. Z4 measurably extends how long the oscillation persists, at 4× the PDE evaluations.
- Y1's behaviour is robust to the seed (Z2).
- RAD does not rescue the paper Fourier+NTK baseline (Z3).

STOPPED for approval.

## Parent → child differences (printed at creation; one change each)

| Child | Parent | Change |
|---|---|---|
| Z1 | Y1 (hard tanh², D4 conv., LR schedule) | `sampler.adaptive none → rad`; RAD k = 1, c = 1, 5 000 candidates, every 500 steps, multinomial without replacement, detached residuals (identical to E3/E5) |
| Z2 | Y1 | `seed 1234 → 1235` (changes Fourier draws, initialization, sampling) |
| Z3 | Y4 (paper Fourier+NTK + LR schedule) | the same RAD as Z1 |
| Z4 | Y1 | `sampler.mini_batch 32 → 128`; steps held at 5 000 (so 4× PDE evaluations; epochs are implicitly 4× too, i.e. 5 steps/epoch) |

## Results (`tables/phaseZ_oscillation.csv`, `tables/phaseZ_temporal.csv`)

| | **Y1** (parent) | Z1 +RAD | Z2 seed 1235 | Z4 mb 128 | Y4 (C0+sched) | Z3 Y4+RAD |
|---|---|---|---|---|---|---|
| L2_paper / L2_exact | 0.774 | 0.718 | 0.799 | **0.526** | 2.75 | 2.60 |
| late-window L2_exact (t ≥ 0.5 s) | 1.06 | 1.01 | 1.23 | 1.07 | 10.7 | 10.1 |
| fitted ω ratio / decay [1/s] (exact 1 / 3.54) | fit fails* | 0.972 / 16.3 | fit fails* | 0.990 / 9.0 | 0 / — | 0 / — |
| amplitude ratio (whole trace) | 0.47 | 0.56 | 0.44 | 0.69 | 0.15 | 0.15 |
| max \|u_t\| ratio | 0.76 | 0.92 | 0.82 | 0.93 | 0.02 | 0.02 |
| IC / BC error | 4.8e-6 / 1.4e-5 | same | same | same | 0.31 / 0.30 | 0.34 / 0.24 |
| relative PDE residual | 0.178 | 0.169 | 0.267 | **0.110** | 71.5 | 51.1 |
| training wall-clock | 300 s | 335 s (RAD 35 s = **+12 %**) | 304 s | **611 s (+104 %)** | 373 s (NTK 56 s) | 402 s (NTK 56 s, RAD 32 s) |
| PDE evaluations | 1.6e5 | 1.6e5 | 1.6e5 | **6.4e5 (4×)** | 1.6e5 | 1.6e5 |
| RAD candidate evaluations | 0 | 4.5e4 | 0 | 0 | 0 | 4.5e4 |
| parameters / training peak RSS | 241 601 / 0.82 GB | 241 601 / 1.37 GB | 241 601 / 0.81 GB | 241 601 / 0.90 GB | 241 601 / 0.82 GB | 241 601 / 1.29 GB |
| **Gate (Phase-D rule + persistence)** | FAIL | **FAIL** (≈2 cycles) | FAIL | **FAIL** (≈4 cycles) | FAIL (static) | FAIL (static) |

\* A single global damped-cosine fit is not meaningful for oscillate-then-collapse traces. The automated rule labelled Z1 and Z4 "OSCILLATORY" on frequency/amplitude/velocity alone. Under the approved **persistence** requirement both fail, as the time-resolved amplitude shows.

## Time-resolved diagnostics (`tables/phaseZ_temporal.csv`; mid-span RMS ratio, predicted/exact)

| window [s] | 0–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | 0.4–0.7 | 0.7–1.0 |
|---|---|---|---|---|---|---|
| Y1 amplitude / velocity | 0.76 / 0.74 | 0.41 / 0.38 | 0.12 / 0.10 | 0.05 / 0.02 | 0.13 / 0.04 | 0.49 / 0.22 |
| Z1 amplitude / velocity | 0.91 / 0.91 | 0.50 / 0.48 | 0.19 / 0.17 | 0.05 / 0.03 | 0.11 / 0.05 | 0.36 / 0.26 |
| Z2 amplitude / velocity | 0.73 / 0.71 | 0.36 / 0.35 | 0.16 / 0.11 | 0.13 / 0.05 | 0.26 / 0.07 | 1.15 / 0.14 |
| **Z4 amplitude / velocity** | 0.92 / 0.92 | **0.78 / 0.78** | **0.54 / 0.55** | 0.22 / 0.18 | 0.11 / 0.04 | 0.46 / 0.18 |
| Y4 / Z3 velocity | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.02 | 0.02 / 0.02 | 0.04 / 0.04 | 0.18 / 0.17 |

Late-window displacement ratios such as 1.15 or 0.46 come from a slow drift or offset of a near-static prediction (velocity ratio ≤ 0.26), not from oscillation.

| share of squared PDE residual per window | 0–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | 0.4–0.7 | 0.7–1.0 |
|---|---|---|---|---|---|---|
| Y1 | **41 %** | 19 % | 5 % | 9 % | 13 % | 13 % |
| Z1 | 11 % | **59 %** | 9 % | 5 % | 8 % | 9 % |
| Z4 | 18 % | 17 % | 13 % | **32 %** | 8 % | 13 % |

The per-window PDE-loss gradient norm is within a factor of about 3–8 across windows in Y1, Z1, Z2 and Z4 (e.g. Y1: 3e5 early vs 1e5 late).

## Interpretation (only what the measurements support)

Described as observed: for Y1, Z1, Z2 and Z4 the model satisfies the IC exactly, reproduces approximately the first cycles, and then **collapses toward a low-amplitude trajectory**. Testing the candidate mechanisms:

- **Temporal drift toward low amplitude: supported.** Amplitude and velocity ratios fall monotonically window by window in all four hard-constrained runs.
- **Residual concentrated in a particular time region: supported.** The window holding the largest residual share coincides with where the amplitude collapses, and it moves with it: 0–0.05 s (Y1) → 0.05–0.1 s (Z1) → 0.2–0.4 s (Z4).
- **Insufficient gradient signal in late windows: not supported** as the dominant cause. Late windows receive gradient norms of the same order as early ones.
- **Stochastic / batch effects: partly supported.** The 4× larger mini-batch (Z4) pushes the collapse front from about 0.05–0.1 s to about 0.2–0.4 s and gives the lowest L2 (0.526) and PDE residual (0.110). But Z4 also used **4× the PDE evaluations and 2× the wall-clock**, so batch size and total physics budget are **confounded** in this comparison.

The four required comparisons:

1. **Y1 → Z1: does adaptive sampling prevent temporal collapse? No.**
   - First-cycle accuracy is better (amplitude 0.91 vs 0.76) and L2 slightly lower (0.718 vs 0.774), but the collapse comes after about 2 cycles, as in Y1. Cost: +12 % wall-clock and +45 000 candidate evaluations.
   - The RAD-selected points carry 1.3–2.2× the mean candidate residual, yet their mean time is 0.45–0.49 s (uniform: 0.5 s). With k = 1, c = 1 the temporal concentration on the collapse region is weak (`tables/phaseZ_rad_updates.csv`, `figures/phaseZ_rad_distribution.png`).
2. **Y1 → Z2: is Y1 robust to initialization? Yes, qualitatively.** The same failure mode (collapse after about 2 cycles); L2 0.799 vs 0.774 (3 %). No evidence of seed sensitivity worth a variance study at this stage.
3. **Y1 → Z4: does batch size change the dynamics? Yes:** persistence extends from about 2 to about 4 cycles, L2 0.774 → 0.526, PDE residual 0.178 → 0.110. This comes with 4× the PDE evaluations, 2.04× the wall-clock and +80 MB of memory, and it still fails the persistence gate.
4. **Y4 → Z3: does RAD rescue the paper baseline? No.** It remains static (velocity ratio ≤ 0.04 up to 0.7 s), with NTK weights pathological as before. RAD gives no benefit to soft Fourier+NTK here, so CASE E does not apply.

## Recommendation: exactly ONE next experiment (requires approval)

**Y1 extended to 20 000 steps, everything else unchanged.** Single change: budget 5k → 20k. That is **6.4e5 PDE evaluations, identical to Z4's**; about 20 min on one core. Save checkpoint snapshots every 5k steps, to measure the collapse-front position vs training. This needs a 3-line trainer addition (snapshot files), with no effect on training.

What it answers:
- **Deconfounds Z4:** at matched PDE evaluations, is the longer persistence due to the larger mini-batch or to the larger total physics budget?
- **Trajectory:** does the collapse front keep advancing with training (5k → 20k), which would make a longer run credible? Or does it stall, which would rule budget out?
- **Matched baseline:** C0 at 20k steps already exists (A1; static, L2 3.26).

Time: it is ≈22:40 UTC. If the front advances and the gate is passed, Mode-2 can follow. If not, the presentation reports the validated diagnostic chain:
- static attractor of the faithful Fourier+NTK;
- ansatz conditioning of the hard constraint;
- representation adequacy;
- temporal collapse, with the measured effects of the schedule, RAD and batch size.
