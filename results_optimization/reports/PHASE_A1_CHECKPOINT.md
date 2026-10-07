# Phase A1 decision checkpoint (FE-D-M1, 20k steps)

**Decision: CASE 3. STOP. A2 and A3 were NOT launched.** The faithful C0 is unexpectedly poor, and its trajectory does not support reaching 4.64e-4. Diagnosis is needed before more compute is spent.

## Runs

All four A1 runs share:
- seed 1234; 20 000 optimizer steps at mini-batch 32 (= 1 000 epochs, **2.2 % of the paper's 9.0e5 steps**);
- 6.4e5 PDE evaluations;
- 1 thread per run, 4 runs concurrently (measured: no per-step slowdown from concurrency);
- float32.

Sources: `tables/phaseA_summary.csv`, `optimization_leaderboard.csv`, `figures/phaseA_A1_*.png`.

## 1–12. Measured results at 20k steps

| # | Metric | B-A vanilla | B-B Fourier | **C0 Fourier+NTK** | C0-rc |
|---|---|---|---|---|---|
| 1 | **L2_paper** | 2.60 | 1.81 | **3.26** | 1.73 |
| 2 | **L2_exact** | 2.60 | 1.81 | **3.26** | 1.73 |
| 3 | log-log slope of L2_exact, last ½ / last ¼ | +0.07 / +0.36 | +0.04 / −0.06 | **+0.13 / +2.74** | +0.03 / +1.36 |
| 4 | best-so-far L2_exact (step) | 1.04 (9k) | 1.77 (11k) | 1.65 (15k) | 1.09 (17k) |
| — | late-window L2_exact (t ≥ 0.5 s) | 6.5 | 6.5 | 13.0 | 5.8 |
| 5 | PDE residual RMS(r)/RMS(u_tt) | 1.4e-5 | 1.99 | **21.7** | 6.8e-3 |
| — | training L_f, first → last | 2.4 → 2.8e-6 | 1.6e12 → 8.2e4 | 1.6e12 → 1.2e7 | 6.0e6 → 1.0 |
| 6 | BC error max (dimensionless) | 0.55 | 0.27 | 0.084 | 0.33 |
| 7 | IC error: u / u_t (dimensionless) | 0.55 / 2e-4 | 0.74 / 4e-4 | 0.14 / 8e-6 | 0.67 / 2e-5 |
| 8 | frequency error vs exact (extractor p95 at this error ≈ 3e-4) | 1.00 | 0.94 | **1.00 (fitted ω ≈ 0)** | 1.00 |
| — | phase error [rad] / amplitude error | 1.0 / 2.0 | 0.46 / 0.99 | 1.5 / 39 | 1.6 / 720 |
| 9 | wall-clock, training only | 13.0 min | 22.8 min | 26.4 min | 26.4 min |
| 10 | PDE evaluations | 6.4e5 | 6.4e5 | 6.4e5 | 6.4e5 |
| 11 | **NTK overhead** (separate from the base PINN cost) | — | — | **+17.8 %** (238 s; 200 updates × 1.19 s) | +17.9 % |
| — | base PINN cost | 39 ms/step | 69 ms/step | 67 ms/step | 67 ms/step |
| 12 | peak RSS: training / final evaluation | 0.75 / 1.16 GB | 0.83 / 1.29 GB | 0.83 / 1.29 GB | 0.85 / 1.28 GB |
| — | parameters / single-point inference | 201 801 / 92 µs | 241 601 / 300 µs | 241 601 / 313 µs | 241 601 / 329 µs |

`L2_paper` equals `L2_exact` to 3 significant figures in every run: at this error level the 4.4e-4 reference floor is invisible.

## Diagnosis evidence

1. **All four baselines end above L2 = 1**, i.e. worse than predicting u ≡ 0. None has a negative convergence slope over the last half.
2. **C0 and C0-rc collapse onto a quasi-static field** (`figures/phaseA_A1_C0_paper_diagnostics.png`, `phaseA_A1_C0_rc_diagnostics.png`).
   - Mid-span sits at about 70 mm (C0) and about 20–27 mm (C0-rc) for the whole 1 s, against a 20.6 Hz decaying oscillation.
   - The fitted frequency is ≈ 0 rad/s, against 129.3.
   - The velocity IC is satisfied *trivially* (L_ut ≈ 3e-10), because u_t ≈ 0 everywhere.
   - C0-rc satisfies the PDE to 0.7 %, but only by also violating the BCs (0.33 A₀) and the IC (0.67 A₀). It is a static wrong solution, not a slow route to the right one.
3. **The training dynamics differ qualitatively from the paper's, at matched epochs.** The paper's Fig. 5(b) is the loss history of this very benchmark (`figures/paper_fig5b_render.png`).

   | at ≈ 1 000 epochs | Paper | C0 |
   |---|---|---|
   | L_r | ≈ 1e5–1e6 | 1.2e7 |
   | L_ut | ≈ 1e-2 to 1e-3 (stays large: the network oscillates) | 3e-10 (static) |

4. **The initial loss scale is inconsistent with the paper** (`profiles/phaseA_init_scale.txt`; untrained models, geometric mean over 3 seeds).

   | Configuration | Initial L_f | Initial L_ut |
   |---|---|---|
   | C0 as resolved (2π, *physical* inputs, biases N(0,1)) | **2.3e12** | 4.5e-2 |
   | 2π, unit-normalised inputs | 8.7e8 | 3.0e-2 |
   | C0-rc with zero biases | 2.4e8 | 1.5 |
   | **Paper Fig. 5(b), epoch 0** (read from the plot, ±1 decade) | **≈ 1e8–1e9** | ≈ 1e1 |

   The paper's starting level is matched by normalised-input conventions, **not** by the "physical-input" resolution of ambiguity A2 that C0 uses. The paper never states its input normalization; its reference code standardises inputs.
5. **NTK drives the static attractor.** In C0, λ_ut grows from 1e13 to 8.5e14 and λ_ux to 1e15 (λ_f = 1). That enforces u_t(x,0) = 0 and u_x = 0 at the ends very strongly while the PDE is unresolved, and a nearly static field satisfies both at no cost. C0 is worse than Fourier alone at 20k steps (3.26 vs 1.81).

## Answers

**A. Is C0 moving in the correct direction?** **No.** Its training PDE loss falls by 5 decades, but its solution error does not: L2_exact is 3.26 at 20k, its best was 1.65, and the last-half slope is +0.13. It converges toward a non-oscillating field.

**B. Is the trajectory plausibly capable of approaching 4.64e-4?** **No.** The slope is non-negative, so no power-law projection exists. The model sits in a qualitatively wrong basin (fitted frequency ≈ 0), and its losses do not track the paper's at matched epochs. "More steps" has no evidence behind it.

**C. Is the result physically correct according to L2_exact?** **No.**
- L2_exact = 3.26; the PDE residual is 21.7× the solution's acceleration scale.
- IC error is 0.14 A₀ and BC error 0.08 A₀; the response is static.

The gap between L2_paper and L2_exact is irrelevant at this level. CASE 4 (frequency bias) does not apply.

**D. Is there evidence of an implementation problem?**
- **No coding error has been found.** The 83 tests confirm:
  - every Eq. 48 residual is exactly zero on an analytic probe;
  - u_xxxx converges at O(h²);
  - the NTK trace is exact;
  - the sampler follows the paper's semantics;
  - runs are bit-reproducible (A0 rerun identical).
- **There is evidence of a reproduction deviation in choices the paper does not specify.** The leading candidates:
  1. **input normalization (A2)**: initial loss 3+ decades above the paper's;
  2. **bias initialization (A14)**: N(0,1) vs zeros changes initial L_ut by about 100×; the zero-bias variants are closer to the paper's L_ut;
  3. possibly how NTK interacts with these scales (λ_ut ≈ 1e14–1e15).

  These are hypotheses until tested.

**E. Is the compute rate reasonable?** **Yes, and as profiled.**
- Base cost is 67 ms/step; NTK adds a measured, separately reported +18 %.
- Training peak memory is 0.83 GB; 4 concurrent 1-thread runs keep the single-run step cost.
- At this rate the full paper budget would take about 19 h per run, so it must not be spent on a configuration already in a static basin.

**F. Is the paper-faithful baseline behaving as expected?** **No.**
- The paper's ladder (Table 5: vanilla 1.11, NTK 0.881, NTK + Fourier 4.64e-4 at 45k epochs) and its Fig. 5(b) dynamics imply an oscillating solution emerging early, with NTK + Fourier best.
- We observe all methods above 1, C0 worse than Fourier alone, and a static collapse.
- Our faithful C0 is therefore not yet a faithful reproduction of the paper's training behaviour, although every documented equation is reproduced.

## Proposed diagnosis, before any A2/A3 compute (requires approval)

These are **repairs to unspecified reproduction choices, not optimizations**. Each is one change from C0.

Each run: 5 000 steps (250 epochs), about 7 min, 4 concurrent, about 15 min in total. Judged against the paper's Fig. 5(b) signatures at about 250 epochs, measured on its own trace (not via L2):
- **oscillation present**: fitted ω within 10 % of 129 rad/s;
- L_r of about 1e6;
- L_ut not collapsed (about 1e-2 to 1e-3);
- L2 trending down.

| ID | Change from C0 | Hypothesis tested |
|---|---|---|
| D1 | `input_norm = unit` (2π kept) | A2: the paper normalised inputs; matches its initial L_r |
| D2 | `bias_init = zeros` | A14: bias scale drives the static collapse |
| D3 | `input_norm = unit` + `bias_init = zeros` | the two together (run only as the 2×2 complement of D1 and D2) |
| D4 | C0-rc + `bias_init = zeros` | reference-code convention with zero biases; also matches the initial L_r |

Decision rule:
- If one variant restores the paper-like oscillating dynamics, it becomes the faithful C0. A1 for that C0 is re-run at 20k (≈ 27 min) and then the original A2/A3 plan proceeds, subject to your approval.
- If none does, the NTK/static-attractor interaction is diagnosed next (λ trajectories, a fixed-λ ablation) before anything else.
