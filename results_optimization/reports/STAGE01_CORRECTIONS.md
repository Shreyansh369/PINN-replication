# Stage 0.1 — Code, physics and reference corrections

**Status: corrections implemented and tested. NO TRAINING HAS BEEN RUN. Awaiting approval to start Phase A.**

All numbers below come from the scripts in `experiments/stage01/` (outputs in `results_optimization/tables/` and `profiles/`). Hardware: 4 vCPU Xeon @ 2.1 GHz, no GPU, torch 2.14 CPU, float32.

---

## 1. What changed

New code. The legacy notebook, `notebook_src/` and `results/` are **unchanged**.

| Path | Content |
|---|---|
| `src/beampinn/physics/beam.py` | exact eigen-roots (FF, SS, CF; any mode), mode shapes and x-derivatives up to order 4, damped modal function, `BeamCase` |
| `src/beampinn/physics/benchmarks.py` | benchmark identities and the **dual-reference system** (`paper`, `exact`; `material` for diagnostics only) |
| `src/beampinn/models/networks.py` | spatio-temporal Fourier PINN (Eqs. 38–43) with configurable `two_pi` / `input_norm`; vanilla MLP |
| `src/beampinn/sampling/samplers.py` | paper epoch / mini-batch sampler, with loss terms built from the BC table |
| `src/beampinn/losses/residuals.py`, `weighting.py` | Eq. 48 terms and loss, Eq. 49 residual, exact-trace NTK (Eq. 37), vectorised |
| `src/beampinn/evaluation/metrics.py`, `frequency.py` | dense grid, both references, physics, BC, IC and spectral metrics, latency |
| `src/beampinn/training/trainer.py` | training loop, compute accounting, checkpoint/resume, run records |
| `src/beampinn/config.py` | central config; unapproved mechanisms refuse to run |
| `src/beampinn/profiling/resources.py`, `utils/io.py` | peak RSS, hardware record, writers that refuse to write into `results/` |
| `experiments/run.py`, `experiments/update_leaderboard.py` | CLI with S0, S1, S2 and PAPER budgets, resumable; leaderboard writer |
| `configs/phaseA/{BA_vanilla, BB_fourier, C0_paper, C0_rc}.json` | Phase A configs, validated by dry run only |
| `tests/` (83 tests) | physics, conventions, model, residuals, sampler, NTK, metrics, config guards, trainer |
| `results_optimization/reports/CONVENTIONS.md` | which convention is used where |
| `optimization_leaderboard.csv` | header now carries **both** `L2` (paper-faithful) and `L2_exact`, plus `L2_late` |

## 2. Canonical benchmark (frozen identity)

`FE-D-M1`:
- **PDE**: `43.73² u_xxxx + u_tt + 7.08 u_t = 0` (Eq. 49). b = 50 N·s/m; steel per Table 3.
- **BCs**: `u = u_x = 0` at x = 0 and x = 2.75 m.
- **IC**: first-mode shape, 0.08 m at mid-span (Fig. 5c); `u_t = 0`.
- **Window**: t ∈ [0, 1] s, 20.59 cycles.
- **Metric**: Eq. 44 relative L2 on a uniform 201 × 2001 grid.
- **Target**: L2_paper < **4.64e-4**.
- Training uses physics only (IC + BC + PDE). It is never compared with the simply-supported 2.3e-3.

## 3. Dual-reference system (`tables/stage01_reference_comparison.csv`)

| Reference | c² | γ | β₁l | f₁ [Hz] | ω_d [rad/s] | Role |
|---|---|---|---|---|---|---|
| **paper** | 43.73² | 7.08 | 4.7300 (printed) | 20.590007 | 129.32239 | **primary gate** |
| **exact** | 43.73² | 7.08 | 4.730040745 | 20.590362 | 129.32462 | exact physics of the trained PDE |
| material | EI/ρA = 1912.046 | b/ρA = 7.0817 | 4.730040745 | 20.588925 | 129.31556 | diagnostic only |

| Pair (201 × 2001) | rel-L2 | late-window rel-L2 | Relative frequency gap |
|---|---|---|---|
| **exact vs paper** | **4.386e-4** | 1.405e-3 | 1.72e-5 |
| material vs paper | 1.357e-3 | 4.35e-3 | −5.28e-5 |
| material vs exact | 1.792e-3 | 5.74e-3 | −7.00e-5 |

The exact-vs-paper value is grid-independent: 4.378e-4 at 101 × 1001 and 4.390e-4 at 401 × 4001.

Supporting measurements:
- **Decomposition** of the 4.386e-4: frequency part 4.383e-4; mode-shape part 1.73e-5.
- The paper reference **satisfies the PDE exactly**, but **violates the slope BC at x = L** (|u_x|·L/A₀ = 2.43e-4), because its rounded root is not a root.
- The paper-shape IC differs from the exact-shape IC by 1.73e-5 rel-L2.

### Consequence for the primary gate
1. **A perfect solver of the trained PDE scores L2_paper = 4.386e-4 = 94.5 % of the target.**
   - The gate is guaranteed only if L2_exact < **2.5e-5** (triangle inequality).
   - It is met at L2_exact < **1.5e-4** if the PINN error happens to be orthogonal to the reference discrepancy.
2. **L2_paper rewards a frequency bias.** A PINN whose frequency is low by about 1.7e-5 matches the paper reference better than the true solution does. Every L2_paper result will therefore be reported **next to L2_exact and the fitted frequency against both references**, so that a "pass" produced by bias rather than accuracy is visible.
3. The published 4.64e-4 is only 5.8 % above this floor. That is *consistent with* the paper's PINN having solved its own PDE almost exactly, with the reported error being mostly reference rounding. This is a **hypothesis**; the PDF cannot confirm it.

## 4. Frequency-extractor uncertainty (`tables/stage01_frequency_extractor_uncertainty.csv`)

The damped-cosine fit at mid-span recovers the exact parameters to machine precision on noise-free traces. It also resolves the 1.72e-5 paper/exact gap to 1e-9.

Monte Carlo, 200 trials per cell, perturbations of rel-L2 size ε:

| ε | White noise: freq p95 | Smooth, 0–60 Hz: freq p95 | In-band, solution envelope: freq / phase p95 |
|---|---|---|---|
| 1e-4 | 2.8e-7 | 1.5e-6 | 3.3e-6 / 1.2e-4 rad |
| **4.64e-4** | 1.2e-6 | 6.9e-6 | **1.4e-5 / 5.3e-4 rad** |
| 1e-3 | 2.6e-6 | 1.5e-5 | 3.1e-5 / 1.2e-3 rad |
| 1e-2 | 2.7e-5 | 1.3e-4 | 3.1e-4 / 1.2e-2 rad |

Conclusions:
- At target-level error, the extractor's frequency uncertainty (≤ 1.4e-5) is of the same order as the 1.7e-5 paper/exact gap. **A 2e-5 hard gate would sit at the resolution limit.** As instructed, frequency, phase and amplitude are **secondary diagnostics**.
- The formal Gauss–Newton standard error underestimates the real uncertainty by 20–50×, so it is not used.

## 5. Paper-faithful baseline: implementation choices

| Item | Implemented | Source |
|---|---|---|
| Fourier map | `[cos 2πBv, sin 2πBv]`, physical v, σ = std, m = 100, σ_x = (1), σ_t = (10, 1), frozen | Eqs. 38–39; Table 4 #12; m from reference code |
| Network | 6 × 200 tanh, shared trunk, products H_x ⊙ H_t, linear head; 241 601 parameters | §4, Eqs. 40–43, Table 4 #12 |
| Initialization | Xavier-normal weights; biases N(0, 1) | reference code (paper NS) |
| PDE | `43.73² u_xxxx + u_tt + 7.08 u_t`, physical units, unscaled | Eq. 49 |
| Loss terms | L_u (IC u0 + BC u = 0), L_ut, L_ux, L_f, each `λ/(2N) Σ r²` | Eqs. 47–48 |
| L_u composition | 320 IC + 320 BC points (160 per end) | **OURS** (paper NS) |
| Sampling | 640 points per term redrawn each epoch; mini-batch 32 → 20 steps/epoch; 45 000 epochs = 9.0e5 steps, 2.88e7 PDE evaluations | Eq. 48, §5.1.1, §5.1.2 |
| NTK | λ_i = Σtr(K_j)/tr(K_i), **exact** trace on 32 rows per term (= the paper mini-batch), every 100 steps, replaced (no EMA), mean-normalized (= paper when N is equal) | Eq. 37; cadence and no-EMA from reference code |
| Optimizer | Adam, constant lr 1e-4, no grad clipping | Table 4 #12 (schedule NS) |
| IC target | exact-root mode shape × 0.08 m | measured 1.7e-5 from paper shape |
| Training data | physics only | Eq. 31 |
| Validation | 101 × 1001 grid (49 samples per period), both references | — |
| Final evaluation | 201 × 2001 grid (97 samples per period), both references | — |

## 6. Status of the Stage 0 defects

| # | Defect | Status |
|---|---|---|
| F1 | Fourier omitted 2π | **Fixed**: configurable, paper convention by default; C0-rc retained |
| F2 | `f_true = mode²` | **Fixed**: frequency compared against each reference's exact parameters |
| F3 | FFT-peak frequency estimate | **Fixed**: parametric fit, uncertainty quantified |
| F4 | 81 × 81 validation grid | **Fixed**: 101 × 1001 |
| F5 | NTK unequal-N sum trace, EMA, interval | **Fixed**: exact mean-normalized trace, reference cadence, configurable |
| F6 | SS-only physics | **Fixed**: FF, SS and CF, damped and undamped, any mode |
| F7 | BC error checked only `u` | **Fixed**: every required derivative at each end |
| F8 | lr schedule, clipping, resampling differed from paper | **Fixed**: paper defaults, all configurable |
| F9 | legacy checkpoints absent | Not recoverable. New runs save `latest.pt` and `final.pt` |
| F10 | NTK host-sync loop | **Fixed**: vectorised; 1.3–4.1× faster than the loop up to 512 rows (slower at 2560, unused) |
| F11 | missing tests | **Fixed**: 83 new tests; 32 legacy checks still pass |
| F12 | stale legacy text | Left as is (legacy files are not edited) |

## 7. Profiling of the new pipeline (`profiles/stage01_step_profile.csv`)

Forward + backward only. Same-process A/B timing showed **±20 % machine jitter** (52–71 ms for the identical step), and no regression against the legacy pipeline.

| Net | mb = 32 step | NTK update (exact, 32 rows × 4 terms) | NTK overhead at every-100 |
|---|---|---|---|
| 6 × 200 | 54–72 ms | 0.67 s | ≈ 9 % |
| 4 × 200 | 39 ms | 0.28 s | ≈ 7 % |
| 3 × 64 | 27 ms | 0.08 s | ≈ 3 % |

- The exact NTK trace cost grows **superlinearly** with the number of rows: 6.8 s for 512 rows, 134 s for 2560. Trace rows are therefore held at 32 per term, so Phase E (mini-batch) is not confounded by NTK cost.
- Revised C0 cost estimates:

| Budget | Steps | Estimated wall-clock |
|---|---|---|
| S1 | 20 000 | ≈ 20–26 min |
| S2 | 100 000 | ≈ 1.7–2.2 h |
| PAPER | 9.0e5 | ≈ 15–19 h |

## 8. Tests

Run with `python -m pytest tests/`: **83 passed.** Highlights:
- The exact reference satisfies its PDE (< 1e-12), every BC (< 1e-9) and the ICs.
- The paper reference satisfies the PDE but violates the slope BC.
- 2π convention check.
- Autograd u_xxxx converges at O(h²) to finite differences.
- float32 PDE-residual error < 1e-3 at paper scales.
- An analytic-probe model gives zero residual on every Eq. 48 term (both groupings).
- Every epoch point is used exactly once.
- Vectorised NTK trace equals the loop to 1e-10.
- Unapproved and rejected mechanisms refuse to run.
- The legacy `results/` tree is write-protected.
- **Resume is exact**: 3 + 3 steps equal 6 straight steps.

The trainer tests take 2–6 optimizer steps on a 2 × 12 network inside pytest temp dirs. These are software tests; nothing is kept and they are not experiments.

## 9. Incident (resolved)

An ad-hoc A/B timing script ran legacy notebook cells from the repo root. Those cells save plots relative to the current directory, and they **overwrote four PNGs in `results/figures/`** and created two empty directories. All were restored byte-identical from git, verified with `git diff HEAD -- results/`, and the empty directories were removed. The Stage-0 scripts that run legacy cells now always `chdir` into a fresh temp directory, and the new package refuses to write into `results/`.

## 10. Not done (out of scope until approved)
- No training of any kind.
- RAD, RAR, annealing, GradNorm, hard constraints, mixed formulation and L-BFGS exist only as guarded config fields.
- No higher modes, no full replication, no presentation.

## 11. Decisions requested
1. Accept the implementation choices in §5, especially the OURS items: the 320/320 L_u split, the 32-row NTK trace cap, and normal bias initialization.
2. **The gate given the floor (§3).** Keep "L2_paper < 4.64e-4" as the primary condition as approved. I additionally propose to **report and require L2_exact ≤ L2_paper** for any claimed pass, so that a pass cannot come from a frequency bias toward the paper's rounded root. Your call.
3. Approve Phase A:
   - S0 sanity on BA_vanilla, BB_fourier, C0_paper and C0_rc;
   - S1 on all four;
   - C0 seeds 2–3 at S1 (noise floor);
   - C0 at S2.

   Estimated **≈ 5–6 h** CPU, then a stop-and-report.

   Following your order, Phase B will be Fourier + RAD (vs B-B) together with Fourier + NTK + RAD (vs C0), and Phase C the same pair for RAR.
