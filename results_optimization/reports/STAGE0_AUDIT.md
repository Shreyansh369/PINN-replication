# Stage 0 — Repository, code, physics and paper audit

**Status: STAGE 0 COMPLETE. NO TRAINING HAS BEEN RUN. AWAITING APPROVAL OF THE EXPERIMENT MATRIX (§I) AND THE BENCHMARK RESOLUTIONS (§C.3).**

Target paper: Söyleyici & Ünver, *Eng. Appl. Artif. Intell.* 141 (2025) 109804, doi:10.1016/j.engappai.2024.109804 (all 20 pages read from the PDF; equations checked against high-resolution renders, not only text extraction).

Everything numeric in this report is reproducible from `experiments/stage0/`:

| Script | Output | What it measures |
|---|---|---|
| `run_existing_tests.py` | `profiles/stage0_existing_tests.txt` | the legacy notebook's 32 pre-training checks (run in a scratch cwd so `results/` is untouched) |
| `benchmark_numerics.py` | `profiles/stage0_benchmark_numerics.txt` | exact eigenvalue roots, frequencies, damping ratios, L2 floors from rounded constants, IC modal content |
| `fourier_support.py` | `profiles/stage0_fourier_support.txt` | target frequency vs Fourier-feature frequency support under three input conventions |
| `profile_step.py` | `profiles/stage0_step_profile.txt` | forward+backward cost per step and per NTK update (no optimizer step, nothing trained) |
| `build_registry.py` | `../paper_benchmark_registry.csv` | 36 rows, every paper number with its table/equation |

Hardware for all timings: 4 vCPU Intel Xeon @ 2.10 GHz, 15 GB RAM, **no GPU**, torch 2.14.0 (CPU), float32. Timing jitter between two identical profiling runs: up to ~15 %.

---

## A. Current code architecture

The code lives as strings inside `notebook_src/sec_{a,b,c,d}.py`, assembled into `beam_pinn_research.ipynb` by `make_notebook.py`. It is not an importable package, has no CLI, and its experiment parameters are set in `base_cfg()` (`sec_c.py:56`) and module-level constants.

### A.1 Implementation map

| Step | Implementation | Location |
|---|---|---|
| INPUT | `(x*, t*)` on `[0,1]²` | — |
| NORMALIZATION | `x* = x/L`, `t* = t/T_ref`, `u* = u/A_n`; `T_ref = 2π/ω₁` (default "window b") or `1 s` (paper window "a"). PDE becomes `α u*_xxxx + u*_tt + ζ u*_t = 0` | `sec_a.py` `NonDim` |
| FOURIER | `γ(v) = [cos(Bv), sin(Bv)]`, `B ~ N(0,σ²)` fixed buffer, m = 64, σ_x = 1, σ_t = (1, 10). **No 2π factor** | `sec_b.py:730` |
| NETWORK | shared tanh trunk (depth × width, Xavier), multiplicative merge `h_x ⊙ h_tk`, concat, linear head (matches paper Eqs. 38–43 and the reference code `Wave1D_NTK_ST_mFF`) | `sec_b.py` `MultiScaleFourierMLP` |
| DERIVATIVES | nested reverse-mode `autograd.grad` per point (`d1`), 4 levels in x, 2 in t | `sec_b.py:230` |
| PDE RESIDUAL | `(α u_xxxx + u_tt + ζ u_t) / ω*²` (division by ω*² is a documented rescaling) | `sec_b.py:230`, `:260` |
| IC | `u(x,0) − sin(nπx)`, `u_t(x,0)` — **simply-supported only** | `sec_b.py:249` |
| BC | `u`, `u_xx` at `x* = 0, 1` — **simply-supported only** | `sec_b.py:254` |
| LOSS WEIGHTING | fixed λ = 1; NTK trace weights (row-subsample, 16 rows, every 200 it, EMA β = 0.5); temporal causal | `sec_b.py:819`, `sec_c.py:659` |
| OPTIMIZER | Adam, lr 1e-3 exponentially decayed to 1e-4 over the run, grad-norm clip 1e4 | `sec_b.py:357`, `:407` |
| SAMPLING | fresh random batch every iteration: 512 collocation (stratified in t), 128 IC, 128 BC. No fixed dataset, no epoch or mini-batch concept | `sec_a.py:755` |
| VALIDATION | rel-L2 on an 81 × 81 grid every 100 iterations; final metrics on a 201 × 201 grid | `sec_b.py:496`, `evaluate` |
| CACHING | checkpoint keyed by an MD5 hash of `RunConfig` | `sec_b.py` `run_experiment` |

### A.2 Reusable as-is (verified by tests)
`BeamParams`/`NonDim` algebra; the damped single-mode modal time function `_modal_time` (exact, including the underdamped branch); `d1`; `MultiScaleFourierMLP` (structure matches the paper); the row-subsample NTK trace estimator (unbiased, validated against the exact trace); the causal weighting code (kept for the negative result); the `RunConfig`-hash caching idea; plotting panels.

### A.3 Fragile, incorrect or hard-coded

| # | Issue | Severity | Evidence |
|---|---|---|---|
| F1 | **Fourier encoding omits the 2π of the paper's Eqs. 38–39.** With σ_t2 = 10 on `t ∈ [0,1]`, no feature reaches the paper-window target frequencies: SS ω₁ = 57.1 rad (5.7 σ), FE ω₁ = 129.4 rad (12.9 σ). Under Eq. 38–39 an average of 36 (SS) and 4 (FE) of 100 features reach it. The paper's reference code also omits 2π but standardises inputs (σ_eff = 34.6 rad/s). | **High**: the legacy paper-window failure (`P_paper_window` 0.967 vs 2.3e-3) is confounded with this. That is a hypothesis to test in Phase A, not a conclusion. | `stage0_fourier_support.txt` |
| F2 | The frequency-error metric hard-codes `f_true = mode²`, which holds only for the one-period normalization. It is wrong for any `nd_paper` run (`P_paper_window`), where the target is 9.08. | Medium (latent; REPORT.md does not quote that number) | `sec_b.py:543` |
| F3 | Frequency is estimated from the FFT peak of a 1 s record (1 Hz bins plus window leakage). That cannot resolve the ≈1e-5 relative frequency accuracy the target demands (see C.4). A parametric damped-sinusoid fit is needed. | Medium | — |
| F4 | The validation grid is 81 × 81. At 20.6 cycles that is 3.9 samples per period, so the convergence curves would be aliased. | Medium | `sec_b.py:496` |
| F5 | The NTK trace uses the sum over N_i points, while batch sizes are unequal (512 vs 128). λ_pde therefore differs by ×4 from the paper's equal-N convention (Eq. 48: all N = 640). EMA β = 0.5 and the 200-iteration interval are OURS. The reference code computes the exact K on 120 points every 100 iterations with no EMA. | Medium (affects paper faithfulness) | `sec_b.py:839` |
| F6 | The physics is simply-supported only: IC = `sin(nπx)`, BC = `u, u_xx`, and the analytical solution and BC error are SS-specific. Fixed-fixed and cantilever are not implemented. | **Blocking** for the proposed canonical benchmark | `sec_b.py:244–254`, `sec_a.py:648` |
| F7 | The `evaluate` BC error checks only `u` at the ends, not the derivative condition. | Low | `sec_b.py:538` |
| F8 | Learning-rate schedule (1e-3 → 1e-4), grad clipping 1e4 and per-iteration resampling all differ from the paper (lr 1e-4; batch 640 with mini-batch 32). | Medium (paper faithfulness) | — |
| F9 | **No checkpoints in the repository** (`*.pt` is gitignored and `results/checkpoints/` is absent). The 22 legacy models cannot be re-evaluated with new metrics. Only their config and history CSVs exist. | Medium | — |
| F10 | NTK per-row loop calls `.item()` per parameter tensor (host sync) and makes 80 sequential backward passes per update. | Efficiency | `sec_b.py:838` |
| F11 | The test suite covers only the SS analytical solution with ζ = 0. There are no tests of the damped PDE residual, the FE/CF mode shapes, the Fourier convention, or float32 4th derivatives *through the network* at paper-window frequencies (test 10 uses an analytic probe). | Medium | `stage0_existing_tests.txt` |
| F12 | Stale text: the README says "N/A (PDF unavailable)"; the §3.2 markdown describes a "1 m, 5 mm deep" beam. | Cosmetic | — |

### A.4 Correction to REPORT.md (no results change)
REPORT.md §3.1 inconsistency #1 ("the paper prints 43.732 instead of 43.732²") is a **text-extraction artefact**. The rendered PDF clearly shows `43.73² ∂⁴u/∂x⁴` in Eqs. 46, 49, A.4, A.7, B.5 and B.8 (pdftotext flattens the superscript into "43.732"). The paper's PDE coefficient is therefore `43.73² = 1912.31`. The legacy code uses `EI/ρA = 1912.05` from the material data. The two differ by 1.4e-4, and C.4 shows that this difference matters at the target accuracy. Inconsistency #2 (Eq. A.4/A.7 print the free-free BCs `u_xx = u_xxx = 0`) is **confirmed real**. Inconsistency #3 (4 vs 6 layers) is confirmed real.

---

## B. Current experimental evidence (preserved, not re-litigated)

All legacy runs: 4 × 200 tanh, m = 64, seed 1234, full-batch per-iteration resampling, ≤ 4000 iterations, 4 CPU threads, float32. **None of them is on a paper benchmark except `P_paper_window` (SS-U-M1).** The headline runs are on "mode 2 in the one-period normalization" (4 cycles), which is the repository's own construct.

| Evidence | Value | Benchmark | Status |
|---|---|---|---|
| Vanilla / Fourier / Fourier+NTK rel-L2 | 0.990 / 0.990 / **0.411** | own (mode 2, 4 cycles) | preserved |
| Fourier cuts IC err 23×, BC err 8×; NTK converts capacity to accuracy | — | own | preserved |
| 2×2 causal ablation: NTK main effect −0.880, interaction **+0.875** (antagonistic); PDE residual 18.6 → 122.7 | — | own | **causal weighting REJECTED** (negative baseline) |
| Causal-strength sweep: off is best at every active strength (reproduced twice) | — | own | preserved |
| σ_t2 sweep inconclusive (spread < 20 % margin, unconverged) | — | own | preserved; F1 means its bandwidths are not the paper's |
| Denormal slowdown 9.5× under strong causal weighting; fixed with `set_flush_denormal` | — | — | preserved |
| `P_paper_window` (Fourier+NTK) | **0.967** at 4000 it | SS-U-M1 (paper 2.3e-3) | confounded by F1 and F8 |
| Mode-1 one-period Fourier+NTK | 0.0110 | own (1 cycle) | not a paper number |
| Determinism and beam-independence of the non-dimensional problem | to 4.4e-16 | — | preserved |
| Pre-training tests | **32/32 PASS** (re-run here) | — | `stage0_existing_tests.txt` |

---

## C. Exact paper benchmarks

Full table: `paper_benchmark_registry.csv` (36 rows: 7 forward/inverse/experimental cases, the Table 4 hyperparameter sweep, the Table 5 ladder, the Fig. 6 mini-batch sweep, the Table 2 wave-equation reference). Summary of the beam forward cases:

| ID | Beam | BCs | Damping | Window | Cycles | Paper rel-L2 | Epochs | Batch / mini-batch | Source |
|---|---|---|---|---|---|---|---|---|---|
| **FE-D-M1** | fixed-fixed, L = 2.75 | u = u_x = 0 | b = 50 (γ = 7.08) | [0,1] s | 20.59 | **4.64e-4** | 45 000 | 640 / 32 | §5.1.1, Eq. 49, Fig. 5, T4 #12, T5, F6 |
| FE-U-M1 | fixed-fixed | u = u_x = 0 | none | [0,1] s | 20.59 | 2.70e-3 | 45 000 | 640 / 32 | §5.1.1, Eq. 46, Fig. 4 |
| SS-U-M1 | simply-supp. | u = u_xx = 0 (A.4 typo) | none | [0,1] s | 9.08 | 2.3e-3 | 30 000 | 960 / NS | App. A, Fig. A.19 |
| SS-D-M1 | simply-supp. | u = u_xx = 0 | b = 50 | [0,1] s | 9.08 | 4.07e-2 | 30 000 | 960 / NS | App. A, Fig. A.20 |
| CF-U-M1 | cantilever, L = 4 | fixed–free | none | [0,1] s | 1.53 | 3.07e-5 | 70 000 | 640 / NS | App. B, Fig. B.22 |
| CF-D-M1 | cantilever | fixed–free | b = 5 (γ = 0.708) | [0,5] s | 7.65 | 7.20e-4 | 70 000 | 640 / NS | App. B, Fig. B.23 |
| CF-D-INV | cantilever | fixed–free | identify b | [0,5] s | — | b err 1.41 %, field 6.35e-2 | 250 000 | 3200 / 128 | §5.1.2 |

The known numbers 4.64e-4 (FE-D-M1) and 2.30e-3 (SS-U-M1) belong to **different** beams, BCs, damping, window frequency (20.6 vs 9.1 cycles), batch size (640 vs 960) and epoch budget (45k vs 30k). They are not interchangeable.

All paper RMSE values are **PINN vs Abaqus FEA at one point, in metres, at an unstated load F**. They cannot be reproduced and are not comparable. Only relative L2 (Eq. 44, vs the analytical solution) is a usable target.

### C.1 Verified constants (float64)

| Quantity | Computed | Paper |
|---|---|---|
| c² = EI/ρA | 1912.046 | prints 43.73² = 1912.313 |
| FE β₁l (root of cos·cosh = 1) | 4.730040745 | 4.7300 |
| FE f₁ | 20.5889 Hz | 20.594 Hz |
| SS f₁ | 9.0825 Hz | 9.085 Hz |
| CF β₁l / f₁ (L = 4) | 1.875104 / 1.5293 Hz | 1.8751 / 1.529 Hz |
| γ = b/ρA (b = 50) | 7.0817 | 7.08 ✓ |
| FE damping ratio γ/(2ω₁) | 0.0274 | Table 3: 5.79e-2 ✗ |
| Envelope e^{−γt/2} at t = 1 s | 0.029 | consistent with Fig. 5(c) |

### C.2 Ambiguities in FE-D-M1 and proposed resolutions (REQUIRES APPROVAL)

| # | Ambiguity | Proposed resolution |
|---|---|---|
| A1 | Depth: §4 says 4 × 200; the selected Table 4 #12 (which *is* 4.64e-4) is 6 × 200. The 4 × 200 entry (#2) gives 5.68e-4. | Target = **4.64e-4** (the stricter number, tied to 6 × 200). 4 × 200 / 5.68e-4 is recorded as a secondary reference. |
| A2 | Fourier convention: Eqs. 38–39 have 2πB; the reference code has B on standardised inputs. | Paper-faithful baseline = **Eq. 38–39 (2π, t in seconds)**. The reference-code convention is run once in Phase A as a measured check. |
| A3 | Fourier feature count not stated. | 100 per mapping (reference code `layers[0] = 200 → 100`). |
| A4 | Rounded constants (β₁l = 4.7300, 43.73², f = 20.594 Hz) are mutually inconsistent at 1e-5…1e-4 relative frequency. | Reference = **exact solution of the trained PDE**: c² = EI/ρA from Table 3 data, exact β₁l root, γ = b/ρA. See C.4. |
| A5 | Table 3 ζ = 5.79e-2 ≠ γ/(2ω₁) = 0.0274. | Use the explicit PDE coefficient 7.08 (b = 50). |
| A6 | Does the forward problem use interior analytical data? Eq. 31 says L_u is "IC and BC data"; §5 says synthetic data "are given to the PINN as training data"; a validation loss is mentioned. | **Pure physics** (IC + BC + PDE, no interior labels). If the paper used interior data, this makes our comparison conservative. |
| A7 | Evaluation grid not stated. | Uniform **201 (x) × 2001 (t)** grid including endpoints (97 samples per period), float64 reference. |
| A8 | Epoch semantics. | Per epoch, draw 640 points per term, shuffle, split into 20 mini-batches of 32 ⇒ 20 steps/epoch. Paper budget = **9.0e5 steps, 2.88e7 PDE residual evaluations**. |
| A9 | LR schedule and NTK update interval not stated. | Paper-faithful: constant lr 1e-4; exact-trace NTK on the current batch every 100 steps (reference code), no EMA. |
| A10 | IC: Eq. 27 is a static deflection, which carries **3.1 %** non-mode-1 content (L2). The reference (Eq. 28) is mode 1 only. | IC = mode-1 projection, scaled to mid-span 0.08 m (Fig. 5c; rel-L2 is amplitude-invariant). |
| A11 | σ order: §4 says σ_t = (1, 10); Table 4 #12 says (10, 1). | The same set; order is irrelevant under a shared trunk. |

If any of A1–A11 is rejected, the target cannot be mapped confidently and Stage 1 must not start.

### C.3 Why the self-consistent reference matters (measured, `stage0_benchmark_numerics.txt`)

Relative-L2 floor caused **only** by a reference/PDE frequency mismatch δ:

| Case | Mismatch source | δ | rel-L2 floor | Paper result |
|---|---|---|---|---|
| FE-D | β₁l 4.7300 vs exact | −1.7e-5 | **4.39e-4** | 4.64e-4 |
| FE-D | 43.73² vs EI/ρA | +7.0e-5 | 1.78e-3 | 4.64e-4 |
| FE-U | β₁l 4.7300 vs exact | −1.7e-5 | 1.28e-3 | 2.70e-3 |
| SS-U | 43.73² vs EI/ρA | +7.0e-5 | **2.27e-3** | 2.3e-3 |

Consequences:
1. The targets sit at the same order as floors created by the paper's own rounding. The paper's reference and PDE must have been consistent to |δ| ≲ 2e-5, or its numbers are partly floor-limited. Which one is unknowable from the PDF. The SS-U coincidence (2.27e-3 vs 2.3e-3) is noted as a **hypothesis only**.
2. Reaching L2 < 4.64e-4 on FE-D needs a learned frequency correct to ≲ 1.5e-5 relative, i.e. ≈ 3e-4 Hz.
3. FE-D rel-L2 is early-time weighted: the envelope falls to 2.9 % by t = 1 s. A **late-window rel-L2 on t ∈ [0.5, 1] s** is reported as a secondary metric so late-time failure cannot hide.

---

## D. Canonical first-mode benchmark (PROPOSED)

**`CANONICAL_FIRST_MODE_BENCHMARK = FE-D-M1-v1`**

- PDE: `EI u_xxxx + ρA u_tt + b u_t = 0`, EI = 13 500 N·m², ρA = 7.0605 kg/m, b = 50 N·s/m
- BCs: `u = u_x = 0` at x = 0 and x = 2.75 m
- IC: `u(x,0) = A₀ U₁(x)/U₁(L/2)`, `u_t(x,0) = 0`, A₀ = 0.08 m,
  `U₁ = cosh βx − cos βx − σ(sinh βx − sin βx)`, β = 4.730040745/L
- Reference: `u = A₀ U₁(x)/U₁(L/2) · e^{−γt/2}[cos ω_d t + γ/(2ω_d) sin ω_d t]`, ω₁ = β²c, ω_d = √(ω₁² − γ²/4), float64
- Domain: x ∈ [0, 2.75] m, t ∈ [0, 1] s
- Normalization: a **method** choice, recorded per run. Evaluation is always in physical coordinates.
- Eval grid: 201 × 2001 uniform. Metric: Eq. 44 rel-L2. Secondary metrics: late-window rel-L2 and the C.2 diagnostics.
- Training information: physics only.

**Why FE-D-M1 and not SS-U-M1**
(i) It is the paper's headline main-text result.
(ii) It is the only case with a full hyperparameter specification (Table 4 #12).
(iii) The paper ran its method ladder (Table 5) and its mini-batch study (Fig. 6) on it, so Phases A and E can be compared against published numbers on the *same* benchmark.
(iv) It is the hardest frequency in the paper (20.6 cycles), so a pass is meaningful.

Costs: fixed-fixed physics must be implemented (F6), and the target is near the rounding floors (C.3). **Alternative:** SS-U-M1 (2.3e-3, code exists, cheaper to bring up) has its own problems: a typo'd BC, hyperparameters given only "as Section 5.1.1", and a target that coincides with a rounding floor.

## E. Exact paper target

**OUR_L2 < 4.64 × 10⁻⁴** on FE-D-M1-v1 (rel-L2, 201 × 2001 grid, vs the self-consistent analytical reference).
Paper method at that number: 6 × 200 tanh, spatio-temporal Fourier (σ_x = 1; σ_t = 10, 1), NTK weights, Adam lr 1e-4, batch 640, mini-batch 32, 45 000 epochs ⇒ **9.0e5 optimizer steps, 2.88e7 PDE residual evaluations, ≈ 241 600 parameters** (with m = 100).
Secondary published references on the same benchmark: 5.68e-4 (4 × 200, T4 #2); ladder 1.00 / 1.11 / 0.881 (T5); mini-batch 7.32e-1 / 6.18e-1 / 6.57e-2 / 1.46e-3 / 4.64e-4 (F6).

## F. Current best baseline

**There is no CURRENT_BEST on FE-D-M1**: the repository has never solved a fixed-fixed beam. `optimization_leaderboard.csv` is created empty. The nearest legacy evidence is `P_paper_window` = 0.967 on SS-U-M1 (paper 2.3e-3), confounded by F1 and F8. CURRENT_BEST will be the Phase-A paper-faithful C0 run.

---

## G. Computational bottlenecks (measured, `stage0_step_profile.txt`)

Forward + backward of the full loss (IC, velocity IC, two BC terms, PDE), N points per term:

| Net | Params | N = 32 | N = 128 | N = 640 | One NTK update (16 rows × 5 terms) |
|---|---|---|---|---|---|
| 3 × 64, m = 64 | 16 705 | 18–24 ms | 25–33 ms | 59–133 ms | 0.2–0.3 s |
| 4 × 200, m = 64 | 146 801 | 33–39 ms | 75–78 ms | 170–183 ms | 0.8–0.9 s |
| 4 × 200, m = 100 | 161 201 | 41–55 ms | 89–99 ms | 214–216 ms | 0.7–0.8 s |
| 6 × 200, m = 100 | 241 601 | 53–60 ms | 106–125 ms | 254–266 ms | 1.2 s |

Findings:
1. **Mini-batch 32 is overhead-dominated on CPU.** 20× more points cost only 4.4–5× more time, so per point it is ≈ 4× less efficient than N = 640. The paper's 9e5-step budget is ≈ **14 h (6 × 200) / 12 h (4 × 200)** of pure stepping here, plus NTK.
2. **NTK overhead**: one update ≈ 1.2 s (6 × 200). At the reference cadence (every 100 steps) that is +12 ms/step, i.e. **+20–25 % wall-clock** at mini-batch 32. It is a training-only cost.
3. **The PDE term dominates**: PDE-only fwd+bwd (6 × 200, N = 32) is 47–48 ms of the 53–60 ms full step, i.e. ≈ 80 %.
4. **Rejected on evidence:** recomputing the derivatives per point with forward-mode `jacfwd` on the separable x-branch is exact (max rel diff ≤ 1.1e-6) but **0.24–0.47× the speed** of nested reverse mode. Not pursued. A tensor-product-grid evaluation (SPINN-style, Cho et al. 2023 — not yet verified against the literature) remains an **untested** idea and is not in the matrix.
5. Long runs exceed one interactive session. The container is reclaimed after inactivity, so every run longer than ~1 h needs periodic checkpoint and resume.

## H. Required code changes (before any screening; reuse legacy code verbatim where correct)

1. **`src/` package** (`physics/`, `models/`, `sampling/`, `losses/`, `optimization/`, `training/`, `evaluation/`, `profiling/`, `utils/`) extracted from `notebook_src`. The legacy notebook and `results/` stay untouched.
2. **Physics abstraction** for BC type {fixed-fixed, simply-supported, cantilever} with damping. Exact float64 references (root-found β), mode shapes, derivative-BC residuals. Tests: PDE residual ≈ 0 with damping, BCs, ICs, f₁ vs paper.
3. **Fourier options** `two_pi ∈ {True, False}`, `input_norm ∈ {physical, unit, standardize}`, `m`; plus the target-frequency vs feature-support diagnostic.
4. **Paper-semantics sampler**: fixed per-epoch dataset, mini-batch iteration, equal N per term. Counters for steps, points, PDE evaluations, gradient evaluations and candidate evaluations.
5. **NTK paper-faithful mode** (equal-N exact trace on the batch, update interval k, optional EMA), vectorised per-sample gradients (`torch.func.vmap(grad)`), validated against the legacy estimator. Overhead accounted separately.
6. **Metrics**: 201 × 2001 grid; late-window L2; a damped-sinusoid least-squares fit at mid-span for frequency, phase, amplitude and damping; BC errors for each condition; IC errors; dimensionless PDE residual RMS(r)/RMS(ρA u_tt); peak RSS RAM; parameters; size; inference latency. **Fixes F2–F4 and F7.**
7. **Run records** under `results_optimization/` (config.json, metrics.json, history.csv, hardware.json, checkpoint), with resume support.
8. New tests: Fourier convention, damped/FE/CF analytics, float32 network 4th derivative at ω = 130 rad/s, NTK vectorised = loop, sampler accounting.
9. Later phases only: RAD/RAR-D, Annealing, GradNorm, hard-constraint transforms, mixed form, L-BFGS.

**Hard-constraint derivation preview (C7, FE-D).** Let `u₀(x) = A₀U₁(x)/U₁(L/2)` and `Φ(x) = x²(L−x)²/L⁴`. Define

`u_θ(x,t) = u₀(x) + t² Φ(x) N_θ(x,t)`.

- **BCs.** u₀ satisfies u₀ = u₀′ = 0 at both ends. Φ has a double root at 0 and L, so `(ΦN)′ = Φ′N + ΦN′ = 0` there for any N. Hence u = u_x = 0 exactly.
- **ICs.** At t = 0 the t² factor gives u = u₀. `u_t = 2tΦN + t²ΦN_t = 0` at t = 0.

All four conditions hold exactly, so only L_pde remains. That leaves NTK, Annealing and GradNorm **nothing to balance**: this is a mechanistic prediction to test. Using u₀ as the base is a problem-specific choice (it uses the known IC shape) and is classified as such.

For simply-supported BCs, `Φ = x(L−x)` does **not** enforce u_xx = 0, and `x³(L−x)³` over-constrains (it forces u_x = 0, which the true solution violates). A moment-free construction (odd extension / sine embedding) must be derived in Phase D if SS is needed.

---

## I. First-mode screening matrix

All on FE-D-M1-v1, seed 1234 for screening, 6 × 200 / m = 100 unless the column varies, mini-batch 32, lr 1e-4, float32, 4 threads. **One change per row** relative to the stated parent.

Funnel budgets (steps at mini-batch 32):
- **S0** = 200 steps (sanity)
- **S1** = 20 000 steps (2.2 % of paper; 6.4e5 PDE evaluations)
- **S2** = 100 000 steps (11 %)
- **CONF** = up to 9.0e5 steps (paper budget) × ≥ 3 seeds

| Phase | ID | Configuration | Parent | Single change | Budget | Decision metric |
|---|---|---|---|---|---|---|
| A | B-A | Vanilla PINN, soft IC/BC, λ = 1 | — | — | S1 | ladder check vs T5 (1.11) |
| A | B-B | Fourier (Eq. 38–39), λ = 1 | B-A | + Fourier | S1 | — |
| A | **C0** | Fourier + NTK (paper-faithful) | B-B | + NTK | S1 → S2 | becomes CURRENT_BEST |
| A | C0-rc | C0 with reference-code input convention | C0 | 2π/standardise | S1 | resolves A2/F1 on evidence |
| B | C1 | Fourier + RAD | B-B | + RAD | S1 | vs B-B |
| B | C2 | Fourier + RAR-D | B-B | + RAR-D | S1 | vs B-B |
| B | C1′ / C2′ | C0 + RAD / C0 + RAR-D *(proposed addition)* | C0 | + sampler | S1 | vs C0 |
| C | C3 | Fourier + Annealing (EMA) | B-B | + Annealing | S1 | vs B-B and C0 |
| C | C4 | Fourier + GradNorm | B-B | + GradNorm | S1 | vs B-B and C0 |
| D | C7 | Hard-constraint Fourier (PDE loss only) | B-B | hard IC/BC | S1 | vs C0 |
| E | E16/E32/E64/E128 | best of A–D, mini-batch 16/32/64/128 at **matched PDE evaluations** (6.4e5) | best | batch | S1 | L2 vs wall-clock |
| F | F-2×64 … F-4×200, 6×200 | best, width × depth ∈ {2×64, 3×64, 3×128, 4×64, 4×128, 4×200} | best | arch | S1 | min params with no S1 regression |
| G | C5/C6/C8–C10/C13/C14 | only pairs whose parts each passed independently | — | pair | S2 | vs CURRENT_BEST |
| H | C11/C12 | mixed form m = u_xx | best | formulation | profile → S1 | reject if cost ↑ without accuracy ↑ |
| I | — | best + short L-BFGS refinement | best S2 ckpt | optimizer | +2k L-BFGS iterations | gain / extra cost |
| J | — | weak/VPINN, decomposition, distillation (C15) | — | — | **not scheduled**; only if A–I fail | — |
| — | causal | **not run** (rejected, negative baseline) | — | — | — | — |

Every run logs all of §10's metrics, gradient norms and weights for every loss term, and the point distributions for the samplers.

## J. Pass/fail gates (proposed thresholds, require approval)

- **Kill at S1** (any one triggers):
  - NaN/Inf;
  - best-so-far L2 > 1.25 × parent at **equal wall-clock** *and* a flatter log-L2 slope over the final 25 %;
  - sampler or weighting overhead > 50 % of step time with < 20 % L2 gain;
  - final L2 > 2 × best-so-far (instability).
  The single-seed noise margin is assumed ~20 % until Phase A measures it with 3 seeds of C0 at S1.
- **Promote S1 → S2**: L2 ≤ 0.8 × parent at equal wall-clock, or a steeper slope with no kill condition.
- **Level 0 FAIL**: L2 ≥ 4.64e-4. **Level 1**: L2 < 4.64e-4 but steps > 9.0e5 *or* PDE evaluations > 2.88e7 *or* wall-clock > 1.1 × C0 at the same steps. **Level 2**: L2 < 4.64e-4 within the paper budget *and* wall-clock ≤ 1.1 × C0.
- **Level 3 CONFIRMED** = Level 2 on **every one of ≥ 3 seeds** (report mean, median, sd, min, max, 95 % t-CI), plus all of the following:
  - late-window L2 < 5e-3;
  - |Δf|/f ≤ 2e-5;
  - phase error ≤ 2e-3 rad;
  - amplitude error ≤ 1e-3;
  - max|u(0,t)|, max|u(L,t)| ≤ 1e-4 A₀ and max|u_x|·L ≤ 1e-3 A₀;
  - max|u(x,0) − u₀| ≤ 5e-4 A₀ and max|u_t(x,0)| ≤ 1e-3 ω₁A₀;
  - RMS(r)/RMS(ρA u_tt) ≤ 1e-2;
  - no NaN, and a rerun of one seed reproduces L2 within 5 %.
- **Do not leave first mode** until Level 3. If C0 at S2 shows the paper-faithful method cannot approach the target within the budget, **STOP and report** before spending CONF compute.

## K. Expected computational cost (this 4-core CPU; ±15 % timing jitter)

| Item | Runs | Est. wall-clock |
|---|---|---|
| Code changes (H1–H8) + tests | — | engineering, no training |
| S0 sanity (all configs) | ~20 | < 10 min total |
| Phase A (B-A, B-B, C0, C0-rc at S1; C0 at S2; 2 extra C0 seeds at S1 for the noise floor) | 7 | ≈ 4 h |
| Phase B (C1, C2, C1′, C2′ at S1) | 4 | ≈ 1.5–2 h (+ sampler overhead, measured) |
| Phase C (C3, C4) | 2 | ≈ 0.8 h |
| Phase D (C7) | 1–2 | ≈ 0.5 h |
| Phase E (4 mini-batch sizes, matched PDE evaluations) | 4 | ≈ 1.5 h |
| Phase F (6 architectures, cheaper than 6 × 200) | 6 | ≈ 1.5 h |
| Phase G (≤ 3 combinations at S2) | ≤ 3 | ≈ 5 h |
| Phases H, I | 2–3 | ≈ 1–2 h |
| **Screening subtotal** | ~30 | **≈ 15–17 h** |
| CONF: ≤ 9e5 steps × 3 seeds (6 × 200 incl. NTK ≈ 17 h/seed; a compact or hard-constrained net could be several times cheaper) | 3 | **≤ 50 h** |
| **Stage 1–4 worst case** | | **≈ 65 h CPU** |

A GPU environment would shorten CONF substantially (not measured here). Parallel single-thread runs may give better throughput than one 4-thread run at mini-batch 32; this will be measured in Phase A before committing CONF compute.

## L. Proposed execution order

1. **(Needs approval)** Accept C.2 resolutions A1–A11, the canonical benchmark (D), the target (E), the gates (J) and the budget (K).
2. Implement H1–H8. Run the full test suite. Stage 0.1 is complete only when the tests pass and F1–F7 are fixed or made configurable.
3. Phase A: S0 sanity → B-A, B-B, C0, C0-rc at S1 → C0 seeds 2–3 at S1 (noise floor) → C0 at S2 → **report and checkpoint decision** (is the target reachable? which convention?).
4. Phases B → C → D → E → F, each one-change-at-a-time with a research-log entry (§50 format) and a leaderboard row. Stop branches by the J kill rules.
5. Phase G combinations only from independently validated mechanisms. Then H and I if justified.
6. Confirmation (CONF, ≥ 3 seeds) of the single best candidate, then the first-mode gate decision. **STOP and report.**
7. Stages 5–8 (higher modes, full replication, freeze, presentation) are not started without a passed first-mode gate and separate approval.

**STOPPED HERE. No training has been run.**
