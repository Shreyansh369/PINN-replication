# Final presentation: summary, sources and claims

**Deck:** `final_scientific_conference_presentation.pptx` (8 slides, 16:9, speaker notes on every slide)
**Built by:** `presentation_build/build_deck.js` from frozen, committed artifacts only. No training or optimization was run for the presentation, and no result was changed.
**Consistency check:** `presentation_build/check_consistency.py`. It recomputes 48 displayed numbers from their source files and checks scope and wording. Output: `presentation_tables/consistency_check.txt` (**ALL CHECKS PASSED**).

**Central conclusion:** We performed a controlled reproducibility and optimization investigation of a Fourier/NTK PINN for damped beam vibration, identified conditioning and optimization bottlenecks, verified that the network can represent the target dynamics, and quantified how training strategy and physics-computation budget affect late-time dynamic persistence. A fully validated optimized PINN was not established within the tested budget.

Path conventions: `RO/` = `results_optimization/`. Every result is for benchmark FE-D-M1 (fixed–fixed beam, Mode 1), seed 1234, float32, one CPU thread, unless stated otherwise.

---

## 1. Slide-by-slide purpose

| # | Title | Purpose | Visual |
|---|---|---|---|
| 1 | Reproducibility and Optimization of Physics-Informed Neural Networks for High-Frequency Beam Vibration | Title, scope and framing | Analytical mid-span reference signal (native chart); fixed–fixed beam with Mode-1 shape |
| 2 | A damped, high-frequency beam transient | The physical problem and the target paper. The paper's 4.64e-4 is labelled **PUBLISHED BASELINE**. Fourier + NTK is stated to be the paper's method, not ours. | Beam schematic, PDE card, baseline card |
| 3 | A controlled, one-change-at-a-time workflow | The audit workflow (8 steps) and the five verified audit findings | Process flow, numbered findings |
| 4 | The faithful baseline settles on a near-static field | The first failure: the paper-faithful Fourier + NTK run (C0) gives a static/low-frequency field while NTK weights reach 1e14–1e15. Presented as an **observed** optimization failure, not as proof of causation. | Frozen mid-span trace (crop); native log-scale NTK-weight chart |
| 5 | The bottleneck is optimization, not capacity | Conditioning of the hard-constraint time factor, and the representation check (supervised fits) | Ansatz comparison cards; supervised trace (crop); table |
| 6 | Later collapse, but never the full window | Optimization progression E → X → Y → Z, with one representative trace | Progression table; Z4-20K callouts; frozen trace (crop) |
| 7 | More physics computation, longer persistence | Collapse time vs cumulative PDE evaluations; why batch size cannot be isolated | Native scatter chart; checkpoint table |
| 8 | What is established — and what is not | ESTABLISHED / NOT ESTABLISHED / NEXT; railway pathway shown as **future work only** | Three cards; future-work flow |

## 2. Every numerical result used, with its source

### Slide 2: problem and published baseline

| Value on slide | Meaning | Source |
|---|---|---|
| 4.64 × 10⁻⁴ | Relative L2 **reported by the paper** (published value, not ours) | Söyleyici & Ünver, EAAI 141 (2025) 109804; `paper_benchmark_registry.csv` (FE-D-M1) |
| c² = 43.73², γ = 7.08 | PDE coefficients (paper Eq. 49) | Paper; `src/beampinn/physics/benchmarks.py` |
| L = 2.75 m, A₀ = 0.08 m, t ∈ [0, 1] s | Geometry, amplitude, window | Paper; `benchmarks.py` |
| 6 × 200 tanh; σ_x = 1; σ_t = 10, 1; Adam lr 1e-4; 45 000 epochs | Paper method | Paper; `RO/reports/STAGE0_AUDIT.md` |
| 20.6 Hz, ω_d = 129.32 rad/s, 20.6 cycles | Analytical Mode-1 damped frequency (20.59 Hz) | `benchmarks.py` (exact reference); `RO/tables/phaseY_oscillation.csv` `w_exact` |
| 3.54 s⁻¹ | Physical decay rate | `RO/tables/phaseY_oscillation.csv` `decay_exact` |

### Slide 3: audit

| Value | Meaning | Source |
|---|---|---|
| 2π omitted | Legacy Fourier convention (paper Eq. 38–39) | `RO/reports/STAGE0_AUDIT.md` (F1) |
| f = mode², 1 Hz FFT bins | Legacy frequency-metric defects | `STAGE0_AUDIT.md` (F2, F3) |
| 81 × 81 → 201 × 2001 | Sparse legacy grid → evaluation grid | `STAGE0_AUDIT.md` (F4); `src/beampinn/evaluation/metrics.py` |
| Standardised inputs, no 2π, N(0,1) biases, decaying LR | Reference-code differences | `STAGE0_AUDIT.md`; `RO/reports/CONVENTIONS.md` |
| 4.7300 vs exact root; 4.386e-4; 94.5 % | Paper-faithful vs exact reference: L2 between them, and as a fraction of 4.64e-4 | `RO/tables/stage01_reference_comparison.csv` (201×2001, exact vs paper); `RO/reports/STAGE01_CORRECTIONS.md` |
| ≈ 1.4e-5 (notes) | Frequency-extractor p95 error at ε = 4.64e-4 | `RO/tables/stage01_frequency_extractor_uncertainty.csv` |

### Slide 4: first failure (C0, Phase A1, 20 000 steps, 6.4e5 PDE evaluations)

| Value | Meaning | Source |
|---|---|---|
| 3.26 | Relative L2 (exact = paper reference) | `RO/tables/phaseA_summary.csv` (C0_paper) |
| 0.06 | Amplitude ratio PINN/exact (0.0593) | `RO/tables/phaseD_oscillation.csv` (C0_paper); `PHASE_Y_OPTIMIZER_DIAGNOSTIC.md` |
| 8.5e14, 9.9e14 | Final λ_ut and λ_ux (λ_f = 1) | `RO/logs/C0_paper__s1234__c41ccb1cdf/history.csv` (last row) |
| ~10¹⁴–10¹⁵ | NTK weight range | same, and `RO/reports/PHASE_A1_CHECKPOINT.md` |
| 0 cycles | No oscillation (fitted ω ≈ 0) | `PHASE_A1_CHECKPOINT.md` |
| 21.7 (notes) | Relative PDE residual | `phaseA_summary.csv` `PDE_residual_rel` |
| NTK chart | λ_u, λ_ut, λ_ux every 1 000 steps | `history.csv` → `presentation_tables/deck_data.json` |
| Trace | C0 mid-span, 0–0.15 s | crop of `RO/figures/phaseA_A1_C0_paper_diagnostics.png` → `presentation_figures/fig_C0_midspan_static.png` |

### Slide 5: conditioning and representation

| Value | Meaning | Source |
|---|---|---|
| ≈ −8 400 (range −8 369 to −1.0) | Required N for g = (t/T)² | `RO/profiles/phaseE_ansatz_conditioning.txt` |
| O(1), −1.9 to −0.16 | Required N for g = tanh²(ω₁t) | same |
| ω₁ = 129.37 rad/s | From the PDE coefficient + clamped-beam eigenproblem | `RO/reports/PHASE_X_DIAGNOSTIC.md`; `benchmarks.py` `fundamental_omega()` |
| IC error 4.8e-6, BC error 1.4e-5 | Hard constraints exact by construction | `RO/reports/PHASE_E_SCREEN.md`, `PHASE_X_DIAGNOSTIC.md` |
| 7.2e3 / 1.5e8 | PDE-gradient norm early/late, E4 | `RO/tables/phaseX_conditioning.csv` |
| 8.4e5 / 9.8e4 | PDE-gradient norm early/late, X2 | same |
| 0.463, 129.25 rad/s | X3 supervised (paper conv.): L2, fitted ω | `RO/tables/phaseX_oscillation.csv` |
| 0.316, 129.37 rad/s | X4 supervised (ref.-code conv.): L2, fitted ω | same |
| 129.32 rad/s | Exact damped ω | same (`w_exact`) |
| 0.909, 112.2 rad/s (notes) | X2 (PDE-trained, tanh²): L2, fitted ω | same |
| Trace | X3 vs exact over 1 s | crop of `RO/figures/phaseX_midspan_traces.png` → `presentation_figures/fig_X3_supervised_trace.png` |

### Slide 6: optimization progression (5 000 steps unless noted)

| Row | Rel. L2 | Persistence | Source |
|---|---|---|---|
| E4: hard constraints, (t/T)² | 2.83 | no oscillation | `RO/tables/phaseE_oscillation.csv`; `PHASE_E_SCREEN.md` |
| X2: tanh²(ω₁t), ref.-code conv. | 0.909 | ω 112.2, over-damped | `phaseX_oscillation.csv` |
| Y1: + LR schedule 1e-3·0.9^(k/1000) | 0.774 | 1.2 cycles | `phaseY_oscillation.csv`; cycles from `phaseY1_20K_checkpoints.csv` (5k = Y1, bit-identical) |
| Z1: Y1 + RAD | 0.718 | 1.6 cycles | `phaseZ_oscillation.csv`; `PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md` |
| Z4: Y1 with mini-batch 128 | 0.526 | 3.2 cycles | `phaseZ_oscillation.csv`; `phaseZ4_20K_checkpoints.csv` (5k) |
| Z4-20K: Z4 with 20 000 steps | 0.263 | 8.1 cycles | `RO/tables/phaseZ4_20K_checkpoints.csv` (20k) |

Callouts for Z4-20K at 20k: frequency **129.2 rad/s** (exact 129.32); fitted decay **5.40 s⁻¹** (exact 3.54); collapse at **0.393 s**; **8.1 / 20.6** cycles. All from `phaseZ4_20K_checkpoints.csv` and `PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md`. "≈ 570× the published 4.64e-4" (notes) is from the same report.
Trace: crop (20k panel) of `RO/figures/phaseZ4_20K_disp_traces.png` → `presentation_figures/fig_Z4_20K_step20k_trace.png`.

**Note on cycle counts.** The slide-6 brief gave approximate values (~2 / ~4 / ~8 cycles). The deck uses the frozen persistence metric instead, as instructed ("use the exact reported values"): **1.2 / 3.2 / 8.1 cycles**. The metric is the first time the local amplitude falls below 50 % of exact, × f_d (`experiments/budget_diagnostic.py`).

### Slide 7: compute budget

| Checkpoint | PDE evaluations (cum.) | Collapse time | Cycles | Rel. L2 | Train time | Source |
|---|---|---|---|---|---|---|
| Y1 5k (mb 32) | 160k | 0.057 s | 1.2 | 0.774 | 268 s | `RO/tables/phaseY1_20K_checkpoints.csv` |
| Y1 10k / 15k (chart) | 320k / 480k | 0.082 / 0.104 s | 1.7 / 2.1 | 0.665 / 0.640 | — | same |
| Y1 20k | 640k | 0.151 s | 3.1 | 0.550 | 1 078 s | same |
| Z4 5k (mb 128) | 640k | 0.153 s | 3.2 | 0.526 | 611 s (original Z4 run) | `phaseZ4_20K_checkpoints.csv`; `phaseZ_oscillation.csv` |
| Z4 10k / 15k (chart) | 1.28M / 1.92M | 0.249 / 0.346 s | 5.1 / 7.1 | 0.389 / 0.290 | — | `phaseZ4_20K_checkpoints.csv` |
| Z4 20k | 2.56M | 0.393 s | 8.1 | 0.263 | 2 290 s | same |

Collapse-time increments per 6.4e5 evaluations on the Z4 recipe: +0.096 s, +0.097 s, +0.047 s. Source: `PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md`; recomputed from the table. Wall-clock ratio 611 s / 1 078 s = 0.57×.

### Slide 8: conclusion

No new numbers. The notes repeat Z4-20K (8.1 of 20.6 cycles, L2 0.263) from slide 6.

### Slide 1: visual

The orange signal and the beam shape are the **analytical reference** (Eq. 28 with the exact root): `presentation_tables/deck_data.json` keys `exact` and `mode_shape`. They are labelled on the slide as the analytical reference, not as a network output.

## 3. Final scientific claims (as presented)

**Established (within the tested configurations and budgets):**
1. The published Fourier + NTK baseline did not reproduce the target dynamics in our controlled implementation. C0: L2 3.26, static/low-frequency field, NTK weights 1e14–1e15. Repairs D1–D4, Y4 (+LR schedule) and Z3 (+RAD) were also static or non-oscillating.
2. The original hard-constraint time factor (t/T)² was poorly conditioned (required network output ≈ −8 400). tanh²(ω₁t) needs O(1).
3. The network can represent the 20.6 Hz solution: supervised X3/X4 recover ω = 129.25 / 129.37 rad/s.
4. LR schedule, batch size and physics-computation budget materially affect optimization: Y1 → Z4 → Z4-20K moves the collapse from 1.2 to 8.1 cycles.
5. RAD did not resolve the observed failure (Z1: 1.6 cycles).

**Not established:**
- No validated optimized PINN.
- No claim of superiority over the paper. Our best measured L2 of 0.263 is roughly 570× the published 4.64e-4.
- No full-window Mode-1 reproduction (best: 8.1 of 20.6 cycles; still over-damped, 5.40 vs 3.54 s⁻¹).
- No Mode-2 generalization result. Mode 2 was not run, because no Mode-1 candidate passed the gate.
- No causal claim that NTK weighting produces the static field. The weights coincide with it; no fixed-λ ablation was run.
- No independent effect of batch size on solution quality. At equal 6.4e5 evaluations, mb 128 and mb 32 reach the same state; only the wall-clock time differs.
- No extrapolation of the persistence trend. The last block slowed to about half the previous advance.

## 4. Limitations

- Single seed (1234) for every 20k-step run. A second seed (Z2) was tested only at 5k and showed the same failure mode.
- Budgets far below the paper's: at most 2.56e6 PDE evaluations, vs 2.88e7 for the paper's 45 000 epochs. CPU only, float32.
- The persistence threshold (amplitude ratio 0.5) is a fixed but arbitrary choice, applied identically to every run. Checkpoint spacing is 5 000 steps.
- The tanh²(ω₁t) time factor uses a declared problem-specific prior (ω₁ from the PDE coefficient and the BC eigenproblem).
- The LR schedule had decayed to 1.2e-4 by 20k steps. It may contribute to the slowdown; this was not tested.
- Timing comparisons mix 4-concurrent and solo runs. Phase A measured no per-step effect, within the documented ±20 % jitter.
- Only Mode 1 of the fixed–fixed damped case was studied. The other benchmark rows in `paper_benchmark_registry.csv` were not trained.
- Author and institution on slide 1 are **placeholders** (`[Author name]`, `[Institution]`) to be filled in before presenting.

## 5. Proposed next research stage

1. **Mode 1 first:** find a genuinely successful Mode-1 candidate (full-window persistence and the L2 gate), then confirm it independently with more seeds and with the exact-physics and paper-faithful references. Candidates for that search include a matched small-batch run at equal evaluations, larger budgets, an LR-floor variant, and a fixed-λ NTK ablation. Each would be a separate, approval-gated experiment.
2. **Higher frequency:** Mode-2 validation of a frozen Mode-1 method, only after step 1 succeeds.
3. **Future work only:** extension toward railway vibration / SHM physics (beam PINN → railway structural/component dynamics → sensor-informed PINN → axle/bearing vibration or AE → physics-constrained fault estimation). No railway experiment was performed, and no railway result is presented or claimed.

## 6. Files

- `final_scientific_conference_presentation.pptx`: the deck.
- `presentation_build/build_deck.js`: deck generator (pptxgenjs). Run with `NODE_PATH` pointing at a pptxgenjs install; argument 1 is the pptx skill's `apply_theme.js`.
- `presentation_build/check_consistency.py`: number, scope and wording check (read-only).
- `presentation_figures/`: crops and copies of frozen figures (C0 trace, X3/X4 supervised traces, Z4-20K traces). No new plots.
- `presentation_tables/deck_data.json`: chart data extracted from frozen CSVs, plus the analytical reference curves.
- `presentation_tables/consistency_check.txt`: output of the consistency check.
