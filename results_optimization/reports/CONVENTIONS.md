# Conventions register (single source of truth)

Every run records its convention in `configs/<run_id>.json` (`model.two_pi`, `model.input_norm`,
`benchmark.pde_coeffs`, `benchmark.ic_shape`). This file states which convention is used where.

## Fourier features and input normalization

| Use | `two_pi` | `input_norm` | Feature map | Source |
|---|---|---|---|---|
| **Paper reproduction (C0, B-A, B-B)** | `True` | `physical` | `[cos(2πBv), sin(2πBv)]`, v = x [m] or t [s] | Paper Eqs. 38–39 (confirmed on the rendered PDF) |
| **C0-rc convention check** | `False` | `standardize` | `[cos(Bv), sin(Bv)]`, v = (v − mean)/std of U[0, L] or U[0, T] | Reference code `Wave1D_NTK_ST_mFF` (MultiscalePINNs), which the paper says it adapted |
| Legacy repository (not used for new runs) | `False` | `unit` | `[cos(Bv), sin(Bv)]`, v ∈ [0, 1] | `notebook_src/sec_b.py:730` |
| **Optimized method** | **TBD by evidence** | **TBD** | One of the rows above. It is fixed after C0 vs C0-rc, recorded in `final_method_config.json`, and never changed silently | — |

The two conventions differ only in that one flag pair; C0 and C0-rc are otherwise identical.

Common to all rows:
- B ~ N(0, σ²), where σ is the **standard deviation** (as in the reference code).
- B is drawn once from a seeded generator and frozen.
- m = 100 features per mapping.
- Mappings: σ_x = (1), σ_t = (10, 1) (Table 4 #12).

Frequency support measured on the actual seeded draws (`tables/stage01_fourier_support_FE-D-M1.csv`), against the temporal target ω_d = 129.3 rad/s:

| Convention | σ_t = 10: max feature | Features ≥ ω_d |
|---|---|---|
| Paper | 168 rad/s | **3/100** |
| Reference code | 93 rad/s | 0/100 |
| Legacy | 27 rad/s | 0/100 |

## Coordinates, PDE and references

- **Derivatives are always taken with respect to physical x [m] and t [s].** Any normalization is a fixed layer inside the model, so the residual is always paper Eq. 49 in physical units: `43.73² u_xxxx + u_tt + 7.08 u_t` (`benchmark.pde_coeffs = paper_eq49`). The output is in metres (`output_scale = 1`).
- **Evaluation is convention-independent.** Every model is evaluated on the same physical 201 × 2001 grid, against both references:
  - `L2_paper`: the paper-faithful reference, Eq. 28 with the printed β₁l = 4.7300 and the coefficients 43.73² and 7.08. **This is the primary gate.**
  - `L2_exact`: the exact-physics reference, the exact solution of the trained PDE (exact root 4.730040745, same coefficients).
  - The `material` reference (coefficients EI/ρA and b/ρA from Table 3) is diagnostic only.
- **IC target** (`benchmark.ic_shape = exact`): the exact-root mode shape × 0.08 m. Its distance from the paper-shape IC is a measured 1.7e-5 rel-L2.
- **Frequency, phase, amplitude and damping errors** come from a damped-cosine least-squares fit at mid-span. They are compared with the exact parameters of each reference. They are **diagnostics, not gates**: the extractor's p95 resolution at target-level structured error is about 1.4e-5 (see the Stage 0.1 report).
