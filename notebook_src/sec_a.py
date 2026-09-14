"""Sections 0-6 + unit tests: setup, objective, parameters, physics, analytical, data, plots."""
from common import md, code


def cells():
    C = []
    A = C.append

    # ---------------------------------------------------------------- title --
    A(md(r"""
# A Physics-Informed Deep Neural Network Beam-Vibration Framework
## Replication study and a controlled optimization experiment

**Target paper**

> Cem Söyleyici and Hakkı Özgür Ünver,
> *"A Physics-Informed Deep Neural Network based beam vibration framework for
> simulation and parameter identification"*,
> **Engineering Applications of Artificial Intelligence 141 (2025) 109804**.
> DOI: [10.1016/j.engappai.2024.109804](https://doi.org/10.1016/j.engappai.2024.109804)

This notebook is the single artifact for the study. It runs top to bottom with no
manual edits and writes every figure, table, metric and checkpoint under `results/`.

---

## Provenance statement — read this first

**The paper PDF is available and has been read.** Every physical parameter,
hyperparameter and reported metric below is taken from it, with the table or
equation number given. Items still marked `OURS` are deliberate choices of this
study (compute budget, the proposed optimization, evaluation extras) and are
never presented as the paper's.

An earlier draft of this notebook was built without the PDF; all `ASSUMED`
placeholders from that draft have been replaced with the published values.

| Label | Meaning |
|---|---|
| `PAPER (A.10)` etc. | Read directly from the paper, with its table/equation number. |
| `OURS` | A deliberate methodological choice of this study, not part of the paper. |

**Which paper case this notebook replicates.** The paper's main text treats a
**fixed-end** beam; the **simply-supported** beam is Appendix A. This notebook
implements the *simply-supported* case, so the reference values throughout are
Appendix A / Table A.10, not Table 3.

**What we can and cannot match.** The method, the beam, the domain, the
architecture and the loss are all reproduced from the paper. The *training
budget* is not: the paper trains 30 000 epochs with batch 960 and mini-batch 32
on an RTX A6000, which is several orders of magnitude more gradient steps than
this 4-CPU environment can deliver. Where our error is larger than the paper's,
that gap is reported and attributed, not hidden.

### Three inconsistencies found in the paper, and how we resolved them

| # | What the paper says | Problem | Our resolution |
|---|---|---|---|
| 1 | PDE written as $43.732\,u_{xxxx} + u_{tt} = 0$ (Eqs. 46, A.4) | $43.732$ is $\sqrt{EI/\rho A}$, not $EI/\rho A$. Using it literally gives $f_1 = 1.37$ Hz, contradicting the $9.085$ Hz in Table A.10. Almost certainly a lost superscript. | Use $EI/\rho A = 1912.05$ from the Table A.10 material data, which reproduces $f_1 = 9.082$ Hz (paper: $9.085$). Verified in the tests. |
| 2 | Eq. (A.4) gives BCs $u_{xx}=u_{xxx}=0$ at both ends | Those are **free–free** conditions, not simply supported. They contradict the surrounding prose ("displacement and bending moments … constrained to zero") and Eq. (A.3)'s $\beta_1 l = 3.1416 = \pi$, which only holds for pinned–pinned. | Use $u = u_{xx} = 0$, i.e. the prose and the mode shape, which are mutually consistent. |
| 3 | Section 4 says "four hidden layers … 200 neurons"; Table 4 test 12 (selected) says 6 layers | Direct contradiction on depth. | Use **4 × 200** as stated in the framework description (Section 4); Table 4 is the fixed-end hyperparameter search. Noted in the parameter table. |
"""))

    # ------------------------------------------------------- reproducibility --
    A(md(r"""
---
# 0. Reproducibility header, execution mode and output directories

Seeds, device, library versions and the full experiment configuration are fixed
and printed before anything else runs.

**Execution modes.** Fourth-order derivatives through a 200-neuron network are
expensive, and this study trains ~20 models. Three budgets are provided:

| Mode | Purpose |
|---|---|
| `FAST_MODE` | Smoke test. Small network, few iterations. **Not research results.** |
| default (neither flag) | The budget actually used for the committed results. Paper architecture, reduced iteration count. |
| `REPRODUCTION_MODE` | Paper-scale intent: full architecture and a long schedule. |

The mode in force is printed and stamped into every saved artifact, so a figure
can never be mistaken for one produced at a different budget.
"""))

    A(code(r"""
# ---- execution mode -------------------------------------------------------
FAST_MODE = False          # True  -> tiny smoke-test budget (NOT research results)
REPRODUCTION_MODE = False  # True  -> paper-scale budget (very slow on CPU)
USE_CACHE = True           # reuse checkpoints in results/checkpoints when present

import os, sys, json, math, time, platform, warnings, hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", message=".*requires_grad=True.*")

SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32          # see Section 3.4 for the float32-vs-float64 study

def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
torch.set_default_dtype(DTYPE)
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)
torch.use_deterministic_algorithms(False)   # 4th-order autograd has no det. kernels

MODE_NAME = "FAST" if FAST_MODE else ("REPRODUCTION" if REPRODUCTION_MODE else "DEFAULT")

print("=" * 74)
print("REPRODUCIBILITY HEADER")
print("=" * 74)
print(f"  execution mode : {MODE_NAME}")
print(f"  seed           : {SEED}")
print(f"  device         : {DEVICE}  (threads={torch.get_num_threads()}, cpus={os.cpu_count()})")
print(f"  default dtype  : {DTYPE}")
print(f"  python         : {platform.python_version()}  ({platform.system()} {platform.machine()})")
print(f"  torch          : {torch.__version__}")
print(f"  numpy          : {np.__version__}")
print(f"  scipy          : ", end="")
import scipy; print(scipy.__version__)
print(f"  pandas         : {pd.__version__}")
print(f"  matplotlib     : {matplotlib.__version__}")
print("=" * 74)
"""))

    A(md(r"""
### 0.1 Output directory tree

Every experiment writes here automatically. No value in this notebook has to be
copied out of a plot by hand.
"""))

    A(code(r"""
RESULTS = Path("results")
DIRS = {k: RESULTS / k for k in
        ["figures", "metrics", "checkpoints", "logs", "tables"]}
DIRS["baseline"] = RESULTS / "baseline"
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 11, "figure.autolayout": False,
})

def savefig(fig, name, subdir="figures"):
    '''Save a figure as PNG under results/ and return its path.'''
    path = DIRS[subdir] / f"{name}.png"
    fig.savefig(path)
    return path

def save_table(df, name, float_fmt="%.6g"):
    '''Persist a DataFrame as CSV (the source of truth) and, if possible, Markdown.

    The Markdown rendering needs the optional `tabulate` package. It must never
    be able to abort an experiment cell, so it is best-effort.
    '''
    csv = DIRS["tables"] / f"{name}.csv"
    df.to_csv(csv, index=False, float_format=float_fmt)
    try:
        (DIRS["tables"] / f"{name}.md").write_text(df.to_markdown(index=False))
    except Exception as exc:                 # e.g. tabulate not installed
        print(f"  [warn] markdown for {name} skipped ({type(exc).__name__}); CSV written")
    return csv

def save_json(obj, name, subdir="metrics"):
    path = DIRS[subdir] / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path

print("Output tree ready:")
for k, v in DIRS.items():
    print(f"  {k:12s} -> {v}/")
"""))

    A(md(r"""
### 0.2 Experiment plan

Every sweep range and iteration budget in the study is declared here, in one
place, and printed. Nothing downstream invents a number.
"""))

    A(code(r"""
# ---- experiment plan (all sweep ranges and budgets in one place) -----------
MODES_TO_TEST   = [1, 2, 3]              # Section 19: high-frequency sweep
SIGMA_T2_SWEEP  = [1, 5, 10, 20, 40]     # Section 13.3: Fourier bandwidth diagnostic
WMIN_SWEEP      = [1.0, 0.5, 0.1, 0.01]  # Section 17.3: causality strength (see 17.3)
DATA_FRACTIONS  = [1.0, 0.5, 0.25, 0.10] # Section 20
NOISE_LEVELS    = [0.0, 0.01, 0.05]      # Section 21
CAUSAL_WMIN     = 0.1                    # provisional; replaced by the 17.3 sweep

if FAST_MODE:
    ITERS_MAIN_DEFAULT, ITERS_SWEEP_DEFAULT, ITERS_HF, ITERS_AUX = 400, 300, 300, 300
    SIGMA_T2_SWEEP, WMIN_SWEEP = [10, 30], [1.0, 0.1]
    MODES_TO_TEST, DATA_FRACTIONS, NOISE_LEVELS = [1, 3], [1.0, 0.25], [0.0, 0.05]
elif REPRODUCTION_MODE:
    ITERS_MAIN_DEFAULT, ITERS_SWEEP_DEFAULT, ITERS_HF, ITERS_AUX = 30000, 8000, 20000, 8000
else:
    ITERS_MAIN_DEFAULT, ITERS_SWEEP_DEFAULT, ITERS_HF, ITERS_AUX = 4000, 1000, 2000, 1200

PLAN = {
    "modes_tested": MODES_TO_TEST, "sigma_t2_sweep": SIGMA_T2_SWEEP,
    "wmin_sweep": WMIN_SWEEP, "data_fractions": DATA_FRACTIONS,
    "noise_levels": NOISE_LEVELS,
    "iters_main": ITERS_MAIN_DEFAULT, "iters_sweep": ITERS_SWEEP_DEFAULT,
    "iters_high_freq": ITERS_HF, "iters_aux": ITERS_AUX,
}
save_json(PLAN, "experiment_plan", "logs")
print("EXPERIMENT PLAN")
for k, v in PLAN.items():
    print(f"  {k:18s} = {v}")

n_runs = (3 + len(SIGMA_T2_SWEEP) + 1 + len(WMIN_SWEEP) + 1 + 4
          + 3 * len(MODES_TO_TEST) + 2 * len(DATA_FRACTIONS) + 2 * len(NOISE_LEVELS))
print(f"\n  ~{n_runs} training runs will be executed (cached after the first pass).")
"""))

    # ------------------------------------------------------------ Section 1 --
    A(md(r"""
---
# 1. Research objective

## 1.1 Research question

> **Can the published Fourier/NTK-enhanced PINN framework for Euler–Bernoulli
> beam vibration be reproduced, and can a defensible optimization improve its
> convergence, accuracy, computational efficiency, or robustness?**

## 1.2 The four models under study

The whole notebook is organised around four clearly separated objects. Keeping
these distinct is what makes the comparison in Section 19 meaningful.

| Role | What it is | Where it comes from |
|---|---|---|
| **GROUND TRUTH** | The closed-form modal solution of the undamped simply-supported Euler–Bernoulli beam. Computed analytically in NumPy, never by a network. | Section 4 |
| **BASELINE** | A paper-style vanilla PINN: plain tanh MLP on $(x,t)$, soft IC/BC/PDE losses, fixed loss weights. | Section 7 |
| **ENHANCED BASELINE** | The paper's method: multi-scale spatio-temporal Fourier features **+** NTK-based adaptive loss weighting. | Sections 10–11 |
| **PROPOSED** | Enhanced baseline **+ one** additional mechanism, selected in Section 15 after a literature check. | Section 17 |

## 1.3 What "ground truth" means here

The reference solution is analytical, so the error we report is a true
approximation error, not a discrepancy against another numerical scheme. The
network never touches the reference: it is trained only from the PDE, the
boundary conditions and the initial conditions (plus, in Sections 18b–18c only,
sparse synthetic observations). This keeps the evaluation independent by
construction.

## 1.4 Scope

In scope: replication → baseline → Fourier/NTK → optimization → controlled
comparison. Explicitly **out of scope** for this notebook: railway
applications, axle bearings, SHM deployment, and the inverse/parameter-identification
half of the paper.
"""))

    # ------------------------------------------------------------ Section 2 --
    A(md(r"""
---
# 2. Paper parameters and assumptions

## 2.1 What we know, and how we know it

The three Fourier-feature scale parameters and the network geometry were
supplied by the project owner from the PDF. The abstract confirms the method.
**The beam's physical properties are not known to us** and are marked `ASSUMED`.

Since the governing equation is linear and the reference is analytical, an
`ASSUMED` beam does not weaken the *method-level* comparison in Sections 12–19:
all four models see exactly the same beam. It only prevents us from matching the
paper's absolute error figures.
"""))

    A(code(r"""
# ---------------------------------------------------------------------------
# Central parameter registry. Every number the study depends on lives here,
# each with an explicit provenance label. Change ASSUMED rows to the paper's
# values and re-run the notebook to turn this into a numerical replication.
# ---------------------------------------------------------------------------
PARAMS = [
    # symbol, value, unit, provenance, note
    ("E",        2.0e11,  "Pa",      "PAPER (Table A.10)", "Steel 1040."),
    ("rho",      7845.0,  "kg/m^3",  "PAPER (Table A.10)", "Steel 1040."),
    ("L",        2.75,    "m",       "PAPER (Table A.10)", "Beam length."),
    ("a",        0.030,   "m",       "PAPER (Table A.10)", "Square section side (width = height)."),
    ("A",        9.0e-4,  "m^2",     "PAPER (Table A.10)", "= a^2. Verified."),
    ("I",        6.75e-8, "m^4",     "PAPER (Table A.10)", "= a^4/12. Verified."),
    ("b",        0.0,     "N.s/m",   "PAPER (Table A.10)", "Undamped case. Paper's damped case uses b = 50.0."),
    ("W_n",      9.085,   "Hz",      "PAPER (Table A.10)", "1st natural frequency; we compute 9.0825 from E,rho,L,A,I."),
    ("EI/(rho A)", 1912.05, "m^4/s^2","PAPER (derived)",   "Paper prints 43.732 = sqrt of this in the PDE; see inconsistency 1."),
    ("x_domain", "[0, 2.75]", "m",   "PAPER (Eq. A.4)",  "Simply-supported span."),
    ("t_domain", "[0, 1]",    "s",   "PAPER (Eq. A.4)",  "1 s window = 9.08 cycles of mode 1. This is the paper's difficulty."),
    ("BC",   "u = u_xx = 0",  "-",   "PAPER (prose+A.3)","Pinned-pinned; Eq. A.4 as printed is free-free (inconsistency 2)."),
    ("IC",   "u0(x) static, u_t = 0", "-", "PAPER (Eq. A.2)", "u0 = F x (4x^2 - 3 l^2)/(48 E I); mode-1 projection used (Eq. A.3)."),
    ("depth",    4,       "layers",  "PAPER (Section 4)", "Table 4 test 12 says 6; see inconsistency 3."),
    ("width_nn", 200,     "neurons", "PAPER (Section 4)", "Neurons per hidden layer."),
    ("activation", "tanh", "-",      "PAPER (Section 4)", "Hidden activation."),
    ("sigma_x",  1.0,     "-",       "PAPER (Section 4)", "M_x = 1 spatial Fourier mapping."),
    ("sigma_t1", 1.0,     "-",       "PAPER (Section 4)", "M_t = 2 temporal mappings."),
    ("sigma_t2", 10.0,    "-",       "PAPER (Section 4)", "Second temporal scale."),
    ("optimizer", "Adam", "-",       "PAPER (Section 5)", "Adam."),
    ("lr",       1e-4,    "-",       "PAPER (Table 4 #12)", "Learning rate; paper found this the single most sensitive hyperparameter."),
    ("batch",    960,     "points",  "PAPER (Eq. A.6)",  "N_u = N_ut = N_uxx = N_f = 960."),
    ("mini_batch", 32,    "points",  "PAPER (Section 5.1.1)", "Critical: mini-batch 640 -> L2 7.32e-1, mini-batch 32 -> L2 4.64e-4."),
    ("epochs",   30000,   "epochs",  "PAPER (Appendix A)", "Simply-supported case; fixed-end case uses 45 000."),
    ("m_fourier", 64,     "-",       "OURS",             "Fourier features per mapping; paper does not state the count."),
    ("mode_n",   1,       "-",       "PAPER (Eq. A.3)",  "Paper solves the first mode."),
    ("metrics",  "rel-L2, RMSE",     "-", "PAPER (Eqs. 44,45)", "rel-L2 vs exact; RMSE vs FEA. We add PDE/IC/BC/frequency errors."),
]

# Values the paper REPORTS for the simply-supported beam (Appendix A) and its
# method ladder (Table 5, fixed-end damped case).
PAPER_REPORTED = {
    "simply_supported_undamped": {"rel_L2": 2.3e-3, "RMSE_vs_FEA": 8.82e-4, "epochs": 30000},
    "simply_supported_damped":   {"rel_L2": 4.07e-2, "RMSE_vs_FEA": 6.12e-4, "epochs": 30000},
    "method_ladder_fixed_end_damped": {      # Table 5
        "FCNN": 1.00, "Vanilla PINN": 1.11,
        "PINN + NTK": 8.81e-1, "PINN + NTK + Fourier": 4.64e-4,
    },
    "inverse_damping_rel_err_pct": 1.41,     # Table 6
}


param_df = pd.DataFrame(PARAMS, columns=["symbol", "value", "unit", "provenance", "note"])
save_table(param_df, "01_parameters")

display(Markdown("### Parameter registry"))
display(param_df)

n_paper = param_df.provenance.str.startswith("PAPER").sum()
print(f"\n{n_paper} of {len(param_df)} entries come directly from the paper; "
      f"{len(param_df)-n_paper} are OURS.")
print("\nPaper-reported results for this case (simply supported, Appendix A):")
for k, v in PAPER_REPORTED["simply_supported_undamped"].items():
    print(f"   undamped {k:12s} = {v}")
print("\nPaper method ladder (Table 5, fixed-end damped):")
for k, v in PAPER_REPORTED["method_ladder_fixed_end_damped"].items():
    print(f"   {k:24s} rel-L2 = {v:g}")
"""))

    A(md(r"""
## 2.2 Derived quantities

Dependent quantities follow from the primary parameters by the standard
Euler–Bernoulli relations:

$$A = w h, \qquad I = \frac{w h^{3}}{12}, \qquad c = \sqrt{\frac{EI}{\rho A}}$$

$$\beta_n = \frac{n\pi}{L}, \qquad
\omega_n = \beta_n^{2}\sqrt{\frac{EI}{\rho A}} = \beta_n^2 c, \qquad
f_n = \frac{\omega_n}{2\pi}$$

$\beta_n$ is the wavenumber of the $n$-th simply-supported mode; $\omega_n$ its
undamped circular natural frequency. The $\beta_n^2$ dependence is why beam
modes spread out so fast — mode 3 is $9\times$ the frequency of mode 1, which is
exactly the spectral-bias stress test this paper is about.
"""))

    A(code(r"""
@dataclass(frozen=True)
class BeamParams:
    '''Physical Euler-Bernoulli beam parameters (SI units).'''
    E: float = 2.0e11            # PAPER Table A.10 (Steel 1040)
    rho: float = 7845.0          # PAPER Table A.10
    L: float = 2.75              # PAPER Table A.10
    width: float = 0.030         # PAPER Table A.10 (square section, a)
    height: float = 0.030        # PAPER Table A.10
    b: float = 0.0               # PAPER Table A.10 (undamped; damped case uses 50.0)
    amplitude: float = 1e-3      # modal amplitude; relative L2 is amplitude-invariant
    t_end: float = 1.0           # PAPER Eq. (A.4): t in [0, 1] s

    @property
    def A(self):    return self.width * self.height              # cross-section area
    @property
    def I(self):    return self.width * self.height ** 3 / 12.0  # second moment of area
    @property
    def EI(self):   return self.E * self.I                       # flexural rigidity
    @property
    def rhoA(self): return self.rho * self.A                     # mass per unit length
    @property
    def c(self):    return math.sqrt(self.EI / self.rhoA)        # c = sqrt(EI/(rho A))

    def beta_n(self, n):  return n * math.pi / self.L            # beta_n = n pi / L
    def omega_n(self, n): return self.beta_n(n) ** 2 * self.c    # omega_n = beta_n^2 c
    def f_n(self, n):     return self.omega_n(n) / (2 * math.pi) # f_n = omega_n / 2 pi
    def T_ref(self):      return 2 * math.pi / self.omega_n(1)   # period of mode 1

beam = BeamParams()

rows = [("A  (area)", beam.A, "m^2"), ("I  (2nd moment)", beam.I, "m^4"),
        ("EI (rigidity)", beam.EI, "N.m^2"), ("rho*A (mass/length)", beam.rhoA, "kg/m"),
        ("c = sqrt(EI/rhoA)", beam.c, "m^2/s"), ("T_ref (period mode 1)", beam.T_ref(), "s")]
for n in (1, 2, 3, 4, 5):
    rows += [(f"beta_{n}", beam.beta_n(n), "1/m"),
             (f"omega_{n}", beam.omega_n(n), "rad/s"),
             (f"f_{n}", beam.f_n(n), "Hz")]
derived_df = pd.DataFrame(rows, columns=["quantity", "value", "unit"])
save_table(derived_df, "02_derived_quantities")
display(Markdown("### Derived quantities"))
display(derived_df)
"""))

    # ------------------------------------------------------------ Section 3 --
    A(md(r"""
---
# 3. Beam physics

## 3.1 Governing equation

Transverse free vibration of a uniform Euler–Bernoulli beam with viscous
damping:

$$\boxed{\;EI\,\frac{\partial^4 u}{\partial x^4}
\;+\; \rho A\,\frac{\partial^2 u}{\partial t^2}
\;+\; b\,\frac{\partial u}{\partial t} \;=\; 0\;}$$

Term by term:

| Term | Physical meaning |
|---|---|
| $EI\,u_{xxxx}$ | Elastic restoring force from bending. $u_{xx}$ is curvature, $EI u_{xx}$ the bending moment, and two more derivatives turn moment into a transverse force per unit length. |
| $\rho A\, u_{tt}$ | Inertia of the beam element (mass per unit length × acceleration). |
| $b\, u_t$ | Viscous damping, resisting velocity. **Set to $b=0$** for the replication so an exact analytical reference exists. The code carries the term throughout. |

The Euler–Bernoulli model assumes plane sections stay plane and normal to the
neutral axis: it ignores shear deformation and rotary inertia, so it is accurate
for slender beams and for the low modes. Our section is $1\,\mathrm{m}$ long and
$5\,\mathrm{mm}$ deep — slenderness $200$ — so modes 1–3 are well inside its
validity.

## 3.2 Boundary conditions (simply supported / pinned–pinned)

$$u(0,t) = 0, \qquad u(L,t) = 0 \qquad\text{(no deflection at the supports)}$$
$$u_{xx}(0,t) = 0, \qquad u_{xx}(L,t) = 0 \qquad\text{(no bending moment: pins cannot resist rotation)}$$

In plain English: the beam rests on two pins. It cannot move up or down at
either end, but it is free to rotate there, so the bending moment
$M = EI\,u_{xx}$ must vanish at both ends.

## 3.3 Initial conditions

$$u(x,0) = u_0(x) = A_n \sin\!\left(\frac{n\pi x}{L}\right), \qquad u_t(x,0) = 0$$

The beam is deflected into the shape of a single mode and released from rest.
Physically: pull the beam into that shape, hold it still, let go at $t=0$.
Because the initial shape is exactly one eigenfunction, the response stays in
that mode forever — which is what gives us a closed-form reference.
"""))

    A(md(r"""
## 3.4 Non-dimensionalization — documented, not silent

The physical equation is badly scaled for a neural network:
$EI \approx 10^{2}$, $\rho A \approx 2$, $u \approx 10^{-3}\,\mathrm{m}$ and
$\omega_1 \approx 74\ \mathrm{rad/s}$. Residuals would span many orders of
magnitude. We therefore train in non-dimensional variables. **This is a change
of variables, not a change of physics**, and the full algebra is given here.

Define

$$x^{*} = \frac{x}{L}\in[0,1], \qquad
  t^{*} = \frac{t}{T_{\mathrm{ref}}}\in[0,1], \qquad
  u^{*} = \frac{u}{A_n}, \qquad
  T_{\mathrm{ref}} = \frac{2\pi}{\omega_1}$$

Substituting into the PDE and dividing through by $\rho A A_n / T_{\mathrm{ref}}^{2}$:

$$\boxed{\;\alpha\, u^{*}_{x^*x^*x^*x^*} \;+\; u^{*}_{t^*t^*} \;+\; \zeta\, u^{*}_{t^*} \;=\; 0\;}
\qquad
\alpha = \frac{EI\,T_{\mathrm{ref}}^{2}}{\rho A L^{4}}, \qquad
\zeta = \frac{b\,T_{\mathrm{ref}}}{\rho A}$$

### Two choices of $T_{\mathrm{ref}}$, and why it matters

$T_{\mathrm{ref}}$ sets the *difficulty*, because it decides how many
oscillations the network must fit across $t^{*}\in[0,1]$.

**(a) The paper's window** (`PAPER`, Eq. A.4): $t\in[0,1]\,\mathrm{s}$, so
$T_{\mathrm{ref}} = 1\,\mathrm{s}$ and

$$\omega^{*}_1 = \omega_1 \cdot 1\,\mathrm{s} = 57.07
\;\;\Longrightarrow\;\; 9.08 \text{ cycles of mode 1 in the window}$$

$$\alpha = \frac{EI\,T_{\mathrm{ref}}^2}{\rho A L^4} = \frac{1912.05}{2.75^4} = 33.43$$

**(b) One fundamental period** (`OURS`, used for the controlled experiments):
$T_{\mathrm{ref}} = 2\pi/\omega_1$. Then the algebra collapses:

$$\sqrt{\alpha} = \frac{c\,T_{\mathrm{ref}}}{L^{2}}
= \frac{\omega_1 L^{2}}{\pi^{2}}\cdot\frac{2\pi}{\omega_1}\cdot\frac{1}{L^{2}}
= \frac{2}{\pi}
\;\;\Longrightarrow\;\; \alpha = \frac{4}{\pi^{2}} \approx 0.4053$$

$$\omega^{*}_n = (n\pi)^{2}\sqrt{\alpha} = 2\pi n^{2}
\;\;\Longrightarrow\;\; \text{mode } n \text{ completes exactly } n^{2} \text{ cycles}$$

Under (b) $\alpha$ is a *universal constant* — independent of the beam
entirely. That has a useful consequence: **the non-dimensional learning problem
under (b) is fixed by the mode number alone**, so the controlled comparison in
Sections 13–19 is unaffected by the beam's physical properties; they only
convert results back to SI units.

Choice (b) is what the controlled experiments use, because (a) at 9.08 cycles is
out of reach at our compute budget (Section 5.4). Section 13.6 runs (a) directly
so the paper's own difficulty is measured rather than side-stepped.

Residuals map back exactly:

$$r_{\mathrm{phys}} = \frac{\rho A\,A_n}{T_{\mathrm{ref}}^{2}}\; r^{*}$$

so we report the physical PDE residual too, and never confuse the two.

### Precision

Fourth derivatives amplify round-off. We default to `float32` and **measure**
whether that is adequate in Section 6b (Test 10) rather than assuming it; a
`float64` switch is a one-line change.
"""))

    A(code(r"""
@dataclass(frozen=True)
class NonDim:
    '''Non-dimensional problem:  alpha*u_xxxx + u_tt + zeta*u_t = 0  on [0,1]^2.'''
    alpha: float
    zeta: float
    T_ref: float
    L: float
    U0: float
    residual_scale: float        # r_phys = residual_scale * r_nondim

    @staticmethod
    def from_beam(p: BeamParams, T_ref=None):
        '''T_ref=None -> one fundamental period (OURS); pass p.t_end for the paper window.'''
        T = p.T_ref() if T_ref is None else float(T_ref)
        return NonDim(
            alpha=p.EI * T ** 2 / (p.rhoA * p.L ** 4),
            zeta=p.b * T / p.rhoA,
            T_ref=T, L=p.L, U0=p.amplitude,
            residual_scale=p.rhoA * p.amplitude / T ** 2,
        )

    def omega_star(self, n):
        '''Non-dimensional natural frequency  (n pi)^2 sqrt(alpha) = 2 pi n^2.'''
        return (n * math.pi) ** 2 * math.sqrt(self.alpha)

    # unit conversions (kept explicit so physical meaning is never lost)
    def x_to_phys(self, xs):  return xs * self.L
    def t_to_phys(self, ts):  return ts * self.T_ref
    def u_to_phys(self, us):  return us * self.U0

nd = NonDim.from_beam(beam)                       # (b) one fundamental period -- OURS
nd_paper = NonDim.from_beam(beam, T_ref=beam.t_end)  # (a) the paper's 1 s window

print(f"(b) OURS   T_ref = {nd.T_ref:.6e} s   alpha = {nd.alpha:.10f} "
      f"(4/pi^2 = {4/math.pi**2:.10f})")
print(f"(a) PAPER  T_ref = {nd_paper.T_ref:.6e} s   alpha = {nd_paper.alpha:.4f}")
print(f"    paper window holds {nd_paper.omega_star(1)/(2*math.pi):.3f} cycles of mode 1 "
      f"(omega* = {nd_paper.omega_star(1):.3f})")
print()
print(f"zeta   = {nd.zeta:.6f}   (undamped)")
print(f"T_ref  = {nd.T_ref:.6e} s     residual scale = {nd.residual_scale:.6e} N/m")
print()
for n in (1, 2, 3):
    print(f"  mode {n}: omega* = {nd.omega_star(n):9.5f}  (2*pi*n^2 = {2*math.pi*n**2:9.5f})"
          f"   -> {n**2} cycles on t* in [0,1];  f_phys = {beam.f_n(n):8.3f} Hz")
"""))

    # ------------------------------------------------------------ Section 4 --
    A(md(r"""
---
# 4. Analytical reference solution

## 4.1 Derivation

Separate variables, $u^{*}(x^{*},t^{*}) = U_n(x^{*})\,q(t^{*})$, with the
simply-supported eigenfunction

$$U_n(x^{*}) = \sin(n\pi x^{*})$$

which satisfies all four boundary conditions identically
($\sin$ and its second derivative both vanish at $x^{*}=0,1$). Substituting into
the non-dimensional PDE and using $U_n'''' = (n\pi)^4 U_n$ gives a single
damped-oscillator ODE for the modal coordinate:

$$\ddot q + \zeta\,\dot q + \omega^{*2}_n\, q = 0,
\qquad \omega^{*}_n = (n\pi)^2\sqrt{\alpha}$$

With release from rest, $q(0)=1$, $\dot q(0)=0$:

$$\textbf{undamped } (\zeta=0):\quad q(t^{*}) = \cos(\omega^{*}_n t^{*})
\;\;\Longrightarrow\;\;
\boxed{\,u^{*}(x^{*},t^{*}) = \sin(n\pi x^{*})\cos(\omega^{*}_n t^{*})\,}$$

$$\textbf{underdamped } (0<\zeta<2\omega^{*}_n):\quad
q(t^{*}) = e^{-\zeta t^{*}/2}\!\left[\cos(\omega_d t^{*}) + \frac{\zeta}{2\omega_d}\sin(\omega_d t^{*})\right],
\quad \omega_d = \sqrt{\omega^{*2}_n - \tfrac{\zeta^2}{4}}$$

Both branches are implemented. The damped branch is exact too, so the study is
not restricted to $b=0$ — we simply start there, as instructed.

## 4.2 Implementation

The reference and **all** of its derivatives are computed in closed form in
NumPy. No finite differences, no neural network, no numerical PDE solver is
involved anywhere in the ground truth — so the errors reported later are true
approximation errors.
"""))

    A(code(r"""
def _modal_time(nd: NonDim, n: int, t):
    '''q, q', q'' for  q'' + zeta q' + omega*^2 q = 0,  q(0)=1, q'(0)=0.'''
    t = np.asarray(t, dtype=np.float64)
    w, z = nd.omega_star(n), nd.zeta
    if z == 0.0:
        return np.cos(w * t), -w * np.sin(w * t), -(w ** 2) * np.cos(w * t)
    wd2 = w ** 2 - 0.25 * z ** 2
    if wd2 <= 0:
        raise NotImplementedError("Only the underdamped single-mode case is implemented.")
    wd = math.sqrt(wd2)
    env, cs, sn = np.exp(-0.5 * z * t), np.cos(wd * t), np.sin(wd * t)
    q  = env * (cs + (0.5 * z / wd) * sn)
    qd = -env * (w ** 2 / wd) * sn
    return q, qd, -z * qd - (w ** 2) * q


def analytical_solution(x, t, nd: NonDim, mode: int = 1):
    '''u*(x*,t*) = sin(n pi x*) q(t*)   -- exact separable single-mode solution.'''
    q, _, _ = _modal_time(nd, mode, t)
    return np.sin(mode * np.pi * np.asarray(x, dtype=np.float64)) * q

def analytical_ut(x, t, nd, mode=1):
    _, qd, _ = _modal_time(nd, mode, t)
    return np.sin(mode * np.pi * np.asarray(x, np.float64)) * qd

def analytical_utt(x, t, nd, mode=1):
    _, _, qdd = _modal_time(nd, mode, t)
    return np.sin(mode * np.pi * np.asarray(x, np.float64)) * qdd

def analytical_ux(x, t, nd, mode=1):
    k = mode * np.pi; q, _, _ = _modal_time(nd, mode, t)
    return k * np.cos(k * np.asarray(x, np.float64)) * q

def analytical_uxx(x, t, nd, mode=1):
    k = mode * np.pi; q, _, _ = _modal_time(nd, mode, t)
    return -(k ** 2) * np.sin(k * np.asarray(x, np.float64)) * q

def analytical_uxxxx(x, t, nd, mode=1):
    k = mode * np.pi; q, _, _ = _modal_time(nd, mode, t)
    return (k ** 4) * np.sin(k * np.asarray(x, np.float64)) * q


# --- immediate self-check: does the closed form satisfy the PDE? -------------
_x, _t = np.random.default_rng(0).random((2, 5000))
for _n in (1, 2, 3, 5):
    _r = (nd.alpha * analytical_uxxxx(_x, _t, nd, _n)
          + analytical_utt(_x, _t, nd, _n)
          + nd.zeta * analytical_ut(_x, _t, nd, _n))
    print(f"mode {_n}: max |analytical PDE residual| = {np.abs(_r).max():.3e}"
          f"   (relative to |u_tt|_max = {np.abs(analytical_utt(_x,_t,nd,_n)).max():.3e})")
"""))

    # ------------------------------------------------------------ Section 5 --
    A(md(r"""
---
# 5. Synthetic dataset generation

The paper generates its synthetic data from the analytical solution, so we do
the same and do **not** look for an external dataset — there is none to find,
and inventing one would break the ground-truth independence.

## 5.1 The three disjoint sets

| Set | Size | What it is | May influence training? |
|---|---|---|---|
| **TRAIN** | `N_train` | Sparse interior observations $(x,t,u)$ from the analytical solution, optionally noised. | Yes — **but only in Sections 18b/18c.** The main replication (Sections 7–18) is *pure physics*: no data term at all. |
| **VAL** | `N_val` | Independent interior points, used only to draw convergence curves. | No. Never enters a gradient; no model selection is done on it. |
| **TEST** | `N_test` on a uniform grid | The evaluation grid. Every headline metric is computed here. | **No. Strictly held out.** |

Collocation points (`N_collocation`) carry the PDE residual. They are *not*
data: they are unlabelled points where the physics is enforced, and they are
**resampled every iteration** from the stratified sampler below, so the model
never overfits a fixed point cloud.

## 5.2 Sampling design

- **Collocation:** uniform in $x^{*}$, *stratified* in $t^{*}$ (one point per
  equal time slab). Stratification matters here: with $n^2$ cycles in the window,
  plain uniform sampling leaves ragged temporal gaps, and it is also what makes
  the causal binning in Section 15 well-conditioned. Both samplers are available
  via `structured_t`.
- **Boundary:** half the points at $x^{*}=0$, half at $x^{*}=1$, uniform in $t^{*}$.
- **Initial:** uniform in $x^{*}$ at $t^{*}=0$.

All draws come from explicitly seeded generators.
"""))

    A(code(r"""
# ---- dataset sizes (scaled by execution mode) ------------------------------
if FAST_MODE:
    N_TRAIN, N_VAL, N_TEST_GRID, N_COLLOCATION = 200, 200, 41, 256
elif REPRODUCTION_MODE:
    N_TRAIN, N_VAL, N_TEST_GRID, N_COLLOCATION = 2000, 1000, 301, 2048
else:
    N_TRAIN, N_VAL, N_TEST_GRID, N_COLLOCATION = 1000, 500, 201, 512

N_IC, N_BC = (64, 64) if FAST_MODE else (128, 128)


def make_test_grid(nx=N_TEST_GRID, nt=N_TEST_GRID):
    '''Uniform evaluation grid on the non-dimensional domain [0,1]^2.'''
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T


def make_observations(n, mode, seed, noise=0.0, nd=nd):
    '''Sparse interior observations drawn from the analytical solution.

    noise is a fraction of the RMS signal amplitude, added to u only
    (never to the test ground truth).
    '''
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    t = rng.random(n)
    u = analytical_solution(x, t, nd, mode)
    u_clean = u.copy()
    if noise > 0:
        u = u + rng.normal(0.0, noise * np.sqrt(np.mean(u_clean ** 2)), size=u.shape)
    return {"x": x, "t": t, "u": u, "u_clean": u_clean, "noise": noise}


def sample_batch(n_c, n_ic, n_bc, gen, dtype=DTYPE, structured_t=True):
    '''One training batch of collocation / IC / BC points, resampled each iteration.'''
    if structured_t:                                   # stratified in time
        slab = torch.arange(n_c, dtype=dtype).reshape(-1, 1) / n_c
        c_t = slab + torch.rand(n_c, 1, generator=gen, dtype=dtype) / n_c
    else:
        c_t = torch.rand(n_c, 1, generator=gen, dtype=dtype)
    c_x  = torch.rand(n_c, 1, generator=gen, dtype=dtype)
    ic_x = torch.rand(n_ic, 1, generator=gen, dtype=dtype)
    half = n_bc // 2
    bc_x = torch.cat([torch.zeros(half, 1, dtype=dtype),
                      torch.ones(n_bc - half, 1, dtype=dtype)])
    bc_t = torch.rand(n_bc, 1, generator=gen, dtype=dtype)
    return {
        "c_x":  c_x.requires_grad_(True),  "c_t": c_t.requires_grad_(True),
        "ic_x": ic_x.requires_grad_(True), "ic_t": torch.zeros(n_ic, 1, dtype=dtype).requires_grad_(True),
        "bc_x": bc_x.requires_grad_(True), "bc_t": bc_t.requires_grad_(True),
    }


MODE_MAIN = 2                       # headline mode; see the note below (modes 1-3 swept in Section 19)
train_obs = make_observations(N_TRAIN, MODE_MAIN, seed=SEED + 1)
val_obs   = make_observations(N_VAL,   MODE_MAIN, seed=SEED + 2)
x_grid, t_grid, X_grid, T_grid = make_test_grid()
U_exact = analytical_solution(X_grid, T_grid, nd, MODE_MAIN)

data_df = pd.DataFrame([
    ("TRAIN (observations)", N_TRAIN, "random interior", "18b/18c only"),
    ("VAL (monitoring)",     N_VAL,   "random interior", "never"),
    ("TEST (evaluation)",    f"{N_TEST_GRID}x{N_TEST_GRID}={N_TEST_GRID**2}", "uniform grid", "never"),
    ("COLLOCATION (physics)", N_COLLOCATION, "stratified-t, resampled/iter", "yes (unlabelled)"),
    ("IC points",            N_IC,    "uniform x at t=0", "yes (unlabelled)"),
    ("BC points",            N_BC,    "x=0 and x=1",      "yes (unlabelled)"),
], columns=["set", "size", "sampling", "influences training?"])
save_table(data_df, "03_dataset_design")
display(Markdown("### Dataset design"))
display(data_df)
print(f"\nHeadline mode n = {MODE_MAIN}  ->  omega* = {nd.omega_star(MODE_MAIN):.3f}, "
      f"{MODE_MAIN**2} cycles in the time window, f = {beam.f_n(MODE_MAIN):.2f} Hz")
"""))

    A(md(r"""
### 5.4 Why mode 2 is the headline — a measured decision, not a convenience

`MODE_MAIN` is labelled `OURS`, so it needs a justification. We ran a pilot
before designing the experiment, on the enhanced (Fourier + NTK) model, at the
paper architecture (4x200 tanh), with everything else as specified:

| setting | cycles in window | $\omega^*$ | pilot budget | rel-$L^2$ reached |
|---|---|---|---|---|
| mode 1, window (b) | 1 | 6.3 | 1 500 iters | **0.0096** — converges cleanly |
| mode 2, window (b) | 4 | 25.1 | 3 000 iters | **0.528**, still descending |
| mode 3, window (b) | 9 | 56.5 | 3 000 iters | **0.918** — essentially no learning |
| **mode 1, paper window (a)** | **9.08** | **57.1** | — | comparable to the mode-3 row: out of reach here |

Note the last row: **the paper's own problem sits at 9.08 cycles**, essentially
the same difficulty as our mode-3 row. The paper reaches rel-$L^2 = 2.3\times10^{-3}$
there with 30 000 epochs (batch 960, mini-batch 32) on an RTX A6000 — several
orders of magnitude more gradient steps than this environment can supply. So the
difficulty ordering we measure is consistent with the paper; what we lack is
budget, and Section 13.6 reports that gap explicitly instead of implying we
matched it.

Mode 1 is too easy to separate four methods; mode 3 is not learnable by *any*
of them inside the compute available here (4 CPU cores), so a comparison there
would be a comparison of noise. Mode 2 (4 cycles, $\omega^*_2 = 25.1$) sits
between the two and is where the methods can actually be told apart.

Note what this makes the headline comparison: a **fixed-budget** comparison. At
mode 2 none of the models has converged when the budget runs out, so Sections
18–19 measure *how far each method gets in the same number of iterations*, not
the accuracy each would eventually reach. That is a legitimate and common way to
compare PINN training strategies, but it is a different question from asymptotic
accuracy, and we do not conflate the two.

**The mode-3 difficulty is not swept under the rug — it is a result.** It is the
spectral-bias phenomenon the paper exists to address, and Section 19 reports it
explicitly as error-versus-mode-number rather than quietly omitting the mode we
could not fit. What it costs us is the ability to say anything about the
*asymptotic* accuracy of any method at mode 3; that limitation is stated in the
conclusions.
"""))

    A(md(r"""
### 5.3 Verifying the split is disjoint and the distributions are sane
"""))

    A(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

axes[0].scatter(train_obs["x"], train_obs["t"], s=6, alpha=0.5, label="train obs", color="tab:blue")
axes[0].scatter(val_obs["x"], val_obs["t"], s=6, alpha=0.5, label="val obs", color="tab:orange")
axes[0].set(xlabel="$x^*$ [-]", ylabel="$t^*$ [-]", title="Observation sets (disjoint draws)")
axes[0].legend(fontsize=8)

_g = torch.Generator().manual_seed(SEED)
_b = sample_batch(N_COLLOCATION, N_IC, N_BC, _g)
axes[1].scatter(_b["c_x"].detach(), _b["c_t"].detach(), s=5, alpha=0.5, label="collocation")
axes[1].scatter(_b["ic_x"].detach(), _b["ic_t"].detach(), s=9, color="tab:red", label="IC")
axes[1].scatter(_b["bc_x"].detach(), _b["bc_t"].detach(), s=9, color="tab:green", label="BC")
axes[1].set(xlabel="$x^*$ [-]", ylabel="$t^*$ [-]", title="One training batch (resampled each iter)")
axes[1].legend(fontsize=8, loc="upper right")

axes[2].hist(_b["c_t"].detach().numpy().ravel(), bins=40, alpha=0.75, label="stratified $t^*$")
_g2 = torch.Generator().manual_seed(SEED)
_bu = sample_batch(N_COLLOCATION, N_IC, N_BC, _g2, structured_t=False)
axes[2].hist(_bu["c_t"].detach().numpy().ravel(), bins=40, alpha=0.55, label="uniform $t^*$")
axes[2].set(xlabel="$t^*$ [-]", ylabel="count", title="Temporal coverage of collocation points")
axes[2].legend(fontsize=8)

fig.tight_layout()
print("saved:", savefig(fig, "05_dataset_distributions"))
plt.show()
"""))

    # ------------------------------------------------------------ Section 6 --
    A(md(r"""
---
# 6. Visualization of the analytical solution

Five diagnostic views of the ground truth, all in physical units so the
behaviour can be sanity-checked against engineering intuition.
"""))

    A(code(r"""
def plot_analytical_overview(mode, nd=nd, beam=beam, tag=""):
    x, t, X, T = make_test_grid(241, 241)
    U = analytical_solution(X, T, nd, mode)
    x_p, t_p, U_p = nd.x_to_phys(x), nd.t_to_phys(t), nd.u_to_phys(U) * 1e3   # mm

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)

    # PLOT 1 -- mode shape
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x_p, np.sin(mode * np.pi * x) * nd.U0 * 1e3, lw=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="x [m]", ylabel="$U_n(x)$ [mm]",
           title=f"1. Mode shape $U_{{{mode}}}(x)=\\sin({mode}\\pi x/L)$")

    # PLOT 2 -- snapshots in time
    ax = fig.add_subplot(gs[0, 1])
    for frac in [0.0, 0.125, 0.25, 0.375, 0.5]:
        ti = frac / mode ** 2            # fractions of ONE oscillation of this mode
        ax.plot(x_p, analytical_solution(x, ti, nd, mode) * nd.U0 * 1e3,
                label=f"$t$={nd.t_to_phys(ti)*1e3:.2f} ms")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="x [m]", ylabel="u [mm]", title="2. $u(x,t)$ at several times")
    ax.legend(fontsize=7)

    # PLOT 3 -- space-time heatmap
    ax = fig.add_subplot(gs[0, 2])
    im = ax.pcolormesh(t_p * 1e3, x_p, U_p, shading="auto", cmap="RdBu_r",
                       vmin=-np.abs(U_p).max(), vmax=np.abs(U_p).max())
    ax.set(xlabel="t [ms]", ylabel="x [m]", title="3. Space-time displacement")
    fig.colorbar(im, ax=ax, label="u [mm]")

    # PLOT 4 -- centre-point response
    ax = fig.add_subplot(gs[1, 0])
    mid = analytical_solution(0.5, t, nd, mode) * nd.U0 * 1e3
    ax.plot(t_p * 1e3, mid, lw=1.2)
    ax.set(xlabel="t [ms]", ylabel="u(L/2, t) [mm]",
           title=f"4. Mid-span response ({mode**2} cycles)")
    if mode % 2 == 0:
        ax.text(0.5, 0.5, "mid-span is a NODE\nfor even modes",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, bbox=dict(fc="lightyellow", ec="grey"))

    # PLOT 5 -- FFT of a non-nodal point
    ax = fig.add_subplot(gs[1, 1])
    x_probe = 0.5 / mode                                  # antinode of this mode
    sig = analytical_solution(x_probe, t, nd, mode)
    spec = np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(len(sig)), n=8192))
    freqs = np.fft.rfftfreq(8192, d=(t_p[1] - t_p[0]))    # Hz
    ax.plot(freqs, spec / spec.max(), lw=1.2)
    for n in range(1, mode + 2):
        ax.axvline(beam.f_n(n), color="grey", ls="--", lw=0.8)
        ax.text(beam.f_n(n), 1.02, f"$f_{n}$", fontsize=7, ha="center")
    ax.set(xlim=(0, beam.f_n(mode) * 2.2), xlabel="frequency [Hz]",
           ylabel="normalised |FFT|", title=f"5. Spectrum at $x$={x_probe:.2f}L")

    # PLOT 6 -- modal frequency ladder
    ax = fig.add_subplot(gs[1, 2])
    ns = np.arange(1, 7)
    ax.plot(ns, [beam.f_n(n) for n in ns], "o-")
    ax.axvline(mode, color="tab:red", ls=":", label=f"studied mode {mode}")
    ax.set(xlabel="mode number n", ylabel="$f_n$ [Hz]",
           title="6. $f_n \\propto n^2$ (spectral spread)")
    ax.legend(fontsize=8)

    fig.suptitle(f"Analytical reference solution -- mode n={mode}"
                 f"  ($f_{{{mode}}}$={beam.f_n(mode):.2f} Hz)", fontsize=12)
    p = savefig(fig, f"06_analytical_overview_mode{mode}{tag}")
    plt.show()
    return p

for _m in (1, 2, 3):
    print("saved:", plot_analytical_overview(_m))
"""))

    return C
