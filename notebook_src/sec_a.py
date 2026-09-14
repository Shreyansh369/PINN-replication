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

## ⚠️ Provenance statement — read this first

**The paper PDF was not available in the environment in which this notebook was
built.** The full text sits behind a publisher paywall and the execution
environment had no route to it. Everything below therefore carries an explicit
provenance label, and nothing is presented as "from the paper" unless it
genuinely is.

| Label | Meaning |
|---|---|
| `PAPER (user)` | Value supplied by the project owner, who has the PDF. Treated as a paper value. |
| `PAPER (abstract)` | Confirmed from the publicly indexed abstract/metadata. |
| `ASSUMED` | **Chosen by us.** The paper's value is unknown. Documented, defensible, and *not* claimed to match the paper. |
| `OURS` | A deliberate methodological choice of this study, not part of the paper. |

**Consequences, stated plainly:**

1. We can replicate the paper's *method* (multi-scale Fourier features + NTK
   adaptive loss weighting on an Euler–Bernoulli beam). We **cannot** claim to
   replicate its *numbers*, because the beam properties, training budget and
   reported errors are unknown to us.
2. Section 12 therefore reports a **method-level reproduction**, and every
   "paper reported" cell in the final table reads `N/A (PDF unavailable)`.
   No paper metric is invented to fill a gap.
3. To convert this into a true numerical replication, replace the `ASSUMED`
   entries in the parameter table in Section 2 — they are all in one place — and
   re-run. Nothing else needs to change.
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
    '''Persist a DataFrame as CSV (machine-readable) and Markdown (report-ready).'''
    csv = DIRS["tables"] / f"{name}.csv"
    df.to_csv(csv, index=False, float_format=float_fmt)
    (DIRS["tables"] / f"{name}.md").write_text(df.to_markdown(index=False))
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
SIGMA_T2_SWEEP  = [1, 5, 10, 20, 30, 60] # Section 13.3: Fourier bandwidth diagnostic
WMIN_SWEEP      = [1.0, 0.5, 0.1, 0.01]  # Section 17.3: causality strength (see 17.3)
DATA_FRACTIONS  = [1.0, 0.5, 0.25, 0.10] # Section 20
NOISE_LEVELS    = [0.0, 0.01, 0.05]      # Section 21
CAUSAL_WMIN     = 0.1                    # provisional; replaced by the 17.3 sweep

if FAST_MODE:
    ITERS_MAIN_DEFAULT, ITERS_SWEEP_DEFAULT, ITERS_HF, ITERS_AUX = 400, 300, 300, 300
    SIGMA_T2_SWEEP, WMIN_SWEEP = [10, 30], [1.0, 0.1]
    MODES_TO_TEST, DATA_FRACTIONS, NOISE_LEVELS = [1, 3], [1.0, 0.25], [0.0, 0.05]
elif REPRODUCTION_MODE:
    ITERS_MAIN_DEFAULT, ITERS_SWEEP_DEFAULT, ITERS_HF, ITERS_AUX = 40000, 20000, 40000, 20000
else:
    ITERS_MAIN_DEFAULT, ITERS_SWEEP_DEFAULT, ITERS_HF, ITERS_AUX = 8000, 3000, 6000, 4000

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
    ("E",        2.1e11,  "Pa",      "ASSUMED",         "Structural steel. Paper value unknown."),
    ("rho",      7850.0,  "kg/m^3",  "ASSUMED",         "Structural steel. Paper value unknown."),
    ("L",        1.0,     "m",       "ASSUMED",         "Beam length. Paper value unknown."),
    ("width",    0.05,    "m",       "ASSUMED",         "Rectangular section width."),
    ("height",   0.005,   "m",       "ASSUMED",         "Rectangular section height."),
    ("b",        0.0,     "N.s/m^2", "OURS",            "Viscous damping. Set to 0: the undamped case has a clean analytical reference (task spec)."),
    ("A_n",      5e-3,    "m",       "ASSUMED",         "Initial modal amplitude (5 mm)."),
    ("sigma_x",  1.0,     "-",       "PAPER (user)",    "Spatial Fourier feature std."),
    ("sigma_t1", 1.0,     "-",       "PAPER (user)",    "Temporal Fourier scale 1."),
    ("sigma_t2", 10.0,    "-",       "PAPER (user)",    "Temporal Fourier scale 2."),
    ("depth",    4,       "layers",  "PAPER (user)",    "Hidden layers."),
    ("width_nn", 200,     "neurons", "PAPER (user)",    "Neurons per hidden layer."),
    ("activation", "tanh", "-",      "PAPER (user)",    "Hidden activation."),
    ("m_fourier", 64,     "-",       "ASSUMED",         "Fourier features per encoding. Paper value unknown."),
    ("optimizer", "Adam", "-",       "ASSUMED",         "Standard for PINNs. Paper's choice unknown."),
    ("lr",       1e-3,    "-",       "ASSUMED",         "Adam initial LR. Paper value unknown."),
    ("lr_decay", 0.1,     "-",       "OURS",            "Exponential decay factor over the full schedule."),
    ("epochs",   "see 0", "iters",   "ASSUMED",         "Paper's budget unknown; ours is set by the execution mode."),
    ("batch",    "see 0", "points",  "ASSUMED",         "Collocation points resampled per iteration."),
    ("mode_n",   3,       "-",       "OURS",            "Headline mode. Modes 1-3 are swept in Section 18."),
    ("x_domain", "[0, L]", "m",      "PAPER (abstract)","Simply-supported span."),
    ("t_domain", "[0, T1]", "s",     "OURS",            "One period of the fundamental mode; mode n then shows n^2 cycles."),
    ("metrics",  "rel-L2, RMSE, max-err, PDE res, IC/BC err, freq err", "-", "OURS", "Evaluation metrics (Section 9)."),
]

param_df = pd.DataFrame(PARAMS, columns=["symbol", "value", "unit", "provenance", "note"])
save_table(param_df, "01_parameters")

from IPython.display import display, Markdown
display(Markdown("### Parameter registry"))
display(param_df)

n_assumed = (param_df.provenance == "ASSUMED").sum()
print(f"\n{n_assumed} of {len(param_df)} entries are ASSUMED (paper value unknown to us).")
print("These are the rows to overwrite for a true numerical replication.")
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
    E: float = 2.1e11
    rho: float = 7850.0
    L: float = 1.0
    width: float = 0.05
    height: float = 0.005
    b: float = 0.0
    amplitude: float = 5e-3      # A_n, initial modal amplitude [m]

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

Two consequences worth checking, because they are strong correctness tests:

$$\sqrt{\alpha} = \frac{c\,T_{\mathrm{ref}}}{L^{2}}
= \frac{\omega_1 L^{2}}{\pi^{2}}\cdot\frac{2\pi}{\omega_1}\cdot\frac{1}{L^{2}}
= \frac{2}{\pi}
\;\;\Longrightarrow\;\; \alpha = \frac{4}{\pi^{2}} \approx 0.4053 \quad\text{(always, for any beam)}$$

$$\omega^{*}_n = \omega_n T_{\mathrm{ref}} = (n\pi)^{2}\sqrt{\alpha} = 2\pi n^{2}
\;\;\Longrightarrow\;\; \text{mode } n \text{ completes exactly } n^{2} \text{ cycles on } t^{*}\in[0,1]$$

So $\alpha$ is a *universal constant* of this non-dimensionalization — it does
not depend on the `ASSUMED` beam properties at all. **That is an important
robustness result for this study:** the non-dimensional learning problem that
all four models actually solve is fixed by the mode number alone, so our
method-level comparison is independent of the beam parameters we had to guess.
The beam properties only set the dictionary back to physical units.

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
    def from_beam(p: BeamParams):
        T = p.T_ref()
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

nd = NonDim.from_beam(beam)

print(f"alpha  = {nd.alpha:.10f}   (4/pi^2 = {4/math.pi**2:.10f})")
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


MODE_MAIN = 3                       # headline mode (modes 1-3 swept in Section 18)
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
