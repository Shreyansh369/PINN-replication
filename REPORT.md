# Replication and Controlled Optimization Study
## A Physics-Informed Deep Neural Network for Euler–Bernoulli Beam Vibration

**Target paper**

> Cem Söyleyici and Hakkı Özgür Ünver,
> *"A Physics-Informed Deep Neural Network based beam vibration framework for
> simulation and parameter identification"*,
> **Engineering Applications of Artificial Intelligence 141 (2025) 109804**.
> DOI: [10.1016/j.engappai.2024.109804](https://doi.org/10.1016/j.engappai.2024.109804)

All numbers in this report were produced by `beam_pinn_research.ipynb` on this
machine. Nothing is copied from the paper except where explicitly labelled
**PAPER**, and no paper value is invented.

**Status:** the experiment run was stopped early by request. 22 of ~30 planned
models completed. Sections 1–7 below are complete; Section 8 lists exactly what
is missing and what it would have answered.

---

## 1. Research question

> Can the published Fourier/NTK-enhanced PINN framework for Euler–Bernoulli beam
> vibration be reproduced, and can a defensible optimization improve its
> convergence, accuracy, efficiency or robustness?

Four objects are kept strictly separate throughout:

| Role | What it is |
|---|---|
| **Ground truth** | Closed-form modal solution, computed analytically in NumPy. Never touches the network. |
| **Baseline** | Vanilla tanh PINN, fixed loss weights. |
| **Enhanced baseline** | The paper's method: multi-scale Fourier features + NTK adaptive weighting. |
| **Proposed** | Enhanced baseline + temporal causal weighting (one added mechanism). |

---

## 2. Physics and the analytical reference

### 2.1 Governing equation

$$EI\,\frac{\partial^4 u}{\partial x^4} + \rho A\,\frac{\partial^2 u}{\partial t^2} + b\,\frac{\partial u}{\partial t} = 0$$

| Term | Meaning |
|---|---|
| $EI\,u_{xxxx}$ | Elastic bending restoring force. $EI u_{xx}$ is the bending moment; two more derivatives convert moment to transverse force per unit length. |
| $\rho A\,u_{tt}$ | Inertia (mass per unit length × acceleration). |
| $b\,u_t$ | Viscous damping. Set to $b=0$ here so an exact reference exists. |

### 2.2 Boundary and initial conditions (simply supported)

$$u(0,t)=u(L,t)=0 \qquad\text{(pinned: no deflection)}$$
$$u_{xx}(0,t)=u_{xx}(L,t)=0 \qquad\text{(no bending moment: pins cannot resist rotation)}$$
$$u(x,0)=u_0(x),\qquad u_t(x,0)=0 \qquad\text{(deflected, released from rest)}$$

### 2.3 Analytical solution

Separating $u = U_n(x)q(t)$ with $U_n(x)=\sin(n\pi x/L)$, which satisfies all
four BCs identically:

$$\beta_n=\frac{n\pi}{L},\qquad
\omega_n=\beta_n^2\sqrt{\frac{EI}{\rho A}},\qquad
f_n=\frac{\omega_n}{2\pi}$$

$$\ddot q+\zeta\dot q+\omega_n^{*2}q=0,\quad q(0)=1,\ \dot q(0)=0$$

$$\textbf{undamped:}\quad u(x,t)=\sin\!\Big(\frac{n\pi x}{L}\Big)\cos(\omega_n t)$$

$$\textbf{underdamped:}\quad q(t)=e^{-\zeta t/2}\Big[\cos\omega_d t+\tfrac{\zeta}{2\omega_d}\sin\omega_d t\Big],\quad
\omega_d=\sqrt{\omega_n^{*2}-\tfrac{\zeta^2}{4}}$$

**Verified to machine precision:** substituting the closed form into the PDE
gives max residual $\sim10^{-13}$ relative to $|u_{tt}|_{\max}$, for modes 1, 2,
3 and 5 (unit test 4).

### 2.4 Non-dimensionalization (documented, not silent)

$$x^*=\frac{x}{L},\qquad t^*=\frac{t}{T_{\mathrm{ref}}},\qquad u^*=\frac{u}{A_n}$$

$$\boxed{\ \alpha\,u^*_{x^*x^*x^*x^*}+u^*_{t^*t^*}+\zeta\,u^*_{t^*}=0\ }
\qquad
\alpha=\frac{EI\,T_{\mathrm{ref}}^2}{\rho A L^4},\quad \zeta=\frac{b\,T_{\mathrm{ref}}}{\rho A}$$

Residuals map back exactly: $r_{\mathrm{phys}}=\dfrac{\rho A\,A_n}{T_{\mathrm{ref}}^2}\,r^*$.

Two choices of $T_{\mathrm{ref}}$ are used:

- **(a) PAPER window** $T_{\mathrm{ref}}=1\,$s $\Rightarrow \alpha=33.43$, $\omega_1^*=57.07$, i.e. **9.08 cycles** in the window.
- **(b) OURS** $T_{\mathrm{ref}}=2\pi/\omega_1 \Rightarrow \alpha=4/\pi^2\approx0.4053$ **for any beam**, and $\omega_n^*=2\pi n^2$, so mode $n$ shows exactly $n^2$ cycles.

**Empirically confirmed:** swapping the entire beam (E, ρ, L, section, amplitude)
left every non-dimensional metric bit-identical and the validation trajectory
agreeing to $4.4\times10^{-16}$. Only the physical-unit conversion changed, by
exactly the factor $\rho A A_n/T_{\mathrm{ref}}^2$ (82.4205 → 35.5756).

---

## 3. Paper parameters (all PAPER, from the PDF)

This notebook replicates the **simply-supported** case, which is the paper's
**Appendix A** (the main text treats a fixed-end beam).

| Symbol | Value | Source |
|---|---|---|
| $E$ | $2.0\times10^{11}$ Pa | Table A.10 (Steel 1040) |
| $\rho$ | 7845.0 kg/m³ | Table A.10 |
| $L$ | 2.75 m | Table A.10 |
| $a$ (square section) | 0.030 m | Table A.10 |
| $A$ | $9.0\times10^{-4}$ m² | Table A.10 (= $a^2$, verified) |
| $I$ | $6.75\times10^{-8}$ m⁴ | Table A.10 (= $a^4/12$, verified) |
| $b$ | 0.0 (damped case: 50.0) | Table A.10 |
| $W_n$ | 9.085 Hz | Table A.10 |
| domain | $x\in[0,2.75]$ m, $t\in[0,1]$ s | Eq. (A.4) |
| network | 4 hidden layers × 200 neurons, tanh | Section 4 |
| Fourier | $M_x=1$ ($\sigma_x=1$), $M_t=2$ ($\sigma_{t}=1,10$) | Section 4 |
| optimizer | Adam, lr $10^{-4}$ | Table 4 (test 12) |
| batching | batch 960, **mini-batch 32** | Eq. (A.6), §5.1.1 |
| epochs | 30 000 | Appendix A |

**Derived and cross-checked against the paper's own frequencies:**

| Quantity | Computed | Paper |
|---|---|---|
| $EI$ | 13 500 N·m² | — |
| $\rho A$ | 7.0605 kg/m | — |
| $c=\sqrt{EI/\rho A}$ | 43.7269 m²/s | prints 43.732 |
| $f_1$ simply supported | **9.0825 Hz** | **9.085 Hz** ✓ |
| $f_1$ fixed-end | **20.5886 Hz** | **20.594 Hz** ✓ |

### 3.1 Three inconsistencies found in the paper

| # | Paper states | Problem | Resolution used |
|---|---|---|---|
| 1 | PDE as $43.732\,u_{xxxx}+u_{tt}=0$ (Eqs. 46, A.4) | 43.732 is $\sqrt{EI/\rho A}$, not $EI/\rho A$. Taken literally, $f_1=1.37$ Hz, contradicting Table A.10's 9.085 Hz. Lost superscript. | Use $EI/\rho A=1912.05$, which reproduces 9.0825 Hz. |
| 2 | Eq. (A.4) BCs $u_{xx}=u_{xxx}=0$ | Those are **free–free**, not simply supported. Contradicts the prose *and* Eq. (A.3)'s $\beta_1 l=3.1416=\pi$. | Use $u=u_{xx}=0$. |
| 3 | §4 "four hidden layers"; Table 4 test 12 "6" | Direct contradiction on depth. | Follow §4 (the framework description). |

---

## 4. Method implementation

### 4.1 Multi-scale Fourier features

$$\gamma_\sigma(v)=\big[\cos(B_\sigma v),\ \sin(B_\sigma v)\big],\qquad B_\sigma\sim\mathcal N(0,\sigma^2)\ \text{fixed}$$

```
x ──► γ_{σx}  ──► trunk ──► h_x ──┐
                                  ├─(⊙)─► h_x ⊙ h_t1 ─┐
t ──► γ_{σt1} ──► trunk ──► h_t1 ─┘                   ├─ concat ─► linear ─► u_θ
t ──► γ_{σt2} ──► trunk ──► h_t2 ──(⊙)─► h_x ⊙ h_t2 ──┘
```

The paper's Eqs. (38)–(43) confirm this structure exactly: shared trunk for the
x and t branches, **pointwise multiplicative** merge, concatenated linear
read-out. The multiplicative merge is what lets the network express
$\sin(\beta x)\cos(\omega t)$ directly.

### 4.2 NTK adaptive loss weighting

$$K_i=J_iJ_i^{\mathsf T},\qquad
\boxed{\ \lambda_i=\frac{\sum_j\operatorname{tr}(K_j)}{\operatorname{tr}(K_i)}\ }$$

matching the paper's Eq. (37). The trace avoids forming $K$:

$$\operatorname{tr}(K_i)=\lVert J_i\rVert_F^2=\sum_{k}\lVert\nabla_\theta f_i(z_k)\rVert^2
\ \approx\ \frac{N_i}{m}\sum_{k\in S}\lVert\nabla_\theta f_i(z_k)\rVert^2,\quad m=16$$

**Estimator validated against the exact trace** (`05_ntk_estimator_validation.csv`).
Row subsampling was chosen over a Hutchinson probe on measured evidence:

| term | row-subsample error | Hutchinson error |
|---|---|---|
| ic | ~1 % | ~21 % |
| vel | ~2 % | ~26 % |
| bc_u | ~2 % | ~17 % |
| bc_m | ~4 % | ~19 % |
| pde | ~9 % | ~11 % |

Hutchinson is unbiased but its variance scales with the NTK off-diagonal mass,
which is large for PINNs.

### 4.3 Loss terms

$$\mathcal L=\lambda_{ic}\mathcal L_{ic}+\lambda_{vel}\mathcal L_{vel}
+\lambda_{bc_u}\mathcal L_{bc_u}+\lambda_{bc_m}\mathcal L_{bc_m}+\lambda_{pde}\mathcal L_{pde}$$

The BC term is split into displacement and moment (`OURS`, documented) because
$|u_{xx}|\sim(n\pi)^2$ while $|u|\sim1$; a combined term would be dominated by
the moment condition by $\sim(n\pi)^4$ and the displacement BC would vanish.

The PDE residual is trained in the equivalent form $\hat r=r^*/\omega_n^{*2}$;
dividing a homogeneous equation by a positive constant does not change its
solution set. **Reported** residuals are always unscaled.

### 4.4 Proposed optimization — temporal causal weighting

$$\mathcal L_{pde}^{\text{causal}}=\frac{\sum_i n_i w_i\mathcal L_i}{\sum_i n_i},
\qquad \boxed{\ w_i=\exp\Big(-\varepsilon\sum_{j<i}\mathcal L_j\Big)\ }\ \text{(stop-gradient)}$$

Bins are weighted by **population** $n_i$, so $w\equiv1$ reproduces the plain
mean $\sum_k\hat r_k^2/N$ **exactly** (unit test 12, to $10^{-6}$ relative).

$\varepsilon$ carries units of 1/loss, so a literature value of $O(1)$ is
meaningless here — measured, it leaves every $w_i=1.00000$ and the mechanism
inert. It is therefore reparameterised by a **dimensionless** target $w_{\min}$:

$$\varepsilon=\frac{-\ln w_{\min}}{S_0},\qquad S_0=\sum_{j<M}\mathcal L_j\ \text{at iteration 1}$$

held fixed thereafter, so weights relax to 1 as the residual falls.
$w_{\min}=1\Rightarrow\varepsilon=0\Rightarrow$ exactly the baseline.

---

## 5. Results

All at mode 2 (4 cycles), 4000 iterations, 4×200 tanh, seed 1234, identical
sampler / optimizer / schedule / test grid. Reference = analytical solution.

### 5.1 Reproduction of the paper's method (`07_reproduction_comparison.csv`)

| Model | rel-L2 | RMSE | PDE res (nd) | IC err | BC err | freq err |
|---|---|---|---|---|---|---|
| M1 vanilla | 0.9903 | 0.4951 | 61.08 | 0.4487 | 0.1617 | 100 % |
| M2 + Fourier | 0.9904 | 0.4952 | 56.07 | 0.0197 | 0.0202 | 13.6 % |
| **M3 + Fourier + NTK** | **0.4110** | **0.2055** | **18.56** | **0.0039** | **0.0027** | **1.43 %** |

**Reading this by rel-L2 alone is misleading.** M1 and M2 are indistinguishable
on rel-L2, but Fourier features cut IC error 23×, BC error 8×, and frequency
error from 100 % (no recognisable oscillation) to 13.6 %. The encoding buys the
*capacity* to represent 4 cycles and satisfy the constraints; NTK balancing is
what converts that capacity into accuracy.

**The paper's own ladder corroborates this from the other direction** (its Table 5,
fixed-end damped):

| Method | PAPER rel-L2 |
|---|---|
| FCNN | 1.00 |
| Vanilla PINN | 1.11 |
| PINN + NTK | 0.881 |
| PINN + NTK + Fourier | 4.64e-4 |

The paper sees NTK *alone* barely beat vanilla; we see Fourier *alone* barely
move rel-L2. Both say the same thing: **neither ingredient suffices alone.**

### 5.2 The paper's own difficulty (`07c_paper_window_gap.csv`)

| Source | rel-L2 | Budget | Hardware |
|---|---|---|---|
| **PAPER** (Appendix A) | **2.30e-3** | 30 000 epochs, batch 960, mini-batch 32 | RTX A6000 |
|  **OURS** (same beam, same 1 s window) | **9.71e-1** | 4 000 full-batch iterations | 4 CPU threads |

A factor of **420**. Attribution, in order of likely size:

1. **Budget** — the paper runs ~10⁶ gradient steps; we run 4×10³.
2. **Mini-batching** — the paper reports this as one of its most influential
   knobs (mini-batch 640 → rel-L2 0.732 vs 32 → 4.64e-4). We use full batches,
   i.e. the regime the paper itself found *worse*.
3. Hardware (GPU vs 4 CPU threads).
4. Fourier feature count, which the paper does not state (we use 64).

**The method and physical setup are reproduced. The published error value is
not, and nothing here should be read as claiming otherwise.**

### 5.3 Ablation — the headline result (`10_ablation.csv`)

Full 2×2 factorial, only the two switches vary:

| cell | NTK | causal | rel-L2 | PDE res | IC err | BC err | freq err |
|---|---|---|---|---|---|---|---|
| A Fourier only | ✗ | ✗ | 0.9904 | 56.1 | 0.0197 | 0.0202 | 13.6 % |
| **B Fourier + NTK** | ✓ | ✗ | **0.4110** | **18.6** | 0.0039 | 0.0027 | **1.43 %** |
| C Fourier + causal | ✗ | ✓ | 1.0561 | 117.8 | 0.0137 | 0.0069 | 5.7 % |
| D **proposed** | ✓ | ✓ | 1.0511 | 122.7 | 0.0071 | 0.0034 | 69.2 % |

**Factorial effects** on $\log$ rel-L2 (negative = improvement):

$$\text{NTK alone}=-0.8796,\qquad
\text{causal alone}=+0.0642,\qquad
\boxed{\text{interaction}=+0.8749\ \textbf{(antagonistic)}}$$

The interaction **almost exactly cancels the NTK main effect** ($+0.8749$ vs
$-0.8796$). Causal weighting does not merely fail to help — it specifically
destroys what NTK weighting achieves.

**Mechanism, predicted before the experiment and then measured.** Section 17.1
flagged the risk: causal weighting deliberately shrinks $\mathcal L_{pde}$ early,
NTK weighting reads the resulting small kernel trace and *inflates* $\lambda_{pde}$
in response, and the two fight. The PDE residual is the direct evidence — adding
causality **raises** it from 18.6 to 122.7.

### 5.4 Causality strength sweep (`09_causal_strength_sweep.csv`)

| $w_{\min}$ | calibrated $\varepsilon$ | rel-L2 @1000 it |
|---|---|---|
| 1.00 (off) | 0.0 | **0.9168** |
| 0.50 | 17.80 | 1.0674 |
| 0.10 | 59.13 | 1.0060 |
| 0.01 | 118.27 | 1.0065 |

Causality-off leads at every active strength. Reproduced identically across two
independent runs, so this is not seed noise.

### 5.5 Fourier bandwidth diagnostic (`06_sigma_t2_sweep.csv`)

| $\sigma_{t2}$ | max\|B\| | spans $\omega^*_2$ | rel-L2 |
|---|---|---|---|
| 1 | 2.18 | ✗ | 0.9486 |
| 5 | 10.92 | ✗ | **0.8419** |
| 10 (paper) | 21.83 | ✗ | 0.9168 |
| 20 | 43.67 | ✓ | 0.9366 |
| 40 | 87.34 | ✓ | 0.9603 |

$\sigma$ is only meaningful relative to the time normalization. **We did not
deviate from the paper's value on this evidence**: the 8.2 % edge at
$\sigma_{t2}=5$ is below the 20 % decisive margin we set in advance, and at this
sweep budget nothing has converged, so the sweep cannot rank $\sigma$ reliably.

### 5.6 Frequency dependence (partial — `16_all_runs_summary.csv`)

| model | mode 1 (1 cycle) | mode 2 (4 cycles) |
|---|---|---|
| vanilla | 0.7023 | 0.9933 |
| Fourier + NTK | **0.0110** | **0.4108** |
| proposed (+causal) | **0.0108** | 1.0473 |

At mode 1 the proposed method is marginally *better* (2 %, within noise); at
mode 2 it is catastrophically worse. Consistent with the mechanism: at 1 cycle
training reaches the regime where causal weighting's early suppression is
repaid; at 4 cycles it never does within budget. The honest statement is
**"antagonistic at the difficulty where it matters"**, not "causality never helps".

### 5.7 Numerical-safety finding: denormals

Strong causal weighting ran 9× slower (122 s → 1160 s), reproducibly. The cause
is **not** contention but **denormal floating point**: at the calibrated
$\varepsilon=118.27$, five of 32 weight bins land in the float32 subnormal range
and four flush to zero, and the gradients they scale go subnormal even when the
weights do not. Measured on this machine:

| | ms per multiply-reduce |
|---|---|
| normal (1e-10) | 1.29 |
| denormal (1e-40) | **12.20** |

A 9.5× penalty, matching the observed slowdown. Fix: flush-to-zero or floor the
weights. **The causal timings in `results/` predate this fix and are inflated;
accuracy figures are unaffected**, since denormals are numerically negligible.

---

## 6. Reproducibility evidence

- **Determinism:** retraining every model from scratch reproduced all
  non-dimensional metrics to printed precision, with validation trajectories
  agreeing to $4.4\times10^{-16}$.
- **Beam independence:** swapping the entire beam left the non-dimensional
  problem and its solution bit-identical, confirming the $\alpha=4/\pi^2$
  derivation empirically.
- **Unit tests** (run before any training): analytical frequency vs formula,
  BCs, ICs, analytical PDE residual, autograd derivative shapes and values
  vs closed form, float32 adequacy for 4th derivatives, seed determinism,
  NTK estimator accuracy, causal-weight identity at $\varepsilon=0$.
- **float32 is adequate:** 4th-derivative relative error $\sim10^{-4}$,
  three orders below the accuracy reachable in training.

---

## 7. Honest conclusions

1. **Paper method: reproduced.** Multi-scale Fourier features + NTK trace
   weighting are implemented faithfully (architecture and weight rule verified
   against Eqs. 38–43 and 37) and run on the paper's beam and domain.
2. **Paper numbers: not reproduced, and not claimed.** 9.71e-1 vs 2.30e-3 on the
   paper's own window, attributable to a ~250× budget deficit and to full-batch
   training — the regime the paper reports as worse.
3. **Both ingredients are needed.** Our data and the paper's Table 5 agree from
   opposite directions that neither Fourier features nor NTK weighting suffices
   alone.
4. **The proposed optimization does not work, and we know why.** A controlled
   2×2 gives interaction $+0.8749$, cancelling the NTK main effect of $-0.8796$.
   The conflict was predicted from the mathematics before the run and confirmed
   by the PDE residual rising 18.6 → 122.7.
5. **Nothing here is claimed as novel.** Causal weighting (Wang, Sankaran &
   Perdikaris, CMAME 2024) and NTK balancing (Wang, Yu & Perdikaris 2022) are
   both published. The contribution is the controlled measurement of their
   interaction, plus the σ-bandwidth and denormal findings.

### Limitations

- **Single seed per configuration.** Differences below ~20 % are not separable
  from initialisation variance.
- **Fixed, truncated budget.** Conclusions concern progress in equal iterations,
  **not** asymptotic accuracy. Causal weighting may well pay off at 10⁶ steps.
- Undamped, single-mode, simply-supported only. The damped analytical branch is
  implemented but not swept; the paper's inverse problem is out of scope.
- Full-batch rather than the paper's mini-batch 32.
- Literature review was a best-effort open-web search, not a systematic one.

---

## 8. What is missing (run stopped early)

22 of ~30 models completed. Not run:

| Missing | Would have answered |
|---|---|
| High-frequency sweep, modes 2–3 for Fourier+NTK and proposed | Whether the mode-1 → mode-2 sign flip of the causal effect is systematic |
| Data-efficiency study (4 fractions × 2 models) | Error vs amount of observed training data |
| Noise robustness (0/1/5 % × 2 models) | Whether NTK weighting degrades gracefully under noisy observations |
| Final dashboard + final research table | Presentation only — all underlying numbers are in `results/tables/` |

To complete them: re-run the notebook. All 22 finished models are cached in
`results/checkpoints/`, so only the missing runs will train.

---

## Figures

| File | Content |
|---|---|
| `06_analytical_overview_mode{1,2,3}.png` | Mode shape, snapshots, space-time heatmap, mid-span response, FFT, frequency ladder |
| `05_dataset_distributions.png` | Train/val/collocation sampling, stratified vs uniform time coverage |
| `07_sigma_t2_sweep.png` | Accuracy and frequency error vs Fourier temporal scale |
| `08_ntk_weights_and_convergence.png` | Adaptive $\lambda_i$ over training; vanilla vs Fourier vs Fourier+NTK convergence |
| `09_error_vs_time.png` | Where error lives in $t^*$ (2.50× late/early ratio) |
| `10_causal_weight_mechanism.png` | Causal weight profiles vs $\varepsilon$ |
| `11_causal_strength_sweep.png` | Accuracy vs causality strength |
| `12_causal_weights_training.png` | Weight front sweeping through training |
| `13_ablation.png` | 2×2 bars, interaction plot, convergence curves |
| `report_M{1,2,3,4}_*.png` | Per-model 8-panel diagnostics: analytical vs prediction, slices, error map, antinode response, loss history, residual distribution, convergence |
