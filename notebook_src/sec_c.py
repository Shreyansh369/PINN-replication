"""Sections 13-17: reproduction comparison, diagnostics, research question,
literature-based candidate search, selected optimization, proposed model."""
from common import md, code


def cells():
    C = []
    A = C.append

    # ----------------------------------------------- Section 13 reproduction --
    A(md(r"""
---
# 13. Reproduction comparison

## 13.1 Training budget actually used

Every model below is trained under **identical** conditions except for the one
mechanism under test: same beam, same PDE, same mode, same IC/BC, same sampler,
same optimizer, same LR schedule, same iteration budget, same seed, same test
grid.
"""))

    A(code(r"""
# ---- training budget, set by the execution mode ---------------------------
if FAST_MODE:
    ITERS_MAIN, ITERS_SWEEP, WIDTH, DEPTH = 400, 300, 64, 3
elif REPRODUCTION_MODE:
    ITERS_MAIN, ITERS_SWEEP, WIDTH, DEPTH = 40000, 20000, 200, 4
else:
    ITERS_MAIN, ITERS_SWEEP, WIDTH, DEPTH = ITERS_MAIN_DEFAULT, ITERS_SWEEP_DEFAULT, 200, 4

BUDGET = dict(mode_name=MODE_NAME, iters_main=ITERS_MAIN, iters_sweep=ITERS_SWEEP,
              depth=DEPTH, width=WIDTH, n_collocation=N_COLLOCATION,
              n_ic=N_IC, n_bc=N_BC, seed=SEED, device=str(DEVICE), dtype=str(DTYPE))
save_json(BUDGET, "training_budget", "logs")

print("TRAINING BUDGET IN FORCE")
for k, v in BUDGET.items():
    print(f"  {k:16s} = {v}")
if FAST_MODE:
    print("\n  *** FAST_MODE: these are smoke-test numbers, NOT research results. ***")
elif not REPRODUCTION_MODE:
    print(f"\n  NOTE: paper architecture ({DEPTH}x{WIDTH}, tanh) at a reduced iteration")
    print("  count. Set REPRODUCTION_MODE=True for the long schedule.")


def base_cfg(name, **kw):
    '''A RunConfig with every controlled variable pinned; kw sets only what varies.'''
    d = dict(name=name, mode=MODE_MAIN, depth=DEPTH, width=WIDTH,
             m_fourier=64, sigma_x=1.0, sigma_t=(1.0, 10.0), res_norm=True,
             n_collocation=N_COLLOCATION, n_ic=N_IC, n_bc=N_BC,
             iters=ITERS_MAIN, lr=1e-3, lr_gamma=0.1,
             ntk_every=200, ntk_rows=16, ntk_beta=0.5,
             causal_bins=32, causal_wmin=CAUSAL_WMIN, seed=SEED)
    d.update(kw)
    return RunConfig(**d)
"""))

    A(md(r"""
## 13.2 The three reproduction models

| Model | Architecture | Loss weighting |
|---|---|---|
| **M1 Vanilla PINN** | plain tanh MLP | fixed $\lambda_i = 1$ |
| **M2 + Fourier features** | multi-scale spatio-temporal Fourier | fixed $\lambda_i = 1$ |
| **M3 + Fourier + NTK** | multi-scale spatio-temporal Fourier | NTK-adaptive $\lambda_i$ |

M3 is the paper's method — our **ENHANCED BASELINE**.
"""))

    A(code(r"""
t_repro = time.time()
run_experiment(base_cfg("M1_vanilla",   arch="vanilla", use_ntk=False), nd)
run_experiment(base_cfg("M2_fourier",   arch="fourier", use_ntk=False), nd)
run_experiment(base_cfg("M3_fourier_ntk", arch="fourier", use_ntk=True), nd)
print(f"\nreproduction block wall time: {(time.time()-t_repro)/60:.1f} min")
"""))

    A(code(r"""
metrics_repro = {}
for nm in ("M1_vanilla", "M2_fourier", "M3_fourier_ntk"):
    metrics_repro[nm] = model_report(nm, nd, MODE_MAIN)
"""))

    A(md(r"""
## 13.3 Diagnostic: does the paper's $\sigma_{t2}$ span our frequency window?

Section 11.3 predicted that $\sigma_{t2}=10$ cannot reach $\omega^{*}_3=56.5$
under our time normalization. Rather than assume, we sweep $\sigma_{t2}$ with
everything else fixed. This is also the honest answer to "why does our result
differ from the paper's": if the paper normalised time differently, its
$\sigma_{t2}=10$ corresponds to a *different* effective bandwidth than ours.
"""))

    A(code(r"""
sigma_sweep = []
for st2 in SIGMA_T2_SWEEP:
    nm = f"S_sigma{st2:g}"
    r = run_experiment(base_cfg(nm, arch="fourier", use_ntk=True,
                                sigma_t=(1.0, float(st2)), iters=ITERS_SWEEP),
                       nd, verbose=False)
    m, _ = evaluate(r["model"], nd, MODE_MAIN, nm)
    B = torch.randn(1, 64, generator=torch.Generator().manual_seed(SEED)) * st2
    sigma_sweep.append({"sigma_t2": st2, "max |B|": float(B.abs().max()),
                        "spans omega*_3": bool(B.abs().max() >= nd.omega_star(MODE_MAIN)),
                        "rel_L2": m["rel_l2"], "freq_rel_err": m["freq_rel_err"],
                        "RMSE": m["rmse"]})

sigma_df = pd.DataFrame(sigma_sweep)
save_table(sigma_df, "06_sigma_t2_sweep")
display(Markdown(f"### $\\sigma_{{t2}}$ sweep at mode {MODE_MAIN} "
                 f"({ITERS_SWEEP} iters, everything else fixed)"))
display(sigma_df)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].semilogy(sigma_df["sigma_t2"], sigma_df["rel_L2"], "o-")
ax[0].axvline(10, color="tab:red", ls="--", label="paper $\\sigma_{t2}=10$")
ax[0].axvline(nd.omega_star(MODE_MAIN) / 2, color="tab:green", ls=":",
              label="$\\omega^*_3/2$")
ax[0].set(xlabel="$\\sigma_{t2}$", ylabel="rel-$L^2$",
          title=f"Accuracy vs temporal Fourier scale (mode {MODE_MAIN})")
ax[0].legend(fontsize=8)
ax[1].semilogy(sigma_df["sigma_t2"], sigma_df["freq_rel_err"].clip(lower=1e-6), "s-",
               color="tab:purple")
ax[1].axvline(10, color="tab:red", ls="--")
ax[1].set(xlabel="$\\sigma_{t2}$", ylabel="relative frequency error",
          title="Frequency error vs $\\sigma_{t2}$")
fig.tight_layout()
print("saved:", savefig(fig, "07_sigma_t2_sweep"))
plt.show()

BEST_SIGMA_T2 = float(sigma_df.loc[sigma_df["rel_L2"].idxmin(), "sigma_t2"])
print(f"\nBest sigma_t2 at mode {MODE_MAIN} under OUR normalization: {BEST_SIGMA_T2:g}")
print(f"Paper-literal sigma_t2 = 10 gives rel-L2 = "
      f"{float(sigma_df.loc[sigma_df.sigma_t2==10,'rel_L2'].iloc[0]):.4e}")
"""))

    A(md(r"""
### 13.3b Which $\sigma_{t2}$ do we carry forward, and why

Two models are kept from here on, and both are reported:

- **M3 (paper-literal)** — $\sigma_{t2}=10$ exactly as supplied. This is the
  faithful reading of the paper.
- **M3b (bandwidth-matched)** — $\sigma_{t2}$ set to the sweep optimum. This is
  labelled `OURS`, not a paper value.

The optimization study in Sections 16–19 uses whichever of the two is the
*stronger* enhanced baseline, because improving on a crippled baseline would be
a meaningless result. Which one that is, is decided by the numbers above, and
stated explicitly below.
"""))

    A(code(r"""
if BEST_SIGMA_T2 != 10.0:
    run_experiment(base_cfg("M3b_fourier_ntk_bw", arch="fourier", use_ntk=True,
                            sigma_t=(1.0, BEST_SIGMA_T2)), nd)
    metrics_repro["M3b_fourier_ntk_bw"] = model_report("M3b_fourier_ntk_bw", nd, MODE_MAIN)
    ENHANCED = ("M3b_fourier_ntk_bw"
                if metrics_repro["M3b_fourier_ntk_bw"]["rel_l2"]
                   < metrics_repro["M3_fourier_ntk"]["rel_l2"]
                else "M3_fourier_ntk")
else:
    ENHANCED = "M3_fourier_ntk"

SIGMA_T2_USED = RUNS[ENHANCED]["cfg"].sigma_t[1]
print(f"ENHANCED BASELINE for the optimization study: {ENHANCED} "
      f"(sigma_t2 = {SIGMA_T2_USED:g})")
print(f"  rel-L2 = {metrics_repro[ENHANCED]['rel_l2']:.4e}")
"""))

    A(md(r"""
## 13.4 Reproduction table

Paper-reported values are `N/A (PDF unavailable)`. **No paper metric is
invented.** The comparison we *can* make is method-level: does adding Fourier
features help, and does adding NTK weighting help on top of that?
"""))

    A(code(r"""
def metric_row(label, met, wall, extra=None):
    row = {"Model": label,
           "rel-L2": met["rel_l2"], "RMSE": met["rmse"], "max err": met["max_err"],
           "PDE res (nd)": met["pde_res_nd"], "PDE res [N/m]": met["pde_res_phys"],
           "IC err": met["ic_err"], "vel-IC err": met["vel_err"], "BC err": met["bc_err"],
           "freq err": met["freq_rel_err"],
           "train [s]": wall, "infer [ms]": met["infer_ms"]}
    if extra:
        row.update(extra)
    return row

repro_rows = [{"Model": "Paper (Söyleyici & Ünver 2025)", "rel-L2": np.nan, "RMSE": np.nan,
               "max err": np.nan, "PDE res (nd)": np.nan, "PDE res [N/m]": np.nan,
               "IC err": np.nan, "vel-IC err": np.nan, "BC err": np.nan,
               "freq err": np.nan, "train [s]": np.nan, "infer [ms]": np.nan}]
for nm, met in metrics_repro.items():
    repro_rows.append(metric_row(nm, met, RUNS[nm]["wall"]))

repro_df = pd.DataFrame(repro_rows)
repro_disp = repro_df.copy()
repro_disp.iloc[0, 1:] = "N/A (PDF unavailable)"
save_table(repro_df, "07_reproduction_comparison")
display(Markdown(f"### Reproduction comparison — mode {MODE_MAIN}, "
                 f"{ITERS_MAIN} iterations, seed {SEED} [{MODE_NAME} mode]"))
display(repro_disp)

print("\nPAPER REPRODUCTION STATUS")
print("-" * 74)
print("  Method-level reproduction : IMPLEMENTED and RUN")
print("      (multi-scale Fourier features + NTK trace-based adaptive weighting,")
print("       Euler-Bernoulli beam, simply supported, single-mode IC)")
print("  Numerical reproduction    : NOT POSSIBLE IN THIS ENVIRONMENT")
print("      reason: the paper PDF was unavailable, so the beam properties,")
print("      training budget and reported error values are unknown to us.")
print(f"  Our rel-L2 (vanilla)      : {metrics_repro['M1_vanilla']['rel_l2']:.4e}")
print(f"  Our rel-L2 (+Fourier)     : {metrics_repro['M2_fourier']['rel_l2']:.4e}")
print(f"  Our rel-L2 (+Fourier+NTK) : {metrics_repro['M3_fourier_ntk']['rel_l2']:.4e}")
if ENHANCED != "M3_fourier_ntk":
    print(f"  Our rel-L2 (bandwidth-matched): {metrics_repro[ENHANCED]['rel_l2']:.4e}")
print("  Paper reported rel-L2     : N/A (PDF unavailable) -- NOT fabricated")
print("-" * 74)
"""))

    A(md(r"""
## 13.5 The adaptive weights over training

This is the plot that shows what NTK weighting actually does.
"""))

    A(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
h = RUNS[ENHANCED]["hist"]
for k in [c for c in h if c.startswith("lam_")]:
    axes[0].semilogy(h["iter"], h[k], lw=1.3, label="$\\lambda_{" + k[4:].replace("_", "\\_") + "}$")
axes[0].set(xlabel="iteration", ylabel="$\\lambda_i$",
            title=f"NTK adaptive loss weights ({ENHANCED})")
axes[0].legend(fontsize=8, ncol=2)

for nm, style in [("M2_fourier", "--"), (ENHANCED, "-")]:
    axes[1].semilogy(RUNS[nm]["hist"]["iter"], RUNS[nm]["hist"]["val_l2"],
                     style, lw=1.4, label=nm)
axes[1].semilogy(RUNS["M1_vanilla"]["hist"]["iter"], RUNS["M1_vanilla"]["hist"]["val_l2"],
                 ":", lw=1.4, label="M1_vanilla")
axes[1].set(xlabel="iteration", ylabel="validation rel-$L^2$",
            title="Convergence: vanilla vs Fourier vs Fourier+NTK")
axes[1].legend(fontsize=8)
fig.tight_layout()
print("saved:", savefig(fig, "08_ntk_weights_and_convergence"))
plt.show()
"""))

    # -------------------------------------------- Section 14 research question -
    A(md(r"""
---
# 14. Research question for the optimization

## 14.1 Grounded in what we just measured

The question below is written **after** looking at Sections 13.2–13.5, not
before. The cell prints the observations it rests on, so the reasoning stays
tied to the actual numbers rather than to a story.
"""))

    A(code(r"""
print("OBSERVED WEAKNESSES OF THE ENHANCED BASELINE")
print("=" * 74)
best = metrics_repro[ENHANCED]
h = RUNS[ENHANCED]["hist"]

print(f"1. Absolute accuracy at mode {MODE_MAIN}: rel-L2 = {best['rel_l2']:.4e}")
print(f"2. Frequency error: {best['freq_rel_err']:.3%} "
      f"(predicted {best['f_pred']:.3f} vs exact {best['f_true']:.3f} cycles)")

# where in TIME does the error live?
_, (x, t, X, T, U, P, E, r) = evaluate(RUNS[ENHANCED]["model"], nd, MODE_MAIN)
err_t = np.sqrt((E ** 2).mean(axis=0))
half = len(t) // 2
ratio = err_t[half:].mean() / max(err_t[:half].mean(), 1e-30)
print(f"3. Error growth in time: RMS error in the second half of the window is")
print(f"   {ratio:.2f}x the first half  -> error accumulates as t* increases.")

# does it plateau?
tail = h["val_l2"][-max(3, len(h["val_l2"]) // 5):]
print(f"4. Convergence: validation rel-L2 changed by "
      f"{(tail[0]-tail[-1])/max(tail[0],1e-30):+.1%} over the final fifth of training.")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].semilogy(t, err_t, lw=1.4)
ax[0].axvline(0.5, color="grey", ls=":")
ax[0].set(xlabel="$t^*$", ylabel="RMS error over $x$",
          title=f"Where the error lives in time ({ENHANCED})")
ax[1].semilogy(t, np.sqrt((analytical_solution(X, T, nd, MODE_MAIN) ** 2).mean(axis=0)),
               lw=1.2, label="signal RMS")
ax[1].semilogy(t, err_t, lw=1.4, label="error RMS")
ax[1].set(xlabel="$t^*$", ylabel="RMS", title="Signal vs error in time")
ax[1].legend(fontsize=8)
fig.tight_layout()
print("\nsaved:", savefig(fig, "09_error_vs_time"))
plt.show()

OBSERVED = {"rel_l2": best["rel_l2"], "freq_err": best["freq_rel_err"],
            "late_early_error_ratio": float(ratio)}
save_json(OBSERVED, "observed_weaknesses")
"""))

    A(md(r"""
## 14.2 The research question

Driven by observation 3 above — **error concentrates at late $t^{*}$** — the
question this study asks is:

> **The reproduced Fourier/NTK PINN balances the loss *terms* against each other,
> but treats every instant in the time window as equally important, and its error
> grows monotonically with $t^{*}$. Does enforcing *temporal causality* — requiring
> early times to be fitted before later ones — compose constructively with NTK
> loss balancing, and does the combination improve accuracy, convergence speed
> and high-frequency robustness on the Euler–Bernoulli beam?**

Why this is a well-posed, single-variable question:

- NTK weighting acts **across loss terms** ($\lambda_{ic}$ vs $\lambda_{pde}$ …).
- Causal weighting acts **within the PDE term, across time**.
- The two are mathematically orthogonal, so whether they help each other,
  are redundant, or actively conflict is a genuine open question — and it is
  answerable with a clean $2\times2$ ablation (Section 18).
"""))

    # --------------------------------------- Section 15 literature candidates --
    A(md(r"""
---
# 15. Literature-based optimization search

Before implementing anything, the candidate strategies were checked against
current literature (searches run 2026-09; databases were reachable only through
web search from this environment, so this is a **best-effort review, not a
systematic one** — stated as a limitation, not glossed over).

## 15.1 Candidate table

Novelty categories: **Established** (published, widely used) · **Adaptation**
(published idea applied to a new setting) · **Combination** (two published ideas
composed) · **Potentially novel** (no prior work found).
"""))

    A(code(r"""
candidates = [
 dict(Candidate="Temporal causal weighting",
      Literature="Wang, Sankaran & Perdikaris, CMAME 421 (2024) 116813 'Respecting causality for training PINNs' (arXiv:2203.07404)",
      Changes="Weights the PDE residual by w_i=exp(-eps*sum_{j<i}L(t_j)); early times must be fitted first.",
      Advantage="Directly targets the observed late-time error growth; ~free (one cumsum per step).",
      Risk="eps needs tuning; too large stalls training at t=0.",
      Difficulty="Low", Novelty="Established"),
 dict(Candidate="Residual-based adaptive refinement (RAR / RAD / RAR-D)",
      Literature="Lu et al., SIAM Rev. 63 (2021) [DeepXDE]; Wu et al., CMAME 403 (2023) 115671",
      Changes="Resamples collocation points towards high PDE residual.",
      Advantage="Strong on problems with localised sharp features.",
      Risk="Our solution is globally smooth and the residual is not spatially localised, so little to exploit; adds resampling cost.",
      Difficulty="Low", Novelty="Established"),
 dict(Candidate="NTK-guided point selection (PINNACLE)",
      Literature="Lau et al., ICLR 2024, 'PINNACLE: PINN Adaptive ColLocation and Experimental points selection'",
      Changes="Selects collocation points by an NTK-eigenspectrum convergence criterion.",
      Advantage="Principled; uses the same kernel the paper already computes.",
      Risk="Needs NTK eigendecomposition -- far too expensive for a 4-CPU budget.",
      Difficulty="High", Novelty="Established"),
 dict(Candidate="Fourier-feature / frequency scheduling",
      Literature="Tancik et al., NeurIPS 2020; Wang, Wang & Perdikaris, CMAME 384 (2021) 113938; 'Iterative training of PINNs with Fourier-enhanced features' (arXiv:2510.19399)",
      Changes="Grows sigma (or the active feature bank) during training, low frequencies first.",
      Advantage="Attacks spectral bias at its source.",
      Risk="Overlaps heavily with what the paper's multi-scale sigma already does -- hard to attribute a gain.",
      Difficulty="Medium", Novelty="Established"),
 dict(Candidate="Curriculum learning over mode number",
      Literature="'Utilizing curriculum learning for high-frequency eigenfunction discovery through a PINN', J. Comput. Design & Eng. (2026), doi:10.1093/jcde/qwag056",
      Changes="Train on mode 1, warm-start mode 2, then mode 3.",
      Advantage="Well matched to beam modal structure.",
      Risk="Changes the training PROBLEM, not just the optimizer -- breaks the controlled comparison (each model would see different data).",
      Difficulty="Medium", Novelty="Established"),
 dict(Candidate="Modal / mode-superposition PINN",
      Literature="'Modal-integrated PINNs for time-varying moving oscillator and beam interaction', Eng. Struct. (2025); SpectONet (arXiv:2607.25790)",
      Changes="Bakes sin(n pi x) modal basis into the architecture.",
      Advantage="Would be extremely accurate here.",
      Risk="It encodes the analytical answer into the model -- the comparison would be meaningless, and it does not generalise to the paper's aims.",
      Difficulty="Medium", Novelty="Established"),
 dict(Candidate="Physics-informed neural operators (PINO/FNO)",
      Literature="Li et al., PINO (2021); 'Physics-Informed Neural Networks and Neural Operators for Parametric PDEs' (arXiv:2511.04576)",
      Changes="Learns a solution operator over a parameter family, not one solution.",
      Advantage="Amortises across beams.",
      Risk="Solves a different problem from the paper; needs a large offline dataset.",
      Difficulty="High", Novelty="Established"),
 dict(Candidate="Causal weighting COMPOSED WITH NTK term balancing",
      Literature="Both parts published separately (Wang/Sankaran/Perdikaris 2024; Wang/Yu/Perdikaris 2022). No study found that composes them on an Euler-Bernoulli beam or reports their interaction.",
      Changes="NTK sets inter-term weights; causal weights set intra-PDE temporal weights. Applied simultaneously.",
      Advantage="Addresses the measured weakness (late-time error) without touching the training problem, so the comparison stays controlled.",
      Risk="The two mechanisms may interact badly: causal weighting shrinks L_pde early, which NTK then re-inflates.",
      Difficulty="Low", Novelty="Combination"),
]
cand_df = pd.DataFrame(candidates)
save_table(cand_df, "08_optimization_candidates")
display(Markdown("### Candidate optimizations, with literature status"))
for c in candidates:
    display(Markdown(
        f"**{c['Candidate']}**  — *{c['Novelty']}*\n\n"
        f"- **Literature:** {c['Literature']}\n"
        f"- **What it changes:** {c['Changes']}\n"
        f"- **Expected advantage:** {c['Advantage']}\n"
        f"- **Risk:** {c['Risk']}\n"
        f"- **Implementation difficulty:** {c['Difficulty']}\n"))
"""))

    A(md(r"""
## 15.2 Novelty statement — stated conservatively

**Nothing in this notebook is claimed to be a novel method.**

- Causal weighting is **Established** (Wang, Sankaran & Perdikaris, CMAME 2024).
- NTK trace-based loss balancing is **Established** (Wang, Yu & Perdikaris, 2022)
  and is already the paper's own contribution.
- Composing the two is classified **Combination**. Our literature search did not
  find a study that reports their *interaction* on a beam-vibration PINN — but
  "we did not find it" is not "it does not exist", and the search was
  best-effort over an open-web index, not a systematic database review.

What this study can honestly claim is an **empirical result**: a controlled,
seed-matched, budget-matched measurement of whether these two mechanisms compose
constructively on this problem, with an ablation that separates their effects.
That is a legitimate contribution at bachelor-thesis level, and it does not
depend on the method being new.
"""))

    # ------------------------------------------- Section 16 selected optimization
    A(md(r"""
---
# 16. Selected optimization

**Selected: temporal causal weighting of the PDE residual, applied on top of the
paper's Fourier + NTK model.**

Checked against the five selection criteria:

| Criterion | How it is met |
|---|---|
| 1. Addresses a real, measured weakness | Section 14.1 observation 3: error in the second half of the time window is measurably larger than the first half. |
| 2. Mathematically explainable | It is a re-weighting of the residual measure in $t$; the mechanism is written out in 17.1. |
| 3. Fast to implement | One `cumsum` and one `exp` per iteration; negligible cost. |
| 4. Fairly comparable | It changes **only** the weighting of an existing loss term. Same architecture, data, sampler, optimizer, budget and seed. |
| 5. Plausible contribution | The $2\times2$ interaction with NTK weighting is not something we found measured anywhere. |

**Rejected, with reasons** (from the table above): RAR/RAD — our residual is not
spatially localised, so there is little to refine; PINNACLE — NTK eigendecomposition
is out of budget; modal PINN — it embeds the analytical answer and would make the
comparison vacuous; mode-curriculum — it changes the training problem and would
break the controlled comparison; PINO — a different problem entirely.

Only **one** mechanism is added. We are not stacking tricks to chase a number.
"""))

    # -------------------------------------------------- Section 17 proposed ----
    A(md(r"""
---
# 17. The proposed model

## 17.1 What changed, why, how, and the mathematics

**BASELINE (enhanced):**  Fourier features + NTK adaptive loss weighting
**PROPOSED:**             Fourier features + NTK adaptive loss weighting **+ temporal causal weighting**

### WHAT changed
The PDE loss changes from a plain mean over collocation points

$$\mathcal{L}_{pde} = \frac{1}{N_c}\sum_{k} \hat r(x_k,t_k)^2$$

to a **causally weighted** mean over $M$ time bins:

$$\mathcal{L}_{pde}^{\text{causal}} = \frac{1}{M}\sum_{i=1}^{M} w_i\, \mathcal{L}_i,
\qquad
\mathcal{L}_i = \frac{1}{|B_i|}\sum_{k\in B_i} \hat r(x_k,t_k)^2$$

$$\boxed{\;w_i = \exp\!\Big(-\varepsilon \sum_{j<i} \mathcal{L}_j\Big),
\qquad w_i \ \text{treated as a constant (stop-gradient)}\;}$$

where $B_i$ is the set of collocation points whose $t^{*}$ falls in bin $i$.

### WHY it should help
$w_i$ is near 1 only once **all earlier** bins already have small residual. So
the network is not rewarded for reducing the residual at $t^{*}=0.9$ while
$t^{*}=0.1$ is still wrong. This matches how the physics actually propagates:
the solution at a later time is *determined by* the earlier state, and a PINN
trained on all times at once is free to violate that ordering — which is
precisely the late-time error growth measured in Section 14.1.

### HOW it interacts with NTK weighting
They act on orthogonal axes:

$$\mathcal{L}_{\text{total}} = \underbrace{\sum_i \lambda_i}_{\text{NTK: across terms}}
\mathcal{L}_i, \qquad
\mathcal{L}_{pde} = \underbrace{\frac{1}{M}\sum_i w_i}_{\text{causal: across time}} \mathcal{L}_i$$

There is a plausible **conflict**: causal weighting deliberately *shrinks*
$\mathcal{L}_{pde}$ early in training, and NTK weighting responds to small
kernel traces by *inflating* $\lambda_{pde}$. Whether the net effect helps is an
empirical question — which is exactly why Section 18 runs the full $2\times2$.

### The one free parameter, and how we make it meaningful

$\varepsilon$ controls how strictly causality is enforced, and
$\varepsilon\to 0$ recovers the unweighted loss **exactly** — so the proposed
model contains the baseline as a limiting case, which is what makes the
comparison fair.

But $\varepsilon$ carries **units of 1/loss**, so a literature value of $O(1)$
is meaningless until the residual scale is known. We measured this: at our
normalised residual scale the per-bin losses are $\sim10^{-5}$, so
$\varepsilon = 1$ leaves every $w_i = 1$ to five decimal places and the
mechanism is completely inert. Sweeping $\varepsilon \in \{0, 0.1, 1, 10\}$ —
the obvious thing to do — would have produced four identical models and a
confident, wrong conclusion that causality "makes no difference".

**`OURS`: we therefore reparameterise $\varepsilon$ by a dimensionless target.**
Let $w_{\min}$ be the causal weight of the *last* time bin at initialisation.
Then

$$w_{\min} = \exp\!\Big(-\varepsilon \textstyle\sum_{j<M}\mathcal{L}_j\Big)
\quad\Longrightarrow\quad
\boxed{\;\varepsilon = \frac{-\ln w_{\min}}{S_0},\qquad
S_0 = \sum_{j<M}\mathcal{L}_j \ \text{at the first iteration}\;}$$

$\varepsilon$ is computed once from $S_0$ and then held **fixed**, so as the
residual falls the weights relax back towards 1 on their own — the "release"
behaviour the original method intends. The swept quantity $w_{\min}$ is
dimensionless and interpretable ("how strongly is the end of the window
suppressed at the start of training"), and $w_{\min}=1$ gives
$\varepsilon = 0$, i.e. **exactly** the enhanced baseline.

The underlying weighting formula is unchanged from Wang, Sankaran & Perdikaris;
only the parameterisation of $\varepsilon$ is ours, and it is labelled as such.
"""))

    A(code(r"""
def causal_weights(res_pde, t_pde, n_bins, eps):
    '''Temporal causality weights (Wang, Sankaran & Perdikaris, CMAME 2024).

        w_i = exp(-eps * sum_{j<i} L_j),   L_j = mean squared residual in time bin j

    w is DETACHED: no gradient flows through the weights, only through L_j.
    Returns (weights, per-bin loss, weighted scalar loss).
    '''
    idx = torch.clamp((t_pde.reshape(-1) * n_bins).long(), 0, n_bins - 1)
    sq = res_pde.reshape(-1) ** 2

    sums = torch.zeros(n_bins, dtype=sq.dtype, device=sq.device).index_add_(0, idx, sq)
    cnts = torch.zeros(n_bins, dtype=sq.dtype, device=sq.device).index_add_(
        0, idx, torch.ones_like(sq))
    bin_loss = sums / cnts.clamp(min=1.0)

    with torch.no_grad():                                  # stop-gradient on w
        cum = torch.cat([torch.zeros(1, dtype=sq.dtype, device=sq.device),
                         torch.cumsum(bin_loss, 0)[:-1]])
        w = torch.exp(-eps * cum)

    occupied = cnts > 0
    loss = (w * bin_loss)[occupied].sum() / occupied.sum().clamp(min=1)
    return w, bin_loss, loss


def calibrate_causal_eps(res_pde, t_pde, n_bins, w_min):
    '''Pick eps ONCE, at the first iteration, from a dimensionless target.

    eps in the original formula has units of 1/loss, so a literature value of
    O(1) is meaningless until the residual scale is known -- at our scale it
    leaves every w_i = 1 and the mechanism inert. We therefore specify the
    DIMENSIONLESS quantity w_min (the causal weight of the LAST time bin at
    initialisation) and solve for eps:

        w_min = exp(-eps * S_0)      =>      eps = -ln(w_min) / S_0
        S_0   = sum_{j<M} L_j  at the first iteration

    eps is then held FIXED for the whole run, so as the residual falls the
    weights relax back to 1 on their own -- the release behaviour the original
    method intends. w_min = 1 gives eps = 0, i.e. exactly the unweighted loss.
    '''
    if w_min >= 1.0:
        return 0.0
    with torch.no_grad():
        _, bin_loss, _ = causal_weights(res_pde.detach(), t_pde, n_bins, 0.0)
        S0 = float(bin_loss[:-1].sum())
    return float(-math.log(w_min) / max(S0, 1e-30))


# --- sanity checks on the mechanism -----------------------------------------
_r = torch.randn(512, 1) * 0.1
_t = torch.rand(512, 1)
_w0, _, _l0 = causal_weights(_r, _t, 32, 0.0)
_plain = (_r ** 2).mean()
check("12. eps=0 recovers the unweighted PDE loss",
      torch.allclose(_w0, torch.ones_like(_w0)) and abs(float(_l0) - float(_plain)) < 1e-5,
      f"causal={float(_l0):.6e} vs plain={float(_plain):.6e}")
_eps_c = calibrate_causal_eps(_r, _t, 32, 0.1)
_w1, _, _ = causal_weights(_r, _t, 32, _eps_c)
check("12b. eps calibration hits the requested minimum weight",
      abs(float(_w1[-1]) - 0.1) < 0.02, f"w_min achieved = {float(_w1[-1]):.4f}")
check("13. causal weights are non-increasing in time",
      bool((_w1[1:] <= _w1[:-1] + 1e-12).all()))
check("14. causal weights are finite and in (0,1]",
      bool(torch.isfinite(_w1).all() and (_w1 > 0).all() and (_w1 <= 1 + 1e-9).all()))
"""))

    A(md(r"""
## 17.2 Illustration of the mechanism

What the weights look like for a residual profile that grows with time — the
situation we actually measured.
"""))

    A(code(r"""
_tt = torch.linspace(0, 1, 2000).reshape(-1, 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for eps in (0.0, 0.5, 2.0, 10.0):
    prof = torch.exp(3 * _tt) * 0.05           # residual growing with t*
    w, bl, _ = causal_weights(prof, _tt, 32, eps)
    ax[0].plot(np.linspace(0, 1, 32), w.numpy(), "o-", ms=3, label=f"$\\varepsilon$={eps}")
ax[0].set(xlabel="$t^*$ bin centre", ylabel="$w_i$",
          title="Causal weights for a residual that grows with $t^*$")
ax[0].legend(fontsize=8)

ax[1].plot(_tt.numpy(), (torch.exp(3 * _tt) * 0.05).numpy(), lw=1.4)
ax[1].set(xlabel="$t^*$", ylabel="assumed $|\\hat r|$",
          title="Assumed residual profile")
fig.tight_layout()
print("saved:", savefig(fig, "10_causal_weight_mechanism"))
plt.show()
"""))

    A(md(r"""
## 17.3 Choosing the causality strength — on validation only

$w_{\min}$ (equivalently $\varepsilon$) is the single hyperparameter the proposed
method introduces, so it must be tuned without touching the test grid. We sweep
it at a reduced budget and pick by validation rel-$L^2$. Note that $w_{\min}=1$
is in the sweep, so the sweep is allowed to conclude that causality does not help.

**Fairness note.** Giving only the proposed model a hyperparameter sweep would
bias the comparison. The enhanced baseline received its own equivalent sweep in
Section 13.3 ($\sigma_{t2}$), at the same reduced budget and the same number of
trials, and both models then run at the full budget with everything else fixed.
"""))

    A(code(r"""
wmin_rows = []
for wmin in WMIN_SWEEP:
    nm = f"E_wmin{wmin:g}"
    r = run_experiment(base_cfg(nm, arch="fourier", use_ntk=True, use_causal=True,
                                sigma_t=(1.0, SIGMA_T2_USED), causal_wmin=float(wmin),
                                iters=ITERS_SWEEP), nd, verbose=False)
    wmin_rows.append({"w_min": wmin, "calibrated eps": r["hist"].get("causal_eps"),
                      "val rel-L2": r["hist"]["val_l2"][-1], "train [s]": r["wall"]})

wmin_df = pd.DataFrame(wmin_rows)
save_table(wmin_df, "09_causal_strength_sweep")
display(Markdown("### Causality-strength sweep (validation only, reduced budget)"))
display(wmin_df)

CAUSAL_WMIN_BEST = float(wmin_df.loc[wmin_df["val rel-L2"].idxmin(), "w_min"])
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.semilogx(wmin_df["w_min"], wmin_df["val rel-L2"], "o-")
ax.axvline(CAUSAL_WMIN_BEST, color="tab:red", ls="--",
           label=f"selected $w_{{min}}$={CAUSAL_WMIN_BEST:g}")
ax.set(xlabel="$w_{min}$ (1.0 = causality off)", ylabel="validation rel-$L^2$",
       title="Causality strength sweep")
ax.legend(fontsize=8)
fig.tight_layout()
print("saved:", savefig(fig, "11_causal_strength_sweep"))
plt.show()
print(f"Selected w_min = {CAUSAL_WMIN_BEST:g}  "
      f"({'causality OFF -- the sweep prefers the baseline' if CAUSAL_WMIN_BEST >= 1.0 else 'causality ON'})")
print("Chosen on validation only; the test grid was not consulted.")
"""))

    A(md(r"""
## 17.4 Train the proposed model at the full budget
"""))

    A(code(r"""
run_experiment(base_cfg("M4_proposed", arch="fourier", use_ntk=True, use_causal=True,
                        sigma_t=(1.0, SIGMA_T2_USED), causal_wmin=CAUSAL_WMIN_BEST), nd)
metrics_proposed = model_report("M4_proposed", nd, MODE_MAIN)
"""))

    A(code(r"""
# how the causal weights evolved during the real run
ch = RUNS["M4_proposed"]["causal"]
if ch["iter"]:
    W = np.array(ch["w"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    im = ax[0].pcolormesh(np.linspace(0, 1, W.shape[1]), ch["iter"], W,
                          shading="auto", cmap="viridis", vmin=0, vmax=1)
    ax[0].set(xlabel="$t^*$ bin", ylabel="iteration",
              title="Causal weights $w_i$ over training")
    fig.colorbar(im, ax=ax[0], label="$w_i$")
    for frac, lbl in [(0.0, "start"), (0.5, "mid"), (1.0, "end")]:
        j = min(int(frac * (len(ch["iter"]) - 1)), len(ch["iter"]) - 1)
        ax[1].plot(np.linspace(0, 1, W.shape[1]), W[j], "o-", ms=3,
                   label=f"{lbl} (it {ch['iter'][j]})")
    ax[1].set(xlabel="$t^*$ bin", ylabel="$w_i$", title="Weight profile snapshots")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    print("saved:", savefig(fig, "12_causal_weights_training"))
    plt.show()
    print("A weight front that sweeps from left to right is the causal curriculum")
    print("working as intended: later bins only 'switch on' once earlier ones are fit.")
"""))

    return C
