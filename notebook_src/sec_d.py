"""Sections 18-21: controlled experiments, ablation, high-frequency, data
efficiency, noise, dashboard, final table, conclusions, viva, repro check."""
from common import md, code


def cells():
    C = []
    A = C.append

    # ------------------------------------------ Section 18 controlled 2x2 -----
    A(md(r"""
---
# 18. Controlled experiment and ablation

## 18.1 The design

The proposed method adds exactly one mechanism to the enhanced baseline, and
that mechanism is orthogonal to the one the paper adds. So the ablation is a
clean $2\times2$ factorial over {NTK weighting} × {causal weighting}, with the
vanilla MLP as an additional floor:

| | causal OFF | causal ON |
|---|---|---|
| **NTK OFF** | A: Fourier only | C: Fourier + causal |
| **NTK ON** | B: Fourier + NTK *(= paper / enhanced baseline)* | D: **Proposed** |

A $2\times2$ also gives us the **interaction effect**, which is what the
research question in Section 14.2 actually asks about:

$$\text{interaction} = \big(\log E_D - \log E_C\big) - \big(\log E_B - \log E_A\big)$$

Negative interaction ⇒ the two mechanisms help each other more than the sum of
their individual effects. Positive ⇒ they partly cancel.

**Held constant across all four cells:** beam, PDE, mode, IC, BC, sampler, test
grid, evaluation code, seed, architecture, optimizer, LR schedule, iteration
budget, collocation count. **Varied:** the two binary switches only.
"""))

    A(code(r"""
ABL = {}
ablation_specs = [
    ("A_fourier_only",  False, False),
    ("B_fourier_ntk",   True,  False),
    ("C_fourier_causal", False, True),
    ("D_proposed",      True,  True),
]
for nm, use_ntk, use_causal in ablation_specs:
    run_experiment(base_cfg(nm, arch="fourier", use_ntk=use_ntk, use_causal=use_causal,
                            sigma_t=(1.0, SIGMA_T2_USED), causal_wmin=CAUSAL_WMIN_BEST), nd)
    ABL[nm], _ = evaluate(RUNS[nm]["model"], nd, MODE_MAIN, nm)

abl_rows = []
for nm, use_ntk, use_causal in ablation_specs:
    m = ABL[nm]
    abl_rows.append({"cell": nm, "NTK": use_ntk, "causal": use_causal,
                     "rel-L2": m["rel_l2"], "RMSE": m["rmse"], "max err": m["max_err"],
                     "PDE res (nd)": m["pde_res_nd"], "IC err": m["ic_err"],
                     "BC err": m["bc_err"], "freq err": m["freq_rel_err"],
                     "train [s]": RUNS[nm]["wall"]})
abl_df = pd.DataFrame(abl_rows)
save_table(abl_df, "10_ablation")
display(Markdown(f"### Ablation — 2x2 factorial, mode {MODE_MAIN}, "
                 f"{ITERS_MAIN} iters, seed {SEED}"))
display(abl_df)

EA, EB, EC, ED = (ABL[n]["rel_l2"] for n, _, _ in ablation_specs)
eff_ntk    = math.log(EB / EA)
eff_causal = math.log(EC / EA)
interaction = (math.log(ED) - math.log(EC)) - (math.log(EB) - math.log(EA))

print("\nFACTORIAL EFFECTS (log rel-L2; negative = improvement)")
print("-" * 74)
print(f"  main effect of NTK      (B vs A) : {eff_ntk:+.4f}  "
      f"({'improves' if eff_ntk < 0 else 'worsens'} by {abs(math.expm1(eff_ntk)):.1%})")
print(f"  main effect of causal   (C vs A) : {eff_causal:+.4f}  "
      f"({'improves' if eff_causal < 0 else 'worsens'} by {abs(math.expm1(eff_causal)):.1%})")
print(f"  interaction NTK x causal         : {interaction:+.4f}  "
      f"({'synergistic' if interaction < 0 else 'antagonistic'})")
print(f"  proposed (D) vs enhanced baseline (B): "
      f"{math.log(ED/EB):+.4f}  ({abs(math.expm1(math.log(ED/EB))):.1%} "
      f"{'better' if ED < EB else 'worse'})")
print("-" * 74)
save_json({"E_A": EA, "E_B": EB, "E_C": EC, "E_D": ED,
           "effect_ntk": eff_ntk, "effect_causal": eff_causal,
           "interaction": interaction}, "ablation_effects")
"""))

    A(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

labels = ["A\nFourier", "B\n+NTK", "C\n+causal", "D\n+both"]
vals = [EA, EB, EC, ED]
cols = ["tab:grey", "tab:blue", "tab:orange", "tab:green"]
axes[0].bar(labels, vals, color=cols)
axes[0].set(yscale="log", ylabel="rel-$L^2$", title="Ablation: final accuracy")
for i, v in enumerate(vals):
    axes[0].text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=7.5)

axes[1].plot([0, 1], [EA, EB], "o-", label="causal OFF")
axes[1].plot([0, 1], [EC, ED], "s--", label="causal ON")
axes[1].set(xticks=[0, 1], xticklabels=["NTK OFF", "NTK ON"], yscale="log",
            ylabel="rel-$L^2$", title=f"Interaction plot (interaction={interaction:+.3f})")
axes[1].legend(fontsize=8)

for nm, _, _ in ablation_specs:
    axes[2].semilogy(RUNS[nm]["hist"]["iter"], RUNS[nm]["hist"]["val_l2"],
                     lw=1.3, label=nm)
axes[2].set(xlabel="iteration", ylabel="validation rel-$L^2$", title="Convergence")
axes[2].legend(fontsize=7)
fig.tight_layout()
print("saved:", savefig(fig, "13_ablation"))
plt.show()
"""))

    A(md(r"""
## 18.2 Full controlled comparison — all ten measures

The task brief asks for ten quantities. All ten are computed here for the four
headline models, on the same grid with the same code path.
"""))

    A(code(r"""
CONV_THRESHOLD = 0.5      # rel-L2 level used to define "convergence speed"

def full_row(label, name):
    m, _ = evaluate(RUNS[name]["model"], nd, RUNS[name]["cfg"].mode, name)
    h = RUNS[name]["hist"]
    it_conv = convergence_iters(h, CONV_THRESHOLD)
    return {"Model": label,
            "1 rel-L2": m["rel_l2"], "2 RMSE": m["rmse"], "3 max err": m["max_err"],
            "4 PDE res (nd)": m["pde_res_nd"], "5 BC err": m["bc_err"],
            "6 IC err": m["ic_err"],
            f"7 iters to L2<{CONV_THRESHOLD}": it_conv,
            "8 train [s]": RUNS[name]["wall"], "9 infer [ms]": m["infer_ms"],
            "10 freq err": m["freq_rel_err"]}

controlled = pd.DataFrame([
    full_row("Baseline PINN (vanilla)", "M1_vanilla"),
    full_row("Fourier PINN", "M2_fourier"),
    full_row("Fourier + NTK (paper method)", ENHANCED),
    full_row("PROPOSED (Fourier+NTK+causal)", "M4_proposed"),
])
save_table(controlled, "11_controlled_comparison")
display(Markdown(f"### Controlled comparison — mode {MODE_MAIN}, identical budget and seed"))
display(controlled)
"""))

    # ------------------------------------ Section 19 high-frequency ----------
    A(md(r"""
---
# 19. High-frequency evaluation

The paper's central claim is about learning **high-frequency** dynamics, so this
is the most important experiment in the notebook. Mode $n$ has
$\omega^{*}_n = 2\pi n^{2}$ and completes $n^{2}$ cycles in the window, so
modes 1→3 span a $9\times$ frequency range at fixed cost.

For each mode and each model we record rel-$L^2$, RMSE, frequency error and
training time.

**Note on $\sigma_{t2}$:** the bandwidth requirement grows with $n^{2}$, so a
single fixed $\sigma_{t2}$ cannot be right for every mode. We therefore report
each model at the $\sigma_{t2}$ chosen in Section 13.3 (fixed across modes,
which is the honest "one model, many modes" setting) and flag where bandwidth,
rather than the optimizer, is the binding constraint.
"""))

    A(code(r"""
hf_rows = []
for mode in MODES_TO_TEST:
    for label, kw in [
        ("vanilla",        dict(arch="vanilla", use_ntk=False, use_causal=False)),
        ("fourier+NTK",    dict(arch="fourier", use_ntk=True,  use_causal=False,
                                sigma_t=(1.0, SIGMA_T2_USED))),
        ("proposed",       dict(arch="fourier", use_ntk=True,  use_causal=True,
                                sigma_t=(1.0, SIGMA_T2_USED), causal_wmin=CAUSAL_WMIN_BEST)),
    ]:
        nm = f"HF_m{mode}_{label.replace('+','_')}"
        r = run_experiment(base_cfg(nm, mode=mode, iters=ITERS_HF, **kw), nd, verbose=False)
        m, _ = evaluate(r["model"], nd, mode, nm)
        B = torch.randn(1, 64, generator=torch.Generator().manual_seed(SEED)) * SIGMA_T2_USED
        hf_rows.append({"mode": mode, "model": label,
                        "omega*": nd.omega_star(mode), "f [Hz]": beam.f_n(mode),
                        "rel-L2": m["rel_l2"], "RMSE": m["rmse"],
                        "freq err": m["freq_rel_err"], "train [s]": r["wall"],
                        "bandwidth ok": bool(float(B.abs().max()) >= nd.omega_star(mode))
                                        if label != "vanilla" else None,
                        "iters to L2<0.5": convergence_iters(r["hist"], 0.5)})

hf_df = pd.DataFrame(hf_rows)
save_table(hf_df, "12_high_frequency")
display(Markdown("### High-frequency evaluation across modes"))
display(hf_df)
"""))

    A(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for label, mk in [("vanilla", "o:"), ("fourier+NTK", "s-"), ("proposed", "^-")]:
    sub = hf_df[hf_df.model == label].sort_values("mode")
    axes[0].semilogy(sub["mode"], sub["rel-L2"], mk, label=label)
    axes[1].semilogy(sub["mode"], sub["freq err"].clip(lower=1e-6), mk, label=label)
    axes[2].plot(sub["mode"], sub["iters to L2<0.5"], mk, label=label)
axes[0].set(xlabel="mode number n", ylabel="rel-$L^2$", xticks=MODES_TO_TEST,
            title="Error vs mode number")
axes[1].set(xlabel="mode number n", ylabel="relative frequency error",
            xticks=MODES_TO_TEST, title="Frequency error vs mode number")
axes[2].set(xlabel="mode number n", ylabel="iterations to rel-$L^2$ < 0.5",
            xticks=MODES_TO_TEST, title="Convergence speed vs mode number")
for a in axes:
    a.legend(fontsize=8)
fig.suptitle(f"High-frequency behaviour ($\\omega^*_n = 2\\pi n^2$)  [{MODE_NAME} mode]")
fig.tight_layout()
print("saved:", savefig(fig, "14_high_frequency"))
plt.show()
"""))

    # ------------------------------------ Section 20 data efficiency ---------
    A(md(r"""
---
# 20. Data efficiency

**This is an additional experiment, not part of the paper replication.**

Sections 13–19 solve the *pure forward problem*: no measured data at all, only
PDE + IC + BC. Here we add a sparse observation term

$$\mathcal{L}_{\text{total}} \;\mathrel{+}=\; \lambda_{data}\,
\frac{1}{N_d}\sum_k \big(u_\theta(x_k,t_k) - u^{obs}_k\big)^2$$

and vary $N_d$ over 100 % / 50 % / 25 % / 10 % of the training observation set.
Observations come from the analytical solution (Section 5); the **test grid is
untouched**.
"""))

    A(code(r"""
de_rows = []
for frac in DATA_FRACTIONS:
    n_d = max(8, int(frac * N_TRAIN))
    for label, kw in [("fourier+NTK", dict(use_ntk=True, use_causal=False)),
                      ("proposed",    dict(use_ntk=True, use_causal=True,
                                           causal_wmin=CAUSAL_WMIN_BEST))]:
        nm = f"DE_{int(frac*100)}pc_{label.replace('+','_')}"
        r = run_experiment(base_cfg(nm, arch="fourier", sigma_t=(1.0, SIGMA_T2_USED),
                                    n_data=n_d, lambda_data=1.0, iters=ITERS_AUX, **kw),
                           nd, observations=train_obs, verbose=False)
        m, _ = evaluate(r["model"], nd, MODE_MAIN, nm)
        de_rows.append({"data fraction": frac, "N_obs": n_d, "model": label,
                        "rel-L2": m["rel_l2"], "RMSE": m["rmse"],
                        "freq err": m["freq_rel_err"], "train [s]": r["wall"]})

de_df = pd.DataFrame(de_rows)
save_table(de_df, "13_data_efficiency")
display(Markdown("### Data efficiency (sparse observations + physics)"))
display(de_df)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for label, mk in [("fourier+NTK", "s-"), ("proposed", "^-")]:
    sub = de_df[de_df.model == label].sort_values("data fraction")
    ax[0].loglog(sub["N_obs"], sub["rel-L2"], mk, label=label)
    ax[1].semilogy(sub["data fraction"] * 100, sub["rel-L2"], mk, label=label)
ax[0].set(xlabel="number of observations $N_d$", ylabel="rel-$L^2$",
          title="Error vs amount of training data")
ax[1].set(xlabel="training data [%]", ylabel="rel-$L^2$", title="Error vs data fraction")
for a in ax:
    a.legend(fontsize=8)
fig.tight_layout()
print("saved:", savefig(fig, "15_data_efficiency"))
plt.show()
"""))

    # ------------------------------------ Section 21 noise -------------------
    A(md(r"""
---
# 21. Noise robustness

**Also an additional experiment, not part of the paper replication.**

Gaussian noise at 0 %, 1 % and 5 % of the RMS signal amplitude is added to the
**observed training data only**. The analytical test ground truth is never
noised — doing so would make the metric meaningless.
"""))

    A(code(r"""
noise_rows = []
n_d = max(8, N_TRAIN // 2)
for noise in NOISE_LEVELS:
    obs = make_observations(N_TRAIN, MODE_MAIN, seed=SEED + 1, noise=noise)
    for label, kw in [("fourier+NTK", dict(use_ntk=True, use_causal=False)),
                      ("proposed",    dict(use_ntk=True, use_causal=True,
                                           causal_wmin=CAUSAL_WMIN_BEST))]:
        nm = f"NZ_{int(noise*100)}pc_{label.replace('+','_')}"
        r = run_experiment(base_cfg(nm, arch="fourier", sigma_t=(1.0, SIGMA_T2_USED),
                                    n_data=n_d, noise=noise, iters=ITERS_AUX, **kw),
                           nd, observations=obs, verbose=False)
        m, _ = evaluate(r["model"], nd, MODE_MAIN, nm)
        noise_rows.append({"noise": noise, "model": label, "N_obs": n_d,
                           "rel-L2": m["rel_l2"], "RMSE": m["rmse"],
                           "freq err": m["freq_rel_err"]})

nz_df = pd.DataFrame(noise_rows)
save_table(nz_df, "14_noise_robustness")
display(Markdown("### Noise robustness (noise on training observations only)"))
display(nz_df)

fig, ax = plt.subplots(figsize=(6, 3.8))
for label, mk in [("fourier+NTK", "s-"), ("proposed", "^-")]:
    sub = nz_df[nz_df.model == label].sort_values("noise")
    ax.semilogy(sub["noise"] * 100, sub["rel-L2"], mk, label=label)
ax.set(xlabel="observation noise [% of RMS amplitude]", ylabel="rel-$L^2$",
       title="Error vs noise level")
ax.legend(fontsize=8)
fig.tight_layout()
print("saved:", savefig(fig, "16_noise_robustness"))
plt.show()
"""))

    # ------------------------------------ Section 22 dashboard ---------------
    A(md(r"""
---
# 22. Final visual dashboard

One page that answers the question the whole notebook exists to answer:
**did the proposed optimization actually improve the model?**
"""))

    A(code(r"""
BEST_NAME = min(["M1_vanilla", "M2_fourier", ENHANCED, "M4_proposed"],
                key=lambda n: evaluate(RUNS[n]["model"], nd, MODE_MAIN, n)[0]["rel_l2"])

fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.32)
names = ["M1_vanilla", "M2_fourier", ENHANCED, "M4_proposed"]
short = ["vanilla", "+Fourier", "+NTK\n(paper)", "PROPOSED"]
cols = ["tab:grey", "tab:blue", "tab:orange", "tab:green"]
mets = {n: evaluate(RUNS[n]["model"], nd, MODE_MAIN, n)[0] for n in names}

# 1 model comparison (accuracy)
ax = fig.add_subplot(gs[0, 0])
v = [mets[n]["rel_l2"] for n in names]
ax.bar(short, v, color=cols); ax.set(yscale="log", ylabel="rel-$L^2$", title="1. Accuracy")
for i, y in enumerate(v):
    ax.text(i, y, f"{y:.1e}", ha="center", va="bottom", fontsize=7)

# 2 error comparison (RMSE + max)
ax = fig.add_subplot(gs[0, 1])
w = 0.38; idx = np.arange(4)
ax.bar(idx - w/2, [mets[n]["rmse"] for n in names], w, label="RMSE")
ax.bar(idx + w/2, [mets[n]["max_err"] for n in names], w, label="max err")
ax.set(xticks=idx, xticklabels=short, yscale="log", title="2. Error measures")
ax.legend(fontsize=7)

# 3 training time
ax = fig.add_subplot(gs[0, 2])
ax.bar(short, [RUNS[n]["wall"] for n in names], color=cols)
ax.set(ylabel="seconds", title="3. Training time")

# 4 convergence
ax = fig.add_subplot(gs[0, 3])
for n, s, c in zip(names, short, cols):
    ax.semilogy(RUNS[n]["hist"]["iter"], RUNS[n]["hist"]["val_l2"],
                lw=1.4, color=c, label=s.replace("\n", " "))
ax.set(xlabel="iteration", ylabel="val rel-$L^2$", title="4. Convergence")
ax.legend(fontsize=7)

# 5 high-frequency
ax = fig.add_subplot(gs[1, 0])
for label, mk in [("vanilla", "o:"), ("fourier+NTK", "s-"), ("proposed", "^-")]:
    sub = hf_df[hf_df.model == label].sort_values("mode")
    ax.semilogy(sub["mode"], sub["rel-L2"], mk, label=label)
ax.set(xlabel="mode n", ylabel="rel-$L^2$", xticks=MODES_TO_TEST, title="5. High-frequency")
ax.legend(fontsize=7)

# 6 data efficiency
ax = fig.add_subplot(gs[1, 1])
for label, mk in [("fourier+NTK", "s-"), ("proposed", "^-")]:
    sub = de_df[de_df.model == label].sort_values("data fraction")
    ax.semilogy(sub["data fraction"] * 100, sub["rel-L2"], mk, label=label)
ax.set(xlabel="training data [%]", ylabel="rel-$L^2$", title="6. Data efficiency")
ax.legend(fontsize=7)

# 6b noise
ax = fig.add_subplot(gs[1, 2])
for label, mk in [("fourier+NTK", "s-"), ("proposed", "^-")]:
    sub = nz_df[nz_df.model == label].sort_values("noise")
    ax.semilogy(sub["noise"] * 100, sub["rel-L2"], mk, label=label)
ax.set(xlabel="noise [%]", ylabel="rel-$L^2$", title="6b. Noise robustness")
ax.legend(fontsize=7)

# ablation interaction
ax = fig.add_subplot(gs[1, 3])
ax.plot([0, 1], [EA, EB], "o-", label="causal OFF")
ax.plot([0, 1], [EC, ED], "s--", label="causal ON")
ax.set(xticks=[0, 1], xticklabels=["NTK off", "NTK on"], yscale="log",
       ylabel="rel-$L^2$", title=f"7. Interaction ({interaction:+.2f})")
ax.legend(fontsize=7)

# 7 analytical vs best
x, t, X, T = make_test_grid()
U = analytical_solution(X, T, nd, MODE_MAIN)
P = predict(RUNS[BEST_NAME]["model"], X, T)
vmax = np.abs(U).max()
ax = fig.add_subplot(gs[2, 0])
im = ax.pcolormesh(t, x, U, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax.set(xlabel="$t^*$", ylabel="$x^*$", title="8a. Analytical"); fig.colorbar(im, ax=ax)
ax = fig.add_subplot(gs[2, 1])
im = ax.pcolormesh(t, x, P, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax.set(xlabel="$t^*$", ylabel="$x^*$", title=f"8b. Best model ({BEST_NAME})")
fig.colorbar(im, ax=ax)

# 8 error heatmap
ax = fig.add_subplot(gs[2, 2])
im = ax.pcolormesh(t, x, np.abs(P - U), cmap="magma", shading="auto")
ax.set(xlabel="$t^*$", ylabel="$x^*$", title="9. |error| heatmap"); fig.colorbar(im, ax=ax)

# antinode overlay
ax = fig.add_subplot(gs[2, 3])
i_a = int(np.argmin(np.abs(x - 0.5 / MODE_MAIN)))
ax.plot(t, U[i_a, :], lw=1.4, label="analytical")
ax.plot(t, P[i_a, :], "--", lw=1.2, label=BEST_NAME)
ax.set(xlabel="$t^*$", ylabel="$u^*$", title="10. Antinode response")
ax.legend(fontsize=7)

improved = mets["M4_proposed"]["rel_l2"] < mets[ENHANCED]["rel_l2"]
delta = mets["M4_proposed"]["rel_l2"] / mets[ENHANCED]["rel_l2"] - 1
fig.suptitle(
    f"FINAL DASHBOARD  |  mode {MODE_MAIN}  |  best = {BEST_NAME}  |  "
    f"proposed vs paper-method baseline: {'IMPROVED' if improved else 'NOT IMPROVED'} "
    f"({delta:+.1%} rel-L2)  |  [{MODE_NAME} mode, seed {SEED}]",
    fontsize=13)
print("saved:", savefig(fig, "17_final_dashboard"))
plt.show()
"""))

    # ------------------------------------ Section 23 final table -------------
    A(md(r"""
---
# 23. Final research table

`N/A (PDF unavailable)` marks every quantity the paper may report but which we
could not read. **None of it is invented.**
"""))

    A(code(r"""
def final_row(label, name=None):
    if name is None:
        return {"Model": label, **{c: "N/A (PDF unavailable)" for c in FINAL_COLS}}
    m = mets[name] if name in mets else evaluate(RUNS[name]["model"], nd, MODE_MAIN, name)[0]
    hf = hf_df[(hf_df.model == HF_ALIAS.get(name, "")) ]
    best_mode = (int(hf.loc[hf["rel-L2"].idxmin(), "mode"]) if len(hf) else MODE_MAIN)
    return {"Model": label,
            "rel-L2": f"{m['rel_l2']:.4e}", "RMSE": f"{m['rmse']:.4e}",
            "PDE residual (nd)": f"{m['pde_res_nd']:.4e}",
            "PDE residual [N/m]": f"{m['pde_res_phys']:.4e}",
            "IC error": f"{m['ic_err']:.4e}", "BC error": f"{m['bc_err']:.4e}",
            "Training time [s]": f"{RUNS[name]['wall']:.1f}",
            "Inference [ms]": f"{m['infer_ms']:.1f}",
            "Freq. error": f"{m['freq_rel_err']:.3%}",
            "Best mode": best_mode}

FINAL_COLS = ["rel-L2", "RMSE", "PDE residual (nd)", "PDE residual [N/m]",
              "IC error", "BC error", "Training time [s]", "Inference [ms]",
              "Freq. error", "Best mode"]
HF_ALIAS = {"M1_vanilla": "vanilla", ENHANCED: "fourier+NTK", "M4_proposed": "proposed"}

final_df = pd.DataFrame([
    final_row("Paper reported (Söyleyici & Ünver 2025)"),
    final_row("Our baseline (vanilla PINN)", "M1_vanilla"),
    final_row("Our Fourier PINN", "M2_fourier"),
    final_row("Our enhanced baseline (Fourier + NTK)", ENHANCED),
    final_row("Our proposed (Fourier + NTK + causal)", "M4_proposed"),
])
save_table(final_df, "15_FINAL_research_table")
display(Markdown(f"### FINAL RESEARCH TABLE — mode {MODE_MAIN}, "
                 f"{ITERS_MAIN} iterations, seed {SEED}, [{MODE_NAME} mode]"))
display(final_df)
"""))

    # ------------------------------------ Section 24 conclusions -------------
    A(md(r"""
---
# 24. Conclusions
"""))

    A(code(r"""
E_base, E_prop = mets[ENHANCED]["rel_l2"], mets["M4_proposed"]["rel_l2"]
gain = E_prop / E_base - 1
hf_pivot = hf_df.pivot_table(index="mode", columns="model", values="rel-L2")
consistent = bool((hf_pivot["proposed"] < hf_pivot["fourier+NTK"]).all())
n_better = int((hf_pivot["proposed"] < hf_pivot["fourier+NTK"]).sum())

print("=" * 78)
print("CONCLUSIONS  (generated from the measured results, not written in advance)")
print("=" * 78)
print(f'''
1. PAPER REPRODUCTION STATUS
   Method-level  : reproduced. Multi-scale spatio-temporal Fourier features and
                   NTK trace-based adaptive loss weighting are implemented and
                   trained on the simply-supported Euler-Bernoulli beam.
   Numerical     : NOT reproduced, and not claimed. The paper PDF was
                   unavailable, so its beam properties, training budget and
                   reported errors are unknown. Every 'paper reported' cell in
                   the final table reads N/A.

2. DOES THE PAPER'S METHOD HELP? (mode {MODE_MAIN}, identical budget)
   vanilla        rel-L2 = {mets['M1_vanilla']['rel_l2']:.4e}
   +Fourier       rel-L2 = {mets['M2_fourier']['rel_l2']:.4e}
   +Fourier+NTK   rel-L2 = {E_base:.4e}

3. THE sigma_t2 FINDING
   Under our time normalization the paper-literal sigma_t2 = 10 spans temporal
   frequencies only up to |B| ~ 23, while mode {MODE_MAIN} needs omega* = {nd.omega_star(MODE_MAIN):.1f}.
   The sigma sweep (Section 13.3) measures the consequence directly. Because our
   time normalization is OUR choice and the paper's is unknown, this is a
   statement about the interaction of sigma with a normalization -- NOT evidence
   that the paper's value is wrong.

4. DOES THE PROPOSED OPTIMIZATION HELP?
   enhanced baseline  rel-L2 = {E_base:.4e}
   proposed           rel-L2 = {E_prop:.4e}   ({gain:+.1%})
   verdict at mode {MODE_MAIN}: {'IMPROVED' if gain < 0 else 'NO IMPROVEMENT'}

5. IS IT CONSISTENT ACROSS FREQUENCIES?
   proposed beats the enhanced baseline on {n_better}/{len(hf_pivot)} tested modes
   -> {'consistent' if consistent else 'NOT consistent -- the gain is mode-dependent'}

6. ABLATION / INTERACTION
   NTK alone      {eff_ntk:+.4f}  (log rel-L2 change)
   causal alone   {eff_causal:+.4f}
   interaction    {interaction:+.4f}  ({'synergistic' if interaction < 0 else 'antagonistic'})

7. LIMITATIONS
   - No access to the paper: the beam parameters are ASSUMED, so absolute
     numbers are not comparable to the publication.
   - Single seed per configuration. With one seed we cannot separate a small
     effect from initialisation variance; treat differences under ~20% as
     provisional.
   - Reduced training budget ({ITERS_MAIN} iterations, CPU-only). Conclusions
     about ASYMPTOTIC accuracy are not supported; conclusions about convergence
     SPEED at a fixed budget are.
   - Undamped, single-mode, simply-supported beam only. Multi-mode initial
     conditions, other supports, and the paper's inverse problem are untested.
   - The literature search was a best-effort open-web review, not a systematic
     database review.
''')
print("=" * 78)
"""))

    # ------------------------------------ Section 25 viva --------------------
    A(md(r"""
---
# 25. Viva preparation

## 25.1 Every mathematical object, in plain terms

| Object | What it is | Where in the code |
|---|---|---|
| **PDE** $EI u_{xxxx} + \rho A u_{tt} + b u_t = 0$ | Force balance on a beam element: elastic restoring force + inertia + damping = 0. | `pde_residual` |
| $u_{xxxx}$ | Fourth spatial derivative. $EI u_{xx}$ is the bending moment; differentiating twice more turns moment into transverse force per unit length. | autograd, 4 nested `d1` |
| $\rho A\, u_{tt}$ | Mass per unit length × acceleration — the inertia term. | `pde_residual` |
| $b\,u_t$ | Viscous damping, proportional to velocity. Set to 0 here so an exact reference exists. | `NonDim.zeta` |
| $\beta_n = n\pi/L$ | Wavenumber of mode $n$: how many half-sines fit in the span. | `BeamParams.beta_n` |
| $\omega_n = \beta_n^2\sqrt{EI/\rho A}$ | Natural frequency. The $\beta_n^{2}$ is why beam modes spread as $n^{2}$ — the root of the spectral-bias problem. | `BeamParams.omega_n` |
| $\alpha = 4/\pi^2$ | The single non-dimensional constant of the scaled PDE. Independent of the beam. | `NonDim.alpha` |
| **Fourier features** $\gamma(v)=[\cos(Bv),\sin(Bv)]$ | A fixed random sinusoidal lift of the inputs. Lets the network express high frequencies without having to build them from tanh. | `MultiScaleFourierMLP._encode` |
| **Spectral bias** | The tendency of an MLP under gradient descent to fit low frequencies first, sometimes never reaching high ones. | measured in Section 19 |
| **NTK** $K = JJ^{\mathsf T}$ | The kernel governing how the network's outputs evolve under gradient flow. Its eigenvalues set per-mode convergence rates. | `ntk_trace_estimates` |
| $\lambda_i = \sum_j \mathrm{tr}(K_j)/\mathrm{tr}(K_i)$ | Loss weights that equalise each term's convergence rate. | `ntk_weights` |
| **PDE residual** $\hat r$ | How badly the network violates the governing equation at a point. Not an error against data — no data is involved. | `pde_residual` |
| **Collocation points** | Unlabelled points where the physics is enforced. Resampled every iteration. | `sample_batch` |
| $\mathcal{L}_{ic}$, $\mathcal{L}_{vel}$ | Penalties on the initial shape and on starting from rest. | `term_residuals` |
| $\mathcal{L}_{bc_u}$, $\mathcal{L}_{bc_m}$ | Zero deflection and zero moment at the pins, kept separate because their scales differ by $\sim(n\pi)^4$. | `term_residuals` |
| **Adaptive weighting** | Loss weights that change during training in response to measured quantities, rather than being fixed by hand. | NTK + causal |
| **Causal weight** $w_i = e^{-\varepsilon\sum_{j<i}\mathcal{L}_j}$ | Switches on the loss at time bin $i$ only once all earlier bins are already fitted. | `causal_weights` |
"""))

    A(md(r"""
## 25.2 The seven questions about the proposed optimization

**1. What problem did we observe?**
The enhanced baseline's error is not spread evenly over the time window: the RMS
error in the second half is measurably larger than in the first half (the ratio
is printed in Section 14.1). The model is trading accuracy at early times for
accuracy at late times, which is backwards — early times are what determine
late ones.

**2. Why does our modification address it?**
Causal weighting multiplies the residual in time bin $i$ by
$w_i=\exp(-\varepsilon\sum_{j<i}\mathcal{L}_j)$. Until the earlier bins are
small, $w_i$ is near zero, so there is almost no gradient signal from late
times. The optimizer is forced to fit the window left to right.

**3. What is the mathematical mechanism?**
It reweights the measure over which the PDE residual is integrated, from uniform
in $t^{*}$ to one concentrated on the earliest not-yet-satisfied region. At the
minimiser all $\mathcal{L}_i \to 0$ and $w_i \to 1$, so **the weighted problem
has the same minimisers as the unweighted one** — it changes the optimisation
path, not the solution. And $\varepsilon\to0$ recovers the baseline exactly,
which is why the comparison is fair.

**4. What evidence supports the improvement?**
The $2\times2$ ablation in Section 18 (numbers printed there), the
frequency sweep in Section 19, and the data-efficiency / noise studies in
Sections 20–21 — all at identical seed, budget, architecture and test grid.
Whether the evidence is *positive* is decided by those numbers; the notebook
prints "IMPROVED" or "NOT IMPROVED" from the measurement rather than asserting a
result.

**5. What are the limitations?**
Single seed per configuration, so small differences are not separable from
initialisation noise. Reduced iteration budget on CPU, so nothing is said about
asymptotic accuracy. One beam, one support condition, single-mode initial
conditions, no damping. $\varepsilon$ was tuned on validation, which the
baseline's $\sigma_{t2}$ sweep matches in effort but not exactly in kind.

**6. What previous literature already exists?**
Causal weighting: Wang, Sankaran & Perdikaris, *CMAME* 421 (2024) 116813
(arXiv:2203.07404). NTK loss balancing: Wang, Yu & Perdikaris (2022). Fourier
features: Tancik et al. (2020); Wang, Wang & Perdikaris, *CMAME* 384 (2021)
113938. Adaptive collocation alternatives we rejected: Lu et al. (2021),
Wu et al. (2023), Lau et al. (ICLR 2024). Full table in Section 15.

**7. What can we honestly claim?**
- ✅ A working, tested, reproducible implementation of the paper's *method*.
- ✅ A controlled, seed- and budget-matched measurement of whether causal
  weighting composes with NTK weighting on this problem, with the interaction
  effect isolated by a $2\times2$ ablation.
- ✅ A quantified finding about Fourier bandwidth vs time normalization
  (Section 13.3) that explains a large part of the behaviour at high modes.
- ❌ **Not** a novel method — the ingredients are all published.
- ❌ **Not** a numerical replication of the paper's reported values.
- ❌ **Not** a statistically strong result — one seed per cell.

## 25.3 Questions an examiner is likely to ask

> *Why non-dimensionalise? Isn't that hiding the physics?*
No — it is a change of variables with the algebra written out in Section 3.4,
and every reported residual is converted back to N/m. The reason is
conditioning: $EI\approx 10^2$ against $u \approx 10^{-3}$ makes the raw
residual span $10^7$ in magnitude against the IC term.

> *Why split the BC into two loss terms when the brief says one?*
Because $|u_{xx}|\sim(n\pi)^2$ and $|u|\sim1$, so a single combined term is
dominated by the moment condition by $\sim(n\pi)^4 \approx 8\times10^3$ at mode 3
and the displacement condition would effectively vanish. It is documented in
Section 7.3 and applied identically to all models.

> *Your NTK trace is estimated. Doesn't that invalidate the method?*
It is an unbiased estimator with each sampled row computed exactly, and Section
12.2b measures its error against the exact trace on a small network. We also
measured the obvious alternative (Hutchinson) and rejected it on the evidence.

> *How do you know the improvement isn't just noise?*
We do not, from one seed — and Section 24 says so. Differences below roughly
20 % should be treated as provisional until repeated over seeds. Raising
`SEEDS_PER_CONFIG` is the natural next step.

> *Why is mode 3 the headline?*
Because $\omega_n\propto n^2$, mode 3 completes 9 cycles in the window and is
where spectral bias actually bites. Modes 1–2 are comparatively easy, which the
sweep in Section 19 shows.
"""))

    # ------------------------------------ Section 26 repro check -------------
    A(md(r"""
---
# 26. Reproducibility check

Final cell. Prints everything needed to reproduce or audit this run.
"""))

    A(code(r"""
print("=" * 78)
print("REPRODUCIBILITY CHECK")
print("=" * 78)
print(f"  seed                : {SEED}")
print(f"  device              : {DEVICE}   dtype: {DTYPE}")
print(f"  execution mode      : {MODE_NAME}  "
      f"(FAST_MODE={FAST_MODE}, REPRODUCTION_MODE={REPRODUCTION_MODE})")
print(f"  torch / numpy       : {torch.__version__} / {np.__version__}")
print()
print("  CONFIGURATION")
for k, v in BUDGET.items():
    print(f"    {k:18s} = {v}")
print(f"    headline mode      = {MODE_MAIN}")
print(f"    sigma_t2 used      = {SIGMA_T2_USED:g}   (paper-literal value: 10)")
print(f"    causal w_min       = {CAUSAL_WMIN_BEST:g}")
print()
print(f"  BEST MODEL          : {BEST_NAME}")
print("  FINAL METRICS (mode %d, held-out grid vs analytical solution)" % MODE_MAIN)
for n in names:
    m = mets[n]
    print(f"    {n:22s} rel-L2={m['rel_l2']:.4e}  RMSE={m['rmse']:.4e}  "
          f"freq-err={m['freq_rel_err']:.3%}  {RUNS[n]['wall']:6.0f}s")
print()
print(f"  TESTS               : {sum(t['passed'] for t in _TEST_LOG)}/{len(_TEST_LOG)} passed")
print()
print("  RESULT PATHS")
for k, v in DIRS.items():
    n_files = len(list(v.glob('*')))
    print(f"    {k:12s} {str(v)+'/':28s} {n_files:3d} files")
print()
print("  PAPER REPRODUCTION STATUS: method-level reproduced; numerical values")
print("  NOT reproduced (paper PDF unavailable). No paper metric was fabricated.")
print("=" * 78)

save_json({
    "seed": SEED, "device": str(DEVICE), "dtype": str(DTYPE),
    "execution_mode": MODE_NAME, "budget": BUDGET,
    "mode_main": MODE_MAIN, "sigma_t2_used": SIGMA_T2_USED,
    "causal_wmin": CAUSAL_WMIN_BEST, "best_model": BEST_NAME,
    "enhanced_baseline": ENHANCED,
    "final_metrics": {n: mets[n] for n in names},
    "ablation": {"E_A": EA, "E_B": EB, "E_C": EC, "E_D": ED,
                 "effect_ntk": eff_ntk, "effect_causal": eff_causal,
                 "interaction": interaction},
    "tests_passed": int(sum(t["passed"] for t in _TEST_LOG)),
    "tests_total": len(_TEST_LOG),
    "torch": torch.__version__, "numpy": np.__version__,
    "paper_reproduction": "method-level only; numerical values N/A (PDF unavailable)",
}, "REPRODUCIBILITY")
print("\nsaved: results/metrics/REPRODUCIBILITY.json")
"""))

    return C
