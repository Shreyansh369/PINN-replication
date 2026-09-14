# PINN replication — Euler–Bernoulli beam vibration

Replication study and controlled optimization experiment for:

> Cem Söyleyici and Hakkı Özgür Ünver,
> *"A Physics-Informed Deep Neural Network based beam vibration framework for
> simulation and parameter identification"*,
> **Engineering Applications of Artificial Intelligence 141 (2025) 109804**.
> DOI: [10.1016/j.engappai.2024.109804](https://doi.org/10.1016/j.engappai.2024.109804)

**Start here:**

| Document | What it is |
|---|---|
| **[REPORT.md](REPORT.md)** | Full write-up: maths, paper parameters, all results, conclusions, limitations |
| **[HOW_TO_RUN.md](HOW_TO_RUN.md)** | Install, budgets, caching, outputs, known issues |
| `beam_pinn_research.ipynb` | The notebook itself — runs top to bottom, no manual edits |
| `results/` | 15 figures, 14 tables, metrics, per-run logs (22 trained models) |

> **Run status:** the experiment run was stopped early by request. 22 of ~30
> models completed; the reproduction comparison, paper-window replication and
> the 2x2 ablation are all complete. REPORT.md section 8 lists exactly what is
> missing. Re-running the notebook reuses the cached models and trains only the
> remainder.

---

## Read this before quoting any number

The paper PDF has been read; all parameters come from it (Appendix A / Table
A.10 — this is the **simply-supported** case, not the fixed-end main text).

- The paper's **method** is reproduced and verified against its own equations:
  the architecture (Eqs. 38-43) and the NTK weight rule (Eq. 37) match.
- The paper's **numbers** are **not** reproduced: ours 9.71e-1 against the
  paper's 2.30e-3 on the paper's own 1 s window. That is a ~250x training-budget
  deficit plus full-batch training (the regime the paper itself reports as
  worse). Stated plainly in REPORT.md section 5.2, not glossed over.
- **The proposed optimization does not work.** A controlled 2x2 gives an
  antagonistic interaction of +0.8749, cancelling the NTK main effect of
  -0.8796. Reported as a negative result with a measured mechanism.

---

## What the study does

| Stage | Notebook section |
|---|---|
| Analytical ground truth (closed form, no solver) | 4 |
| Baseline: vanilla tanh PINN | 7–10 |
| Enhanced baseline: Fourier features + NTK weighting (the paper's method) | 11–13 |
| Research question, formed from measured weaknesses | 14 |
| Literature-checked candidate optimizations | 15–16 |
| Proposed: + temporal causal weighting | 17 |
| Controlled 2×2 ablation with interaction effect | 18 |
| High-frequency sweep (modes 1–3) | 19 |
| Data efficiency and noise robustness (additional, not in the paper) | 20–21 |
| Dashboard, final table, conclusions, viva notes | 22–26 |

### Honesty rules the notebook follows

- Ground truth is analytical and never touches the network.
- No paper metric is invented; missing ones read `N/A (PDF unavailable)`.
- The proposed optimization is classified **Combination**, not novel — both
  ingredients are published, and the literature table in Section 15 cites them.
- Conclusions in Section 24 are generated from the measured arrays, so the
  notebook prints "IMPROVED" or "NOT IMPROVED" according to the data.
- Test data never influences training; `ε` and `σ_t2` are tuned on validation.

---

## Running it

```bash
pip install -r requirements.txt
jupyter lab beam_pinn_research.ipynb        # or:
jupyter nbconvert --to notebook --execute beam_pinn_research.ipynb
```

### Execution modes (top cell of Section 0)

| Flag | Budget | Use |
|---|---|---|
| `FAST_MODE = True` | small net, few hundred iterations | smoke test — **not research results** |
| both `False` (default) | paper architecture (4×200 tanh), reduced iteration count | what produced the committed results |
| `REPRODUCTION_MODE = True` | paper architecture, long schedule | closest to paper scale; slow on CPU |

`USE_CACHE = True` reuses checkpoints in `results/checkpoints/`, keyed by a hash
of the full run configuration, so re-running only retrains what changed. Delete
that directory to force a clean retrain.

CUDA is used automatically when available; otherwise CPU.

---

## Outputs

```
results/
  figures/      PNG plots
  tables/       CSV + Markdown tables
  metrics/      JSON metrics, including REPRODUCIBILITY.json
  checkpoints/  model weights + training history
  logs/         per-run config JSON and history CSV
```

## Known limitations

- Single seed per configuration — differences below roughly 20 % are not
  separable from initialisation variance.
- Reduced training budget on CPU: conclusions about convergence *speed* at a
  fixed budget are supported; conclusions about *asymptotic* accuracy are not.
- Undamped, single-mode, simply-supported beam only. The damped analytical
  branch is implemented but not swept. The paper's inverse/parameter-identification
  problem is out of scope.
- The literature review was a best-effort open-web search, not a systematic
  database review.

## Scope

Deliberately excluded: railway applications, axle bearings, SHM deployment, and
the final railway architecture.

---

## Repository layout

```
beam_pinn_research.ipynb   the deliverable — self-contained, runs top to bottom
notebook_src/              build scripts that generate the notebook
                           (python make_notebook.py rewrites the .ipynb)
requirements.txt
results/                   generated outputs (created on first run)
```

`notebook_src/` exists only so the notebook can be regenerated and diffed
sensibly in git; the notebook itself has no dependency on it and needs no
supporting module at runtime.
