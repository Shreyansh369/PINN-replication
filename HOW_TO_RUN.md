# How to run

## 1. Install

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
```

That installs `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `jupyter`,
`nbformat` and `tabulate`.

> **`tabulate` matters.** Without it `DataFrame.to_markdown()` raises and the
> table-saving helper degrades to CSV only. The notebook handles its absence
> gracefully, but install it to get the `.md` tables.

CUDA is used automatically if available; otherwise CPU. Nothing needs changing.

---

## 2. Run

### Interactively

```bash
jupyter lab beam_pinn_research.ipynb
```

Then *Run All*. It executes top to bottom with no manual edits.

### Headless

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output beam_pinn_research_executed.ipynb \
  beam_pinn_research.ipynb
```

---

## 3. Choose a budget first

The top code cell of Section 0:

```python
FAST_MODE         = False   # True -> tiny smoke test, NOT research results
REPRODUCTION_MODE = False   # True -> paper-scale, very slow on CPU
USE_CACHE         = True    # reuse finished checkpoints
```

| Mode | Network | Iterations | Wall time (4 CPU threads) | Use |
|---|---|---|---|---|
| `FAST_MODE = True` | 3×64 | 300–400 | **~10 min** | smoke test / debugging |
| both `False` *(default)* | 4×200 | 1000–4000 | **~3.5 h** | what produced `results/` |
| `REPRODUCTION_MODE = True` | 4×200 | 8000–30000 | **days on CPU** | needs a GPU |

Measured rate on 4 CPU threads: **~0.12 s per iteration** for a 4×200 network
with 512 collocation points and 4th-order derivatives.

**Start with `FAST_MODE = True`** to confirm the environment works (~10 min),
then switch it off for the real run.

---

## 4. Caching — how to resume

Every run is keyed by an MD5 of its full `RunConfig` (excluding only the label),
so re-running the notebook retrains **only what changed**:

```
results/checkpoints/run_<hash>.pt
```

- Interrupted run? Just re-execute — finished models load instantly.
- Want a clean retrain? `rm -rf results/checkpoints/` or set `USE_CACHE = False`.
- Changing *any* config field (learning rate, iterations, σ, seed, mode, tag)
  produces a new hash and retrains that run.

The shipped `results/` contains 22 trained models. Re-running the notebook as-is
will reuse them and train only the missing ones (see REPORT.md §8).

---

## 5. Outputs

Everything is written automatically; no value needs copying out of a plot.

```
results/
  figures/      PNG plots (15 files)
  tables/       CSV + Markdown tables (14 files)
  metrics/      JSON metrics, incl. REPRODUCIBILITY.json, ablation_effects.json
  checkpoints/  trained weights + training history  (gitignored, *.pt)
  logs/         per-run config JSON and per-run history CSV
```

Start with `results/tables/16_all_runs_summary.csv` for a one-line-per-run
overview.

---

## 6. Reading the notebook

26 sections, markdown before every code block. The order that matters:

| Want | Go to |
|---|---|
| What is being asked | §1 research objective |
| Paper values + the 3 inconsistencies found | §2 parameter registry |
| The maths (PDE, BCs, non-dimensionalization) | §3 |
| Analytical reference + derivatives | §4 |
| Unit tests (run before any training) | §6b |
| Baseline PINN + residual by autograd | §7 |
| Fourier features | §11 |
| NTK weighting + estimator validation | §12 |
| **Reproduction comparison** | §13 |
| **Paper's own 1 s window, and the gap** | §13.6 |
| Literature review of candidate optimizations | §15 |
| Proposed method + its mathematics | §17 |
| **2×2 ablation and interaction effect** | §18 |
| Conclusions (generated from measured arrays) | §24 |
| **Viva preparation** | §25 |
| Reproducibility check | §26 |

---

## 7. Changing the experiment

All sweep ranges and budgets live in **one cell** (§0.2):

```python
MODES_TO_TEST   = [1, 2, 3]
SIGMA_T2_SWEEP  = [1, 5, 10, 20, 40]
WMIN_SWEEP      = [1.0, 0.5, 0.1, 0.01]   # 1.0 = causality off
DATA_FRACTIONS  = [1.0, 0.5, 0.25, 0.10]
NOISE_LEVELS    = [0.0, 0.01, 0.05]
```

All physical parameters live in the `PARAMS` registry (§2), each labelled
`PAPER (<table>)` or `OURS`.

**To switch to the paper's damped case:** set `b = 50.0` in `BeamParams`. The
underdamped analytical branch is already implemented and tested.

**To raise statistical confidence:** the single biggest weakness is one seed per
configuration. Loop `SEED` over several values and report mean ± spread;
differences below ~20 % are currently not separable from initialisation noise.

---

## 8. Known issues

**Denormal slowdown with strong causal weighting.** At small `w_min`
(large ε) the causal weights enter the float32 subnormal range, and denormal
arithmetic costs ~9.5× on CPU (measured: 12.20 ms vs 1.29 ms per
multiply-reduce). The causal timings in the shipped `results/` are inflated by
this; accuracy is unaffected. To fix, add near the top of Section 0:

```python
torch.set_flush_denormal(True)
```

**Long runs on CPU.** Fourth derivatives through a 200-neuron network dominate
the cost and scale poorly with threads (4 threads give ~3× over 1). A GPU is
strongly preferred for `REPRODUCTION_MODE`.
