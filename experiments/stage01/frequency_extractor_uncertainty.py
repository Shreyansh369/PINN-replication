"""Monte-Carlo resolution of the damped-cosine frequency/phase/amplitude extractor (no training).

The exact FE-D-M1 mid-span trace is perturbed by synthetic errors of controlled relative-L2
size eps, and the extractor's error is recorded. This tells us which frequency/phase errors
are resolvable at a given accuracy level, before any such error is used as a gate.

Perturbation families (all scaled so ||e|| / ||y|| = eps on the 2001-sample trace):
  white      i.i.d. Gaussian
  smooth     random sum of 6 damped sinusoids, frequencies U[0, 60] Hz, random phases,
             envelope exp(-gamma t / 2) times a random linear ramp (PINN-like structured error)
  envelope   eps-sized smooth error with the solution's own envelope and frequency band
             (sinusoids within +-2 Hz of f1) - the hardest case for frequency extraction
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from beampinn.evaluation.frequency import compare_fits, fit_damped_cosine  # noqa: E402
from beampinn.physics.benchmarks import get_benchmark  # noqa: E402

bm = get_benchmark("FE-D-M1")
ref = bm.reference("exact")
t = np.linspace(0, bm.t_end, 2001)
y = ref.u(ref.x_norm, t)
a, lam, w, phi = ref.damped_cosine_params()
a *= ref.A0
f1 = w / (2 * math.pi)
rng = np.random.default_rng(20261006)
N_TRIALS = 200
EPS = [1e-5, 1e-4, 2.5e-4, 4.64e-4, 1e-3, 1e-2]


def perturbation(kind):
    if kind == "white":
        e = rng.standard_normal(t.size)
    else:
        e = np.zeros_like(t)
        for _ in range(6):
            fr = rng.uniform(0, 60) if kind == "smooth" else f1 + rng.uniform(-2, 2)
            e += rng.standard_normal() * np.cos(2 * math.pi * fr * t + rng.uniform(0, 2 * math.pi))
        e *= np.exp(-0.5 * ref.gamma * t) * (1 + rng.uniform(-1, 1) * t)
    return e


rows = []
print(f"FE-D-M1 mid-span trace, exact wd = {w:.6f} rad/s, {N_TRIALS} trials per cell")
base = compare_fits(fit_damped_cosine(t, y), a, lam, w, phi)
print(f"noise-free: freq {base['frequency_error']:.2e} phase {base['phase_error']:.2e} "
      f"amp {base['amplitude_error']:.2e} damping {base['damping_error']:.2e}")
rows.append(dict(kind="none", eps=0.0, **{f"{k}_median": v for k, v in base.items()}))
for kind in ("white", "smooth", "envelope"):
    for eps in EPS:
        errs = {k: [] for k in ("frequency_error", "phase_error", "amplitude_error", "damping_error", "frequency_se")}
        for _ in range(N_TRIALS):
            e = perturbation(kind)
            e *= eps * np.linalg.norm(y) / np.linalg.norm(e)
            c = compare_fits(fit_damped_cosine(t, y + e), a, lam, w, phi)
            for k in errs:
                errs[k].append(c[k])
        row = dict(kind=kind, eps=eps)
        for k, v in errs.items():
            v = np.asarray(v)
            row[f"{k}_median"], row[f"{k}_p95"] = float(np.median(v)), float(np.quantile(v, 0.95))
        rows.append(row)
        print(f"  {kind:8s} eps={eps:8.2e}  freq med {row['frequency_error_median']:.2e} p95 {row['frequency_error_p95']:.2e} | "
              f"phase p95 {row['phase_error_p95']:.2e} rad | amp p95 {row['amplitude_error_p95']:.2e} | "
              f"damp p95 {row['damping_error_p95']:.2e} | formal se med {row['frequency_se_median']:.2e}")

out = ROOT / "results_optimization" / "tables" / "stage01_frequency_extractor_uncertainty.csv"
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys)
    wr.writeheader(); wr.writerows(rows)
print(f"wrote {out}")
