"""Quantify the PAPER-FAITHFUL vs EXACT-PHYSICS references for FE-D-M1 (no training).

Outputs results_optimization/tables/stage01_reference_comparison.csv and prints a summary.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from beampinn.evaluation.metrics import grid, rel_l2  # noqa: E402
from beampinn.physics.benchmarks import frequency_hz, get_benchmark  # noqa: E402

bm = get_benchmark("FE-D-M1")
R = {k: bm.reference(k) for k in ("paper", "exact", "material")}
TARGET = bm.paper_L2
rows = []


def add(**k):
    rows.append(k)
    print("  " + "  ".join(f"{a}={b:.6g}" if isinstance(b, float) else f"{a}={b}" for a, b in k.items()))


print("References (FE-D-M1):")
for k, c in R.items():
    print(f"  {k:9s} c2={c.c2:.4f} gamma={c.gamma:.5f} beta1*l={c.beta_l:.9f} "
          f"f1={frequency_hz(c):.6f} Hz  wd={c.omega_d:.6f} rad/s")

print("\nPairwise relative L2 (first argument = 'prediction', second = reference):")
for nx, nt in [(201, 2001), (101, 1001), (401, 4001)]:
    _, _, X, T = grid(bm.L, bm.t_end, nx, nt)
    late = T >= 0.5
    for a, b in [("exact", "paper"), ("material", "paper"), ("material", "exact")]:
        Ua, Ub = R[a].u(X, T), R[b].u(X, T)
        add(grid=f"{nx}x{nt}", pred=a, ref=b, L2=rel_l2(Ua, Ub), L2_late=rel_l2(Ua[late], Ub[late]),
            max_abs_over_A0=float(np.abs(Ua - Ub).max() / bm.A0),
            rel_freq_diff=(R[a].omega_d - R[b].omega_d) / R[b].omega_d,
            target=TARGET, floor_over_target=rel_l2(Ua, Ub) / TARGET)

x = np.linspace(0, bm.L, 20001)
ic_diff = float(np.abs(R["paper"].u0(x) - R["exact"].u0(x)).max() / bm.A0)
ic_l2 = rel_l2(R["paper"].u0(x), R["exact"].u0(x))
slope = abs(R["paper"].u(bm.L, 0.0, 1, 0)) * bm.L / bm.A0
print(f"\nIC mode-shape difference paper vs exact: max {ic_diff:.3e} A0, rel-L2 {ic_l2:.3e}")
print(f"Paper reference slope-BC violation at x=L: |u_x| L / A0 = {slope:.3e}")
add(grid="1-D x (20001)", pred="paper_u0", ref="exact_u0", L2=ic_l2, L2_late=float("nan"),
    max_abs_over_A0=ic_diff, rel_freq_diff=0.0, target=TARGET, floor_over_target=float("nan"))

# floor decomposition: frequency-only vs shape-only contributions (201 x 2001)
_, _, X, T = grid(bm.L, bm.t_end, 201, 2001)
from beampinn.physics.beam import modal_time  # noqa: E402
Up = R["paper"].u(X, T)
U_freq_only = bm.A0 * R["exact"].mode_shape(X) * modal_time(R["paper"].omega, R["paper"].gamma, T)
U_shape_only = bm.A0 * R["paper"].mode_shape(X) * modal_time(R["exact"].omega, R["exact"].gamma, T)
f_only = rel_l2(R["exact"].u(X, T), U_freq_only)
s_only = rel_l2(R["exact"].u(X, T), U_shape_only)
floor = rel_l2(R["exact"].u(X, T), Up)
print(f"\nFloor decomposition on 201x2001: total {floor:.4e} | frequency part {f_only:.4e} | shape part {s_only:.4e}")
add(grid="201x2001", pred="exact", ref="paper(freq only)", L2=f_only, L2_late=float("nan"),
    max_abs_over_A0=float("nan"), rel_freq_diff=(R["paper"].omega_d - R["exact"].omega_d) / R["exact"].omega_d,
    target=TARGET, floor_over_target=f_only / TARGET)
add(grid="201x2001", pred="exact", ref="paper(shape only)", L2=s_only, L2_late=float("nan"),
    max_abs_over_A0=float("nan"), rel_freq_diff=0.0, target=TARGET, floor_over_target=s_only / TARGET)

# Headroom: L2_paper of prediction P = ||P - Up|| / ||Up||. With P = Ue + E:
#   triangle inequality: floor - e <= L2_paper <= floor + e   (e = ||E||/||Up||)
#   if E is orthogonal to (Ue - Up): L2_paper = sqrt(floor^2 + e^2)
e_worst = TARGET - floor
e_orth = math.sqrt(max(TARGET ** 2 - floor ** 2, 0.0))
print(f"\nHeadroom for L2_paper < {TARGET:g} (floor {floor:.4e} = {floor/TARGET:.1%} of target):")
print(f"  guaranteed if L2_exact < {e_worst:.3e}  (triangle inequality, worst case)")
print(f"  if the PINN error is orthogonal to the reference discrepancy: L2_exact < {e_orth:.3e}")
add(grid="201x2001", pred="headroom", ref="paper", L2=floor, L2_late=float("nan"),
    max_abs_over_A0=float("nan"), rel_freq_diff=float("nan"), target=TARGET,
    floor_over_target=floor / TARGET, worst_case_L2_exact_needed=e_worst, orthogonal_L2_exact_needed=e_orth)

out = ROOT / "results_optimization" / "tables" / "stage01_reference_comparison.csv"
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {out}")
