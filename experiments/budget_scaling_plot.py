"""Collapse time, L2_exact and PDE residual vs cumulative PDE evaluations for the budget runs
(reads tables/phaseY1_20K_checkpoints.csv and tables/phaseZ4_20K_checkpoints.csv)."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "experiments"))
from phaseE_figures import style, BG, INK, MUTED
RO = ROOT / "results_optimization"
series = {}
for tab in ("phaseY1_20K", "phaseZ4_20K"):
    for r in csv.DictReader(open(RO / "tables" / f"{tab}_checkpoints.csv")):
        if r["run"] in ("Y1_20K", "Z4_20K"):
            series.setdefault(r["run"], []).append(r)
COL = {"Y1_20K": "#eb6834", "Z4_20K": "#4a3aa7"}
LAB = {"Y1_20K": "Y1 recipe, mini-batch 32", "Z4_20K": "Z4 recipe, mini-batch 128"}
for key, ylab, fname, logy in (("collapse_time_s", "collapse time [s] (local amplitude ratio < 0.5)", "collapse", False),
                               ("L2_exact", "L2_exact (201x2001 grid)", "L2", True),
                               ("PDE_residual_rel", "relative PDE residual RMS(r)/RMS(u_tt)", "pde_residual", True)):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG); style(ax)
    for run, rows in series.items():
        x = [float(r["pde_evaluations_cum"]) for r in rows]; y = [float(r[key]) for r in rows]
        (ax.semilogy if logy else ax.plot)(x, y, "-o", color=COL[run], lw=2, ms=5)
        for r, xx, yy in zip(rows, x, y):
            ax.annotate(f"{int(r['step'])//1000}k", (xx, yy), xytext=(0, 7), textcoords="offset points",
                        fontsize=7, color=COL[run], ha="center")
        ax.annotate(LAB[run], (x[-1], y[-1]), xytext=(8, -3), textcoords="offset points", fontsize=8, color=COL[run])
    if key == "collapse_time_s":
        ax.axhline(1.0, color=INK, lw=1, ls=":"); ax.set_ylim(0, 1.05)
        ax.text(5e4, 0.97, "full window = 1 s (20.6 cycles)", fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("cumulative PDE-residual evaluations", fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    ax.set_xlim(0, 3.3e6)
    fig.tight_layout(); fig.savefig(RO / "figures" / f"budget_{fname}_vs_pde_evals.png", dpi=150, facecolor=BG); plt.close(fig)
print("ok")
