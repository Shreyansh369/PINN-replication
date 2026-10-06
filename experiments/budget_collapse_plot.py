"""Collapse time vs cumulative PDE evaluations (reads tables/phaseY1_20K_checkpoints.csv)."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "experiments"))
from phaseE_figures import style, BG, INK, MUTED
RO = ROOT / "results_optimization"
rows = list(csv.DictReader(open(RO / "tables" / "phaseY1_20K_checkpoints.csv")))
y = [r for r in rows if r["run"] == "Y1_20K"]
o = [r for r in rows if r["run"] != "Y1_20K"]
fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=BG); style(ax)
ax.plot([float(r["pde_evaluations_cum"]) for r in y], [float(r["collapse_time_s"]) for r in y], "-o", color="#eb6834", lw=2, ms=6)
for r in y:
    ax.annotate(f"{int(r['step'])//1000}k steps", (float(r["pde_evaluations_cum"]), float(r["collapse_time_s"])),
                xytext=(-10, 9), textcoords="offset points", fontsize=7, color=INK)
ax.annotate("Y1-20K (mini-batch 32)", (float(y[-1]["pde_evaluations_cum"]), float(y[-1]["collapse_time_s"])),
            xytext=(8, -14), textcoords="offset points", fontsize=8, color="#eb6834")
offs = {"Y1_X2_lrsched": (8, -12), "Z1_Y1_rad": (8, 6), "Z2_Y1_seed1235": (8, -24), "Z4_Y1_mb128": (-150, 12)}
for r in o:
    xx, yy = float(r["pde_evaluations_cum"]), float(r["collapse_time_s"])
    ax.plot([xx], [yy], "D", color=MUTED, ms=6)
    ax.annotate(f"{r['run']} (mb {r['mini_batch']}, 5k steps)", (xx, yy), xytext=offs.get(r["run"], (8, 0)),
                textcoords="offset points", fontsize=7, color=MUTED)
ax.axhline(1.0, color=INK, lw=1, ls=":"); ax.text(2e4, 0.97, "full window = 1 s (20.6 cycles)", fontsize=7, color=MUTED, va="top")
ax.set_xlim(0, 7.2e5); ax.set_ylim(0, 1.05)
ax.set_xlabel("cumulative PDE-residual evaluations", fontsize=8); ax.set_ylabel("collapse time [s]  (local amplitude ratio < 0.5)", fontsize=8)
fig.tight_layout(); fig.savefig(RO / "figures" / "phaseY1_20K_collapse_vs_steps.png", dpi=150, facecolor=BG)
print("ok")
