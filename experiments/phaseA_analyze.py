"""Phase A analysis: tables + figures from results_optimization run records (no training).

    python experiments/phaseA_analyze.py [--no-figures]

Convergence slope: least-squares slope of log10 L2 vs log10 step over the validation points in
the last half of the run (and the last quarter). Projection to the target uses that power law;
it is an INDICATION of the trajectory, not a prediction.
"""
import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RO = ROOT / "results_optimization"
TARGET = 4.64e-4
FLOOR = 4.386e-4          # L2_paper of the exact solution (stage01_reference_comparison.csv)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]   # validated, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def load_runs(budgets):
    runs = []
    for mpath in sorted(glob.glob(str(RO / "logs" / "*" / "metrics.json"))):
        m = json.load(open(mpath))
        if m.get("budget_label") not in budgets:
            continue
        with open(Path(mpath).parent / "history.csv") as f:
            h = list(csv.DictReader(f))
        runs.append((m, h))
    return runs


def series(h, key):
    pts = [(int(r["step"]), float(r[key]), float(r["train_seconds"])) for r in h if r.get(key) not in (None, "")]
    return np.array(pts) if pts else np.zeros((0, 3))


def slope(s, frac):
    s = s[s[:, 0] >= (1 - frac) * s[-1, 0]]
    s = s[s[:, 0] > 0]
    if len(s) < 3:
        return float("nan")
    return float(np.polyfit(np.log10(s[:, 0]), np.log10(s[:, 1]), 1)[0])


def at_step(s, step):
    if len(s) == 0 or step > s[-1, 0]:
        return float("nan")
    return float(np.exp(np.interp(step, s[:, 0], np.log(s[:, 1]))))


def at_time(s, sec):
    if len(s) == 0 or sec > s[-1, 2]:
        return float("nan")
    return float(np.exp(np.interp(sec, s[:, 2], np.log(s[:, 1]))))


def freq_resolution(eps):
    """p95 frequency/phase error of the extractor for in-band ('envelope') error of size eps."""
    t = list(csv.DictReader(open(RO / "tables" / "stage01_frequency_extractor_uncertainty.csv")))
    rows = sorted((float(r["eps"]), float(r["frequency_error_p95"]), float(r["phase_error_p95"]))
                  for r in t if r["kind"] == "envelope")
    e = np.array(rows)
    le = np.log10(np.clip(eps, e[0, 0], e[-1, 0]))
    return (float(10 ** np.interp(le, np.log10(e[:, 0]), np.log10(e[:, 1]))),
            float(10 ** np.interp(le, np.log10(e[:, 0]), np.log10(e[:, 2]))))


def summarise(m, h):
    sp, se = series(h, "L2_paper"), series(h, "L2_exact")
    steps = m["optimizer_steps"]
    base = m["train_seconds"] - m["ntk_seconds"]
    s_half, s_q = slope(se, 0.5), slope(se, 0.25)
    proj_steps = (steps * (TARGET / m["L2_exact"]) ** (1 / s_half)
                  if s_half < 0 and m["L2_exact"] > TARGET else float("nan"))
    sec_per_step = m["train_seconds"] / steps
    best_run = np.minimum.accumulate(se[:, 1]) if len(se) else np.array([np.nan])
    spikes = float(np.max(se[len(se) // 4:, 1] / best_run[len(se) // 4:])) if len(se) > 4 else float("nan")
    fres, pres = freq_resolution(m["L2_exact"])
    lf = [float(r["L_f"]) for r in h if r.get("L_f")]
    return {
        "run_id": m["run_id"], "method": m["name"], "seed": m["seed"], "budget": m["budget_label"],
        "steps": steps, "L2_paper": m["L2_paper"], "L2_exact": m["L2_exact"],
        "L2_late_paper": m["L2_late_paper"], "L2_late_exact": m["L2_late_exact"],
        "best_L2_paper": float(sp[:, 1].min()), "best_L2_exact": float(se[:, 1].min()),
        "best_L2_exact_step": int(se[np.argmin(se[:, 1]), 0]),
        "slope_last_half": s_half, "slope_last_quarter": s_q,
        "L2_exact_at_5k": at_step(se, 5000), "L2_exact_at_10k": at_step(se, 10000), "L2_exact_at_20k": at_step(se, 20000),
        "L2_exact_at_15min": at_time(se, 900),
        "projected_steps_to_target": proj_steps,
        "projected_hours_to_target": proj_steps * sec_per_step / 3600 if proj_steps == proj_steps else float("nan"),
        "paper_budget_steps": 900000, "max_spike_ratio_after_25pct": spikes,
        "PDE_residual_rel": m["PDE_residual_rel"], "L_f_first": lf[0] if lf else float("nan"),
        "L_f_last": lf[-1] if lf else float("nan"),
        "BC_error_max": m["BC_error_max"], "IC_error_max": m["IC_error_max"],
        "IC_u_max": m["IC_u_max"], "IC_ut_max": m["IC_ut_max"],
        "frequency_error_exact": m["frequency_error_exact"], "frequency_error_paper": m["frequency_error_paper"],
        "phase_error_exact": m["phase_error_exact"], "amplitude_error_exact": m["amplitude_error_exact"],
        "damping_error_exact": m["damping_error_exact"],
        "freq_extractor_p95_at_this_L2": fres, "phase_extractor_p95_at_this_L2": pres,
        "train_seconds": m["train_seconds"], "base_seconds": base, "ntk_seconds": m["ntk_seconds"],
        "ntk_overhead_pct": 100 * m["ntk_seconds"] / base if base > 0 else float("nan"),
        "base_ms_per_step": 1e3 * base / steps, "ntk_s_per_update": m["ntk_seconds"] / max(m["ntk_updates"], 1),
        "ntk_updates": m["ntk_updates"], "ntk_row_gradients": m["ntk_row_gradients"],
        "pde_evaluations": m["pde_evaluations"], "training_points": m["training_points"],
        "grad_evaluations": m["grad_evaluations"], "peak_rss_train_mb": m["peak_rss_mb"],
        "peak_rss_eval_mb": m.get("peak_rss_eval_mb", float("nan")), "parameters": m["parameters"],
        "model_size_bytes": m["model_size_bytes"], "inference_single_point_us": m["inference_single_point_us"],
        "inference_grid_ms": m["inference_grid_ms"], "status": m["status"],
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def seed_stats(rows):
    out = []
    for k in ("L2_paper", "L2_exact", "best_L2_exact", "slope_last_half", "PDE_residual_rel",
              "frequency_error_exact", "base_ms_per_step", "ntk_overhead_pct"):
        v = np.array([r[k] for r in rows], dtype=float)
        n = len(v)
        sd = float(v.std(ddof=1)) if n > 1 else float("nan")
        ci = 4.303 * sd / math.sqrt(n) if n == 3 else float("nan")       # t_{0.975, 2}
        out.append({"metric": k, "n": n, "mean": float(v.mean()), "median": float(np.median(v)), "sd": sd,
                    "cv": sd / abs(v.mean()) if v.mean() else float("nan"), "min": float(v.min()),
                    "max": float(v.max()), "ci95_halfwidth": ci})
    return out


# ------------------------------------------------------------------- figures
def _style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_convergence(runs, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax, key, lab in [(axes[0], "L2_paper", "paper-faithful rel-L2"), (axes[1], "L2_exact", "exact-physics rel-L2")]:
        _style(ax); ax.set_facecolor("#fcfcfb")
        for i, (m, h) in enumerate(runs):
            s = series(h, key)
            ax.loglog(s[:, 0], s[:, 1], lw=2, color=COLORS[i % len(COLORS)])
            ax.annotate(f"{m['name']} (s{m['seed']})", (s[-1, 0], s[-1, 1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color=INK, va="center")
        ax.axhline(TARGET, color=INK, lw=1, ls="--")
        ax.text(ax.get_xlim()[0] * 1.2, TARGET * 1.15, "paper 4.64e-4", fontsize=7, color=MUTED)
        if key == "L2_paper":
            ax.axhline(FLOOR, color=MUTED, lw=1, ls=":")
            ax.text(ax.get_xlim()[0] * 1.2, FLOOR * 0.72, "floor of exact solution 4.39e-4", fontsize=7, color=MUTED)
        ax.set_xlabel("optimizer steps (mini-batch 32)", fontsize=8, color=INK)
        ax.set_ylabel(lab + " (101x1001 validation grid)", fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


def fig_ntk(m, h, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = [k for k in h[0] if k.startswith("lam_")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor="#fcfcfb")
    for i, k in enumerate(keys):
        s = series(h, k)
        axes[0].semilogy(s[:, 0], s[:, 1], lw=2, color=COLORS[i])
        axes[0].annotate(k[4:], (s[-1, 0], s[-1, 1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK)
        g = series(h, "gradnorm_" + k[4:])
        if len(g):
            axes[1].semilogy(g[:, 0], g[:, 1], lw=2, color=COLORS[i])
            axes[1].annotate(k[4:], (g[-1, 0], g[-1, 1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK)
    for ax, yl in [(axes[0], "NTK weight lambda_i (Eq. 37)"), (axes[1], "||grad_theta L_i|| (unweighted)")]:
        _style(ax); ax.set_facecolor("#fcfcfb"); ax.set_xlabel("optimizer steps", fontsize=8, color=INK)
        ax.set_ylabel(yl, fontsize=8, color=INK)
    fig.suptitle(f"{m['name']} seed {m['seed']} ({m['budget_label']}): loss-balancing diagnostics", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


def fig_diagnostics(m, path):
    """Error map, mid-span response (early/late), residual histogram, from final.pt."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import torch
    from beampinn.config import ExperimentConfig
    from beampinn.evaluation.metrics import grid, predict, _autograd_field
    from beampinn.losses.residuals import pde_residual
    from beampinn.models.networks import build_model
    from beampinn.training.trainer import resolve_problem
    blob = torch.load(RO / "checkpoints" / m["run_id"] / "final.pt", weights_only=False)
    cfg = ExperimentConfig.from_dict(blob["config"])
    torch.set_num_threads(4)
    bm, refs, c2, g, _ = resolve_problem(cfg)
    model = build_model(cfg, bm); model.load_state_dict(blob["model"])
    x, t, X, T = grid(bm.L, bm.t_end, 201, 2001)
    P = predict(model, X, T); Ue = refs["exact"].u(X, T)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="#fcfcfb")
    ax = axes[0, 0]
    im = ax.pcolormesh(t, x, np.abs(P - Ue) / bm.A0, shading="auto", cmap="Blues")
    fig.colorbar(im, ax=ax).set_label("|u_pred - u_exact| / A0", fontsize=8)
    ax.set_xlabel("t [s]", fontsize=8); ax.set_ylabel("x [m]", fontsize=8); ax.set_title("absolute error map", fontsize=9)
    i_mid = 100
    for ax, (a, b), ttl in [(axes[0, 1], (0, 0.15), "mid-span, t in [0, 0.15] s"), (axes[1, 0], (0.85, 1.0), "mid-span, t in [0.85, 1] s")]:
        sel = (t >= a) & (t <= b)
        ax.plot(t[sel], Ue[i_mid, sel] * 1e3, lw=2, color=COLORS[0], label="exact")
        ax.plot(t[sel], P[i_mid, sel] * 1e3, lw=1.2, ls="--", color=COLORS[1], label="PINN")
        _style(ax); ax.set_xlabel("t [s]", fontsize=8); ax.set_ylabel("u(L/2, t) [mm]", fontsize=8)
        ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7, frameon=False)
    _, _, Xr, Tr = grid(bm.L, bm.t_end, 51, 501)
    r = _autograd_field(model, Xr, Tr, lambda xx, tt: pde_residual(model, xx, tt, c2, g))
    rel = r / np.sqrt(np.mean(refs["exact"].u(Xr, Tr, 0, 2) ** 2))
    ax = axes[1, 1]; _style(ax)
    ax.hist(rel.ravel(), bins=80, color=COLORS[0], edgecolor="#fcfcfb", linewidth=0.5, log=True)
    ax.set_xlabel("PDE residual / RMS(u_tt exact)", fontsize=8); ax.set_ylabel("count (51x501 grid)", fontsize=8)
    ax.set_title("residual distribution", fontsize=9)
    fig.suptitle(f"{m['name']} seed {m['seed']} at {m['optimizer_steps']:,} steps: L2_paper {m['L2_paper']:.3e}, "
                 f"L2_exact {m['L2_exact']:.3e}", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--budgets", default="S1,S2")
    a = ap.parse_args()
    runs = load_runs(set(a.budgets.split(",")))
    if not runs:
        print("no runs"); return
    rows = [summarise(m, h) for m, h in runs]
    write_csv(RO / "tables" / "phaseA_summary.csv", rows)
    c0 = [r for r in rows if r["method"] == "C0_paper" and r["budget"] == "S1"]
    if len(c0) > 1:
        write_csv(RO / "tables" / "phaseA_C0_seed_stats.csv", seed_stats(c0))
    for r in rows:
        print(f"{r['method']:11s} s{r['seed']} {r['budget']}: L2_paper {r['L2_paper']:.3e} L2_exact {r['L2_exact']:.3e} "
              f"best_exact {r['best_L2_exact']:.3e} slope(1/2) {r['slope_last_half']:+.2f} (1/4) {r['slope_last_quarter']:+.2f} "
              f"PDE {r['PDE_residual_rel']:.2e} BC {r['BC_error_max']:.2e} IC {r['IC_error_max']:.2e} "
              f"f {r['frequency_error_exact']:.1e} base {r['base_ms_per_step']:.0f}ms/st NTK +{r['ntk_overhead_pct']:.0f}%")
    if a.no_figures:
        return
    fig_dir = RO / "figures"
    s1 = [(m, h) for m, h in runs if m["budget_label"] == "S1" and m["seed"] == 1234]
    if s1:
        fig_convergence(s1, fig_dir / "phaseA_A1_convergence.png", "Phase A1: 20k-step baselines on FE-D-M1 (seed 1234)")
    seeds = [(m, h) for m, h in runs if m["name"] == "C0_paper" and m["budget_label"] == "S1"]
    if len(seeds) > 1:
        fig_convergence(seeds, fig_dir / "phaseA_A2_C0_seeds.png", "Phase A2: C0 seed variability at 20k steps")
    for m, h in runs:
        if m["name"] == "C0_paper" and m["budget_label"] == "S2":
            fig_convergence([(m, h)], fig_dir / "phaseA_A3_C0_100k_convergence.png", "Phase A3: C0 to 100k steps")
            fig_ntk(m, h, fig_dir / "phaseA_A3_C0_100k_ntk.png")
            fig_diagnostics(m, fig_dir / "phaseA_A3_C0_100k_diagnostics.png")
    print("figures written")


if __name__ == "__main__":
    main()
