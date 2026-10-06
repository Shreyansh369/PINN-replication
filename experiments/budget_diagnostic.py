"""Y1-20K budget diagnostic: checkpoint-resolved metrics, persistence/collapse time, figures.
Evaluation only, no training.

Persistence measure (applied identically to every model):
  local amplitude A(t) = half peak-to-peak of the mid-span displacement within a sliding window of
  one damped period P = 2 pi / omega_d centred at t; R(t) = A_pred(t) / A_exact(t).
  collapse time  = first t >= P/2 with R(t) < 0.5   (= t_end if never)
  persistence    = collapse time * f_d   [cycles]   (full window = 20.6 cycles)
The same is computed for the velocity (R_v).

    python experiments/budget_diagnostic.py <y1_20k_run_id> --compare <run_id> ...
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from beampinn.config import ExperimentConfig  # noqa: E402
from beampinn.evaluation.metrics import evaluate_full, predict  # noqa: E402
from beampinn.models.networks import build_model  # noqa: E402
from beampinn.training.trainer import DTYPES, build_hard, resolve_problem  # noqa: E402
from phaseE_figures import velocity_trace, style, BG, INK, MUTED, EXACT  # noqa: E402

RO = ROOT / "results_optimization"
STEP_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]   # validated categorical slots 2,3,4,7


def load_ckpt(run_id, fname):
    blob = torch.load(RO / "checkpoints" / run_id / fname, weights_only=False)
    cfg = ExperimentConfig.from_dict(blob["config"])
    bm, refs, c2, g, _ = resolve_problem(cfg)
    model = build_model(cfg, bm).to(DTYPES[cfg.precision])
    if cfg.loss.hard_constraints in ("ff_tsq", "ff_tanh2"):
        model = build_hard(model, cfg, bm, refs).to(DTYPES[cfg.precision])
    model.load_state_dict(blob["model"])
    return cfg, bm, refs, c2, g, model, blob


def local_amp(y, t, P):
    dt = t[1] - t[0]; h = max(1, int(round(P / 2 / dt)))
    out = np.empty_like(y)
    for i in range(len(y)):
        w = y[max(0, i - h):i + h + 1]
        out[i] = 0.5 * (w.max() - w.min())
    return out


def persistence(model, ref, T):
    t = np.linspace(0, T, 4001)
    xn = ref.x_norm
    up = predict(model, np.full_like(t, xn)[:, None], t[:, None]).ravel()
    vp = velocity_trace(model, xn, t)
    ue, ve = ref.u(xn, t), ref.u(xn, t, 0, 1)
    P = 2 * np.pi / ref.omega_d
    R = local_amp(up, t, P) / local_amp(ue, t, P)
    Rv = local_amp(vp, t, P) / local_amp(ve, t, P)
    ok = t >= P / 2

    def first_below(r):
        idx = np.where(ok & (r < 0.5))[0]
        return float(t[idx[0]]) if len(idx) else float(T)
    tc, tcv = first_below(R), first_below(Rv)
    fd = ref.omega_d / (2 * np.pi)
    return {"collapse_time_s": tc, "persistence_cycles": tc * fd, "collapse_time_vel_s": tcv,
            "persistence_cycles_vel": tcv * fd, "full_window_cycles": T * fd}, (t, up, vp, ue, ve, R, Rv)


def at_step(h, step, key):
    rows = [r for r in h if r.get(key) and int(r["step"]) <= step]
    return float(rows[-1][key]) if rows else float("nan")


def checkpoint_row(run_id, fname, h):
    cfg, bm, refs, c2, g, model, blob = load_ckpt(run_id, fname)
    met, _ = evaluate_full(model, bm, refs, c2, g)
    pers, traces = persistence(model, refs["exact"], bm.t_end)
    step = int(blob["step"])
    row = {"run": cfg.name, "step": step, "mini_batch": cfg.sampler.mini_batch,
           "L2_exact": met["L2_exact"], "L2_paper": met["L2_paper"], "L2_late_exact": met["L2_late_exact"],
           "PDE_residual_rel": met["PDE_residual_rel"], "IC_error_max": met["IC_error_max"],
           "BC_error_max": met["BC_error_max"], "fit_w": met["fit_w"],
           "frequency_error_exact": met["frequency_error_exact"], "fit_decay": met["fit_lam"],
           "amplitude_ratio": None, "max_ut_ratio": None,
           "train_seconds_cum": at_step(h, step, "train_seconds"),
           "pde_evaluations_cum": step * cfg.sampler.mini_batch, **pers}
    t, up, vp, ue, ve, R, Rv = traces
    row["amplitude_ratio"] = float(up.std() / ue.std())
    row["max_ut_ratio"] = float(np.abs(vp).max() / np.abs(ve).max())
    return row, traces


def main(run_id, compare):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    h = list(csv.DictReader(open(RO / "logs" / run_id / "history.csv")))
    rows, traces = [], {}
    for s in (5000, 10000, 15000, 20000):
        r, tr = checkpoint_row(run_id, f"step_{s}.pt", h); rows.append(r); traces[s] = tr
        print(f"{s:6d}: L2e {r['L2_exact']:.3f} L2p {r['L2_paper']:.3f} PDE {r['PDE_residual_rel']:.3f} "
              f"w {r['fit_w']:.1f} decay {r['fit_decay']:.2f} amp {r['amplitude_ratio']:.2f} ut {r['max_ut_ratio']:.2f} "
              f"collapse {r['collapse_time_s']:.3f}s ({r['persistence_cycles']:.1f} cyc; vel {r['persistence_cycles_vel']:.1f}) "
              f"t {r['train_seconds_cum']:.0f}s pde {r['pde_evaluations_cum']:.0f}")
    cmp_rows = []
    for rid in compare:
        hh = list(csv.DictReader(open(RO / "logs" / rid / "history.csv")))
        r, _ = checkpoint_row(rid, "final.pt", hh)
        m = json.load(open(RO / "logs" / rid / "metrics.json"))
        r["train_seconds_cum"] = m["train_seconds"]; r["pde_evaluations_cum"] = m["pde_evaluations"]
        cmp_rows.append(r)
        print(f"compare {r['run']:16s} step {r['step']} mb {r['mini_batch']}: L2e {r['L2_exact']:.3f} PDE {r['PDE_residual_rel']:.3f} "
              f"collapse {r['collapse_time_s']:.3f}s ({r['persistence_cycles']:.1f} cyc) t {r['train_seconds_cum']:.0f}s pde {r['pde_evaluations_cum']:.0f}")
    with open(RO / "tables" / "phaseY1_20K_checkpoints.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows + cmp_rows)

    # Figures 1-2: displacement / velocity small multiples (exact behind each checkpoint)
    for kind, idx_p, idx_e, unit, scale, fname in (("displacement", 1, 3, "u [mm]", 1e3, "disp"),
                                                   ("velocity", 2, 4, "u_t [m/s]", 1.0, "vel")):
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), facecolor=BG, sharex=True)
        for ax, (s, col) in zip(axes, zip((5000, 10000, 15000, 20000), STEP_COLORS)):
            tr = traces[s]; style(ax)
            ax.plot(tr[0], tr[idx_e] * scale, color=EXACT, lw=1.5, label="exact")
            ax.plot(tr[0], tr[idx_p] * scale, color=col, lw=1.2, label=f"PINN @ {s:,} steps")
            tc = [r for r in rows if r["step"] == s][0]["collapse_time_s" if kind == "displacement" else "collapse_time_vel_s"]
            ax.axvline(tc, color=INK, lw=1, ls="--")
            ax.text(tc + 0.005, ax.get_ylim()[1] * 0.7, f"collapse {tc:.3f} s", fontsize=7, color=INK)
            ax.set_ylabel(unit, fontsize=8); ax.legend(fontsize=7, frameon=False, loc="upper right")
        axes[-1].set_xlabel("t [s]", fontsize=8)
        fig.suptitle(f"Y1-20K: mid-span {kind} at checkpoints (dashed line: amplitude ratio < 0.5)", fontsize=10, color=INK)
        fig.tight_layout(); fig.savefig(RO / "figures" / f"phaseY1_20K_{fname}_traces.png", dpi=150, facecolor=BG); plt.close(fig)

    # Figure 3: collapse time vs steps (+ comparison points)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG); style(ax)
    st = [r["step"] for r in rows]; ct = [r["collapse_time_s"] for r in rows]
    ax.plot(st, ct, "-o", color=STEP_COLORS[0], lw=2, ms=6)
    ax.annotate("Y1-20K (mb 32)", (st[-1], ct[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK)
    for r, mk in zip(cmp_rows, ("s", "D", "^", "v")):
        ax.plot([r["step"]], [r["collapse_time_s"]], mk, color=MUTED, ms=7)
        ax.annotate(f"{r['run']} (mb {r['mini_batch']})", (r["step"], r["collapse_time_s"]), xytext=(6, -2),
                    textcoords="offset points", fontsize=7, color=MUTED)
    ax.axhline(1.0, color=INK, lw=1, ls=":"); ax.text(500, 0.96, "full window (20.6 cycles)", fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("optimizer steps", fontsize=8); ax.set_ylabel("collapse time [s] (amplitude ratio < 0.5)", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(RO / "figures" / "phaseY1_20K_collapse_vs_steps.png", dpi=150, facecolor=BG); plt.close(fig)

    # Figures 4-5: loss history and validation L2 / PDE residual vs steps
    s_loss = np.array([(int(r["step"]), float(r["L_f"])) for r in h if r.get("L_f")])
    s_l2 = np.array([(int(r["step"]), float(r["L2_exact"])) for r in h if r.get("L2_exact")])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=BG)
    style(axes[0]); axes[0].semilogy(s_loss[:, 0], s_loss[:, 1], color=STEP_COLORS[0], lw=1.5)
    axes[0].set_ylabel("training loss = PDE loss L_f (single term, hard constraints)", fontsize=8)
    style(axes[1]); axes[1].semilogy(s_l2[:, 0], s_l2[:, 1], color=STEP_COLORS[0], lw=1.5, label="L2_exact (validation)")
    axes[1].semilogy(st, [r["PDE_residual_rel"] for r in rows], "o--", color=STEP_COLORS[3], lw=1.2, label="relative PDE residual (checkpoints)")
    axes[1].legend(fontsize=7, frameon=False)
    for a in axes:
        a.set_xlabel("optimizer steps", fontsize=8)
    fig.tight_layout(); fig.savefig(RO / "figures" / "phaseY1_20K_loss_and_residual.png", dpi=150, facecolor=BG); plt.close(fig)
    print("written")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmp_ = a[a.index("--compare") + 1:] if "--compare" in a else []
    main(a[0], cmp_)
