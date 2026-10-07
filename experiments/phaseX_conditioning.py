"""Phase X conditioning diagnostics (evaluation only, no training).

For hard-constrained runs: statistics of the inner network output N(x,t) (overall and near t = 0),
the required N* for the exact solution, and the parameter-gradient norm of the PDE loss restricted
to an early-time window vs a late-time window. Also writes the loss-history figure.

    python experiments/phaseX_conditioning.py <hard_run_id> ... --supervised <run_id> ...
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from beampinn.losses.residuals import pde_residual  # noqa: E402
from phaseE_figures import load, style, BG, INK, MUTED  # noqa: E402

RO = ROOT / "results_optimization"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def grad_norm(model, x, t, c2, g):
    params = [p for p in model.parameters() if p.requires_grad]
    r = pde_residual(model, x.clone().requires_grad_(True), t.clone().requires_grad_(True), c2, g)
    loss = 0.5 * (r ** 2).mean()
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return float(torch.sqrt(sum((q.double() ** 2).sum() for q in gs if q is not None))), float(loss)


def conditioning(run_id):
    cfg, bm, refs, model, blob = load(run_id)
    from beampinn.training.trainer import resolve_problem
    _, _, c2, g, _ = resolve_problem(cfg)
    dt = next(model.parameters()).dtype
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(4000, 1, generator=gen, dtype=torch.float64) * bm.L
    t = torch.rand(4000, 1, generator=gen, dtype=torch.float64) * bm.t_end
    with torch.no_grad():
        N = model.net(x.to(dt), t.to(dt)).double().abs().numpy().ravel()
        t0 = torch.rand(4000, 1, generator=gen, dtype=torch.float64) * 0.01
        N0 = model.net(x.to(dt), t0.to(dt)).double().abs().numpy().ravel()
    # required N* for the exact solution (away from the ends, where Phi > 0.05)
    xs = x.numpy().ravel(); ts = t.numpy().ravel()
    keep = model.phi(torch.from_numpy(xs)).numpy() > 0.05
    gfac = model.g(torch.from_numpy(ts)).numpy()
    keep &= gfac > 1e-8
    Nstar = (refs["exact"].u(xs, ts) - refs["exact"].u0(xs)) / (gfac * model.phi(torch.from_numpy(xs)).numpy() * model.A0)
    Nstar = np.abs(Nstar[keep])
    xe = torch.rand(512, 1, generator=gen, dtype=torch.float64) * bm.L
    te = torch.rand(512, 1, generator=gen, dtype=torch.float64) * 0.02
    tl = 0.5 + torch.rand(512, 1, generator=gen, dtype=torch.float64) * 0.5
    ge, le = grad_norm(model, xe.to(dt), te.to(dt), c2, g)
    gl, ll = grad_norm(model, xe.to(dt), tl.to(dt), c2, g)
    return {"run_id": run_id, "name": cfg.name, "time_factor": cfg.loss.hard_constraints,
            "N_max": float(N.max()), "N_p50": float(np.median(N)), "N_p99": float(np.quantile(N, 0.99)),
            "N_near_t0_max": float(N0.max()), "N_near_t0_p50": float(np.median(N0)),
            "Nstar_required_max": float(Nstar.max()), "Nstar_required_p50": float(np.median(Nstar)),
            "gradnorm_pde_early_t<0.02": ge, "gradnorm_pde_late_t>0.5": gl,
            "pde_loss_early": le, "pde_loss_late": ll}


def loss_history_figure(run_ids, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=BG)
    for i, rid in enumerate(run_ids):
        h = list(csv.DictReader(open(RO / "logs" / rid / "history.csv")))
        name = json.load(open(RO / "logs" / rid / "metrics.json"))["name"]
        key = "L_d" if any(r.get("L_d") for r in h) else "L_f"
        s = np.array([(int(r["step"]), float(r[key])) for r in h if r.get(key)])
        v = np.array([(int(r["step"]), float(r["L2_exact"])) for r in h if r.get("L2_exact")])
        for ax, a in ((axes[0], s), (axes[1], v)):
            ax.semilogy(a[:, 0], a[:, 1], lw=2, color=COLORS[i % len(COLORS)])
            ax.annotate(f"{name} ({key})" if ax is axes[0] else name, (a[-1, 0], a[-1, 1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color=INK, va="center")
    for ax, yl in ((axes[0], "training loss (L_f: PDE, L_d: supervised data)"), (axes[1], "L2_exact (101x1001 validation grid)")):
        style(ax); ax.set_xlabel("optimizer steps", fontsize=8); ax.set_ylabel(yl, fontsize=8)
    axes[1].axhline(4.64e-4, color=INK, lw=1, ls="--")
    axes[1].text(100, 5.5e-4, "paper 4.64e-4", fontsize=7, color=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(4)
    args = sys.argv[1:]
    hard = args[:args.index("--supervised")] if "--supervised" in args else args
    sup = args[args.index("--supervised") + 1:] if "--supervised" in args else []
    rows = [conditioning(r) for r in hard]
    with open(RO / "tables" / "phaseX_conditioning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['name']:22s} {r['time_factor']:9s} |N| max {r['N_max']:9.3g} p50 {r['N_p50']:8.3g} p99 {r['N_p99']:8.3g} | "
              f"near t=0: max {r['N_near_t0_max']:8.3g} p50 {r['N_near_t0_p50']:8.3g} | required |N*| max {r['Nstar_required_max']:9.3g} "
              f"p50 {r['Nstar_required_p50']:7.3g} | grad PDE early {r['gradnorm_pde_early_t<0.02']:.3g} late {r['gradnorm_pde_late_t>0.5']:.3g}")
    loss_history_figure(hard + sup, RO / "figures" / "phaseX_loss_history.png")
    print("written")
