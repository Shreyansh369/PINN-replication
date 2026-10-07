"""Time-resolved diagnostics of trained models (evaluation only, no training).

Per time window [a, b):
  res_rms   RMS PDE residual / RMS(u_tt exact over the whole field)
  res_share share of the total squared residual
  amp_ratio RMS of the predicted mid-span displacement / RMS of the exact one     (persistence)
  vel_ratio RMS of the predicted mid-span velocity / RMS of the exact one
  grad      || d L_pde(window) / d theta ||  (PDE loss restricted to 512 random points in the window)

    python experiments/temporal_diagnostics.py <run_id> ... [--csv out.csv]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from phaseE_figures import load, velocity_trace  # noqa: E402
from beampinn.evaluation.metrics import grid, predict, _autograd_field  # noqa: E402
from beampinn.losses.residuals import pde_residual  # noqa: E402
from beampinn.training.trainer import resolve_problem  # noqa: E402

EDGES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]


def window_grad(model, bm, a, b, c2, g, dtype):
    gen = torch.Generator().manual_seed(11)
    x = (torch.rand(512, 1, generator=gen, dtype=torch.float64) * bm.L).to(dtype).requires_grad_(True)
    t = (a + torch.rand(512, 1, generator=gen, dtype=torch.float64) * (b - a)).to(dtype).requires_grad_(True)
    r = pde_residual(model, x, t, c2, g)
    params = [p for p in model.parameters() if p.requires_grad]
    gs = torch.autograd.grad(0.5 * (r ** 2).mean(), params, allow_unused=True)
    return float(torch.sqrt(sum((q.double() ** 2).sum() for q in gs if q is not None)))


def diagnose(rid):
    cfg, bm, refs, model, _ = load(rid)
    _, _, c2, g, _ = resolve_problem(cfg)
    dtype = next(model.parameters()).dtype
    ref = refs["exact"]
    _, t, X, T = grid(bm.L, bm.t_end, 41, 1001)
    r = _autograd_field(model, X, T, lambda x, tt: pde_residual(model, x, tt, c2, g))
    scale = np.sqrt(np.mean(ref.u(X, T, 0, 2) ** 2))
    tot = (r ** 2).sum()
    tt = np.linspace(0, bm.t_end, 4001)
    xn = ref.x_norm
    up = predict(model, np.full_like(tt, xn)[:, None], tt[:, None]).ravel()
    vp = velocity_trace(model, xn, tt)
    ue, ve = ref.u(xn, tt), ref.u(xn, tt, 0, 1)
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (T >= a) & ((T < b) if b < bm.t_end else (T <= b))
        w = (tt >= a) & ((tt < b) if b < bm.t_end else (tt <= b))
        rows.append({"run": cfg.name, "window": f"[{a},{b})",
                     "res_rms": float(np.sqrt(np.mean(r[m] ** 2)) / scale),
                     "res_share": float((r[m] ** 2).sum() / tot),
                     "amp_ratio": float(np.sqrt(np.mean(up[w] ** 2)) / np.sqrt(np.mean(ue[w] ** 2))),
                     "vel_ratio": float(np.sqrt(np.mean(vp[w] ** 2)) / np.sqrt(np.mean(ve[w] ** 2))),
                     "grad": window_grad(model, bm, a, b, c2, g, dtype)})
    return rows


if __name__ == "__main__":
    torch.set_num_threads(4)
    args = sys.argv[1:]
    out = None
    if "--csv" in args:
        out = args[args.index("--csv") + 1]; args = args[:args.index("--csv")]
    allrows = []
    print("windows:", [f"[{a},{b})" for a, b in zip(EDGES[:-1], EDGES[1:])])
    for rid in args:
        rows = diagnose(rid); allrows += rows
        print(f"{rows[0]['run']:16s} amp  " + " ".join(f"{r['amp_ratio']:5.2f}" for r in rows))
        print(f"{'':16s} vel  " + " ".join(f"{r['vel_ratio']:5.2f}" for r in rows))
        print(f"{'':16s} res% " + " ".join(f"{100*r['res_share']:5.1f}" for r in rows))
        print(f"{'':16s} grad " + " ".join(f"{r['grad']:5.0e}" for r in rows))
    if out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
