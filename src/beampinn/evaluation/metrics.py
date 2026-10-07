"""Full evaluation against BOTH references (paper-faithful and exact-physics).

Evaluation grid (approved resolution A7): uniform 201 (x) x 2001 (t) including endpoints,
float64 reference. Primary metric: paper Eq. 44 relative L2 on that grid.
"""
import math
import time

import numpy as np
import torch

from ..losses.residuals import dxk, d, pde_residual
from ..physics.beam import BC_ORDERS
from .frequency import fit_damped_cosine, compare_fits

EVAL_NX, EVAL_NT = 201, 2001
VAL_NX, VAL_NT = 101, 1001         # cheaper grid for training-time validation (49 pts/period)
PDE_NX, PDE_NT = 51, 501


def grid(L, T, nx, nt):
    x = np.linspace(0.0, L, nx)
    t = np.linspace(0.0, T, nt)
    X, Tm = np.meshgrid(x, t, indexing="ij")
    return x, t, X, Tm


@torch.no_grad()
def predict(model, X, T, chunk=16384):          # 16k: lower peak RSS and faster on 1 thread
    dt = next(model.parameters()).dtype
    xf = torch.from_numpy(X.reshape(-1, 1)).to(dt)
    tf = torch.from_numpy(T.reshape(-1, 1)).to(dt)
    out = [model(xf[i:i + chunk], tf[i:i + chunk]) for i in range(0, len(xf), chunk)]
    return torch.cat(out).double().numpy().reshape(X.shape)


def rel_l2(P, U):
    return float(np.linalg.norm(P - U) / np.linalg.norm(U))


def l2_pair(P, refs, X, T, t_late=0.5):
    """rel-L2 (full window and late window t >= t_late*T) vs each reference."""
    out = {}
    late = T >= t_late * T.max() - 1e-12
    for k, ref in refs.items():
        U = ref.u(X, T)
        out[f"L2_{k}"] = rel_l2(P, U)
        out[f"L2_late_{k}"] = rel_l2(P[late], U[late])
        out[f"RMSE_{k}"] = float(np.sqrt(np.mean((P - U) ** 2)))
        out[f"maxerr_{k}"] = float(np.abs(P - U).max())
    return out


def validation_metrics(model, refs, L, T):
    _, _, X, Tm = grid(L, T, VAL_NX, VAL_NT)
    return l2_pair(predict(model, X, Tm), refs, X, Tm)


def _autograd_field(model, X, T, fn, chunk=1024):   # 4096 peaked at ~3 GB (4th-order graph)
    dt = next(model.parameters()).dtype
    xs = torch.from_numpy(X.reshape(-1, 1)).to(dt)
    ts = torch.from_numpy(T.reshape(-1, 1)).to(dt)
    vals = []
    for i in range(0, len(xs), chunk):
        x = xs[i:i + chunk].clone().requires_grad_(True)
        t = ts[i:i + chunk].clone().requires_grad_(True)
        vals.append(fn(x, t).detach().double())
    return torch.cat(vals).numpy().reshape(X.shape)


def physics_metrics(model, bm, ref, c2, gamma):
    """PDE / BC / IC errors, dimensionless (see STAGE01 report for definitions)."""
    L, T, A0 = bm.L, bm.t_end, ref.A0
    _, _, X, Tm = grid(L, T, PDE_NX, PDE_NT)
    r = _autograd_field(model, X, Tm, lambda x, t: pde_residual(model, x, t, c2, gamma))
    utt_ref = ref.u(X, Tm, 0, 2)
    out = {"PDE_residual_rel": float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(utt_ref ** 2))),
           "PDE_residual_rms": float(np.sqrt(np.mean(r ** 2)))}
    # BCs: every required condition at each end, on the evaluation time grid
    t = np.linspace(0.0, T, EVAL_NT)
    bc_max, bc_mean = 0.0, []
    names = {0: "u", 1: "ux", 2: "uxx", 3: "uxxx"}
    for end, orders in BC_ORDERS[bm.bc_type].items():
        xe = np.full_like(t, 0.0 if end == "left" else L)
        for k in orders:
            v = _autograd_field(model, xe[:, None], t[:, None], lambda x, tt, k=k: dxk(model(x, tt), x, k))
            nv = np.abs(v).ravel() * (L ** k) / A0        # dimensionless: d^k u/dx^k * L^k / A0
            out[f"BC_{names[k]}_{end}_max"] = float(nv.max())
            bc_max = max(bc_max, float(nv.max())); bc_mean.append(float(nv.mean()))
    out["BC_error_max"], out["BC_error_mean"] = bc_max, float(np.mean(bc_mean))
    # ICs
    x = np.linspace(0.0, L, EVAL_NX)
    z = np.zeros_like(x)
    u0p = predict(model, x[:, None], z[:, None]).ravel()
    ut0 = _autograd_field(model, x[:, None], z[:, None], lambda xx, tt: d(model(xx, tt), tt)).ravel()
    e_u = np.abs(u0p - ref.u0(x)) / A0
    e_v = np.abs(ut0) / (ref.omega * A0)
    out.update(IC_u_max=float(e_u.max()), IC_u_mean=float(e_u.mean()),
               IC_ut_max=float(e_v.max()), IC_ut_mean=float(e_v.mean()),
               IC_error_max=float(max(e_u.max(), e_v.max())))
    return out


def spectral_metrics(model, bm, refs):
    """Damped-cosine fit at x_norm (mid-span for FE mode 1) vs the exact parameters."""
    out = {}
    ref_e = refs["exact"]
    t = np.linspace(0.0, bm.t_end, EVAL_NT)
    xn = np.full_like(t, ref_e.x_norm)
    y = predict(model, xn[:, None], t[:, None]).ravel()
    fit = fit_damped_cosine(t, y)
    for k, ref in refs.items():
        a, lam, w, phi = ref.damped_cosine_params()
        e = compare_fits(fit, a * ref.A0, lam, w, phi)
        out.update({f"{n}_{k}": v for n, v in e.items()})
    out.update(fit_w=fit["w"], fit_lam=fit["lam"], fit_a=fit["a"], fit_phi=fit["phi"],
               fit_offset=fit["c"], fit_rms_resid=fit["rms_fit_resid"])
    return out


def inference_metrics(model, bm, reps=5):
    _, _, X, Tm = grid(bm.L, bm.t_end, EVAL_NX, EVAL_NT)
    predict(model, X, Tm)
    t0 = time.perf_counter()
    for _ in range(reps):
        predict(model, X, Tm)
    grid_ms = (time.perf_counter() - t0) / reps * 1e3
    dt = next(model.parameters()).dtype
    x1, t1 = torch.tensor([[1.0]], dtype=dt), torch.tensor([[0.5]], dtype=dt)
    with torch.no_grad():
        for _ in range(20):
            model(x1, t1)
        ts = []
        for _ in range(200):
            a = time.perf_counter(); model(x1, t1); ts.append(time.perf_counter() - a)
    return {"inference_grid_ms": grid_ms, "inference_grid_points": int(X.size),
            "inference_us_per_point": grid_ms * 1e3 / X.size,
            "inference_single_point_us": float(np.median(ts) * 1e6)}


def evaluate_full(model, bm, refs, c2, gamma):
    _, _, X, Tm = grid(bm.L, bm.t_end, EVAL_NX, EVAL_NT)
    P = predict(model, X, Tm)
    out = {"eval_grid": f"{EVAL_NX}x{EVAL_NT}", "finite": bool(np.isfinite(P).all())}
    out.update(l2_pair(P, refs, X, Tm))
    out.update(physics_metrics(model, bm, refs["exact"], c2, gamma))
    out.update(spectral_metrics(model, bm, refs))
    out.update(inference_metrics(model, bm))
    return out, P
