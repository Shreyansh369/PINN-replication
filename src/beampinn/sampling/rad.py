"""Residual-based Adaptive Distribution (RAD), Wu, Zhu, Tan, Kartha & Lu, CMAME 403 (2023) 115671.

Every `every` optimizer steps (at an epoch boundary):
    1. draw n_candidates points uniformly in the space-time domain;
    2. evaluate |r| (PDE residual, detached, chunked; no parameter gradients);
    3. p(z) proportional to |r(z)|^k / E[|r|^k] + c          (Wu et al. Eq. for RAD; c > 0 keeps exploration)
    4. draw the n_per_term PDE points from the candidates according to p, WITHOUT replacement
       (no duplicates); this becomes the PDE pool, reshuffled every epoch until the next update.
IC/BC terms are untouched (they keep their per-epoch uniform redraw, or are absent with hard
constraints). The cost of step 2 is reported separately as candidate evaluations / rad_seconds.
"""
import time

import numpy as np
import torch

from ..losses.residuals import pde_residual


def candidate_residuals(model, x, t, c2, gamma, dtype, chunk=1024):
    out = []
    for a in range(0, len(x), chunk):
        xx = x[a:a + chunk].to(dtype).requires_grad_(True)
        tt = t[a:a + chunk].to(dtype).requires_grad_(True)
        out.append(pde_residual(model, xx, tt, c2, gamma).detach().abs().double())
    return torch.cat(out).reshape(-1)


def rad_probabilities(r, k, c):
    w = r ** k
    p = w / w.mean() + c
    return p / p.sum()


def rad_update(model, sampler, c2, gamma, rcfg, dtype):
    t0 = time.perf_counter()
    g = sampler.gen
    n_c = rcfg.n_candidates
    x = torch.rand(n_c, 1, generator=g, dtype=torch.float64) * sampler.L
    t = torch.rand(n_c, 1, generator=g, dtype=torch.float64) * sampler.T
    r = candidate_residuals(model, x, t, c2, gamma, dtype)
    p = rad_probabilities(r, rcfg.k, rcfg.c)
    idx = torch.multinomial(p, sampler.n, replacement=False, generator=g)
    sampler.set_pde_pool(x[idx], t[idx])
    sel = r[idx]
    q = np.quantile(r.numpy(), [0.5, 0.9, 0.99])
    stats = {"rad_r_p50": float(q[0]), "rad_r_p90": float(q[1]), "rad_r_p99": float(q[2]),
             "rad_r_max": float(r.max()), "rad_r_mean_all": float(r.mean()), "rad_r_mean_selected": float(sel.mean()),
             "rad_t_mean_selected": float(t[idx].mean()), "rad_x_mean_selected": float(x[idx].mean()),
             "rad_seconds_update": time.perf_counter() - t0}
    return n_c, stats, (x[idx].numpy().ravel(), t[idx].numpy().ravel())
