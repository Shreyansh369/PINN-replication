"""Residual vectors for every loss term, and the loss reduction of paper Eq. 48.

All derivatives are reverse-mode autograd w.r.t. PHYSICAL x and t. NTK weighting needs
the residual VECTORS, so they are returned unreduced.
"""
import torch


def d(y, v):
    return torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]


def dxk(u, x, k):
    for _ in range(k):
        u = d(u, x)
    return u


def pde_residual(model, x, t, c2, gamma, scale=1.0):
    """r = c2 u_xxxx + u_tt + gamma u_t   (paper Eq. 49 form)."""
    u = model(x, t)
    u_xxxx = dxk(u, x, 4)
    u_t = d(u, t)
    u_tt = d(u_t, t)
    return scale * (c2 * u_xxxx + u_tt + gamma * u_t)


def term_residuals(model, batch, c2, gamma, pde_scale=1.0):
    out = {}
    for name, b in batch.items():
        s = b["spec"]
        x = b["x"].detach().clone().requires_grad_(s.kind in ("dx", "pde"))
        t = b["t"].detach().clone().requires_grad_(s.kind in ("dt", "pde"))
        if s.kind in ("value", "data"):
            out[name] = model(x, t) - b["target"]
        elif s.kind == "dt":
            out[name] = d(model(x, t), t)
        elif s.kind == "dx":
            out[name] = dxk(model(x, t), x, s.order)
        elif s.kind == "pde":
            out[name] = pde_residual(model, x, t, c2, gamma, pde_scale)
        else:
            raise ValueError(s.kind)
    return out


def reduce_loss(r, reduction="half_mean"):
    if reduction == "half_mean":      # Eq. 48: (1 / 2N) sum r^2
        return 0.5 * (r ** 2).mean()
    if reduction == "mean":
        return (r ** 2).mean()
    raise ValueError(reduction)
