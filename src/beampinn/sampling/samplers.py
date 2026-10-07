"""Training-point samplers with paper mini-batch semantics and explicit accounting.

Paper semantics (Sec. 5.1.1, Eq. 48, Sec. 5.1.2): each epoch draws N = 640 points per loss
term ("batch size"), and the optimiser walks through them in mini-batches of 32, i.e.
N / mb = 20 optimizer steps per epoch. Points are redrawn every epoch ("sampled randomly
during training ... for each epoch", Sec. 5.1.2). 45 000 epochs => 9.0e5 steps.

A loss TERM is described by (kind, derivative order, point set):
    value : residual = u(x,t) - target          (IC displacement and/or BC displacement)
    dt    : residual = u_t(x,0)                 (IC velocity)
    dx    : residual = d^k u/dx^k at an end     (BC slope / moment / shear)
    pde   : residual = c2 u_xxxx + u_tt + gamma u_t
"""
from dataclasses import dataclass

import numpy as np
import torch

from ..physics.beam import BC_ORDERS

_DNAME = {1: "ux", 2: "uxx", 3: "uxxx"}


@dataclass
class TermSpec:
    name: str
    kind: str            # 'value' | 'dt' | 'dx' | 'pde'
    order: int = 0       # x-derivative order for 'dx'
    n_ic: int = 0        # 'value' only: rows that are IC points (rest are BC points)
    ends: tuple = ()     # BC ends ('left', 'right') carrying this order


def term_specs(bc_type, grouping, n, ic_fraction_in_u=0.5):
    orders = BC_ORDERS[bc_type]
    ends_of = {}
    for end, ords in orders.items():
        for k in ords:
            ends_of.setdefault(k, []).append(end)
    specs = []
    if grouping == "paper":               # Eq. 48 / A.6 / B.7
        n_ic = int(round(ic_fraction_in_u * n))
        specs.append(TermSpec("u", "value", 0, n_ic, tuple(ends_of.get(0, ()))))
        specs.append(TermSpec("ut", "dt"))
        for k in sorted(k for k in ends_of if k > 0):
            specs.append(TermSpec(_DNAME[k], "dx", k, 0, tuple(ends_of[k])))
        specs.append(TermSpec("f", "pde"))
    elif grouping == "data_only":         # DIAGNOSTIC: supervised fit of the exact solution
        specs.append(TermSpec("d", "data"))
    elif grouping == "pde_only":          # hard constraints: IC/BC satisfied exactly by construction
        specs.append(TermSpec("f", "pde"))
    elif grouping == "split":
        specs.append(TermSpec("ic_u", "value", 0, n, ()))
        specs.append(TermSpec("ic_ut", "dt"))
        if 0 in ends_of:
            specs.append(TermSpec("bc_u", "value", 0, 0, tuple(ends_of[0])))
        for k in sorted(k for k in ends_of if k > 0):
            specs.append(TermSpec("bc_" + _DNAME[k], "dx", k, 0, tuple(ends_of[k])))
        specs.append(TermSpec("pde", "pde"))
    else:
        raise ValueError(grouping)
    return specs


class PaperEpochSampler:
    """Draws `n` points per term each epoch and yields aligned mini-batches."""

    def __init__(self, benchmark, ic_fn, scfg, grouping, seed, dtype=torch.float32, data_fn=None):
        self.L, self.T = benchmark.L, benchmark.t_end
        self.ic_fn = ic_fn                       # float64 numpy u0(x)
        self.data_fn = data_fn                   # float64 numpy u(x, t) for the data diagnostic
        self.n, self.mb = scfg.n_per_term, scfg.mini_batch
        self.resample = scfg.resample
        self.specs = term_specs(benchmark.bc_type, grouping, self.n, scfg.ic_fraction_in_u)
        self.dtype = dtype
        self.gen = torch.Generator().manual_seed(seed + 7)
        self.epoch, self.pos = 0, 0
        self.data = None
        self.points_processed = {s.name: 0 for s in self.specs}
        self.pde_pool = None                     # set by RAD; replaces the uniform PDE redraw

    @property
    def steps_per_epoch(self):
        return self.n // self.mb

    # ------------------------------------------------------------- drawing
    def _u(self, k):
        return torch.rand(k, 1, generator=self.gen, dtype=torch.float64)

    def _bc_points(self, k, ends):
        per = [k // len(ends) + (1 if i < k % len(ends) else 0) for i in range(len(ends))]
        xs = torch.cat([torch.full((p, 1), 0.0 if e == "left" else self.L, dtype=torch.float64)
                        for p, e in zip(per, ends)])
        return xs, self._u(k) * self.T

    def _draw(self):
        data = {}
        for s in self.specs:
            if s.kind == "pde":
                if self.pde_pool is not None:        # adaptive pool (reshuffled below)
                    x, t = self.pde_pool
                else:
                    x, t = self._u(self.n) * self.L, self._u(self.n) * self.T
                tgt = torch.zeros(self.n, 1, dtype=torch.float64)
            elif s.kind == "data":
                x, t = self._u(self.n) * self.L, self._u(self.n) * self.T
                tgt = torch.from_numpy(self.data_fn(x.numpy(), t.numpy()))
            elif s.kind == "dt":
                x, t = self._u(self.n) * self.L, torch.zeros(self.n, 1, dtype=torch.float64)
                tgt = torch.zeros(self.n, 1, dtype=torch.float64)
            elif s.kind == "dx":
                x, t = self._bc_points(self.n, s.ends)
                tgt = torch.zeros(self.n, 1, dtype=torch.float64)
            else:   # value: n_ic IC rows + BC displacement rows
                n_bc = self.n - s.n_ic
                xi = self._u(s.n_ic) * self.L
                parts_x, parts_t = [xi], [torch.zeros(s.n_ic, 1, dtype=torch.float64)]
                parts_g = [torch.from_numpy(self.ic_fn(xi.numpy()))]
                if n_bc:
                    xb, tb = self._bc_points(n_bc, s.ends)
                    parts_x.append(xb); parts_t.append(tb)
                    parts_g.append(torch.zeros(n_bc, 1, dtype=torch.float64))
                x, t, tgt = torch.cat(parts_x), torch.cat(parts_t), torch.cat(parts_g)
            perm = torch.randperm(self.n, generator=self.gen)
            data[s.name] = (x[perm], t[perm], tgt[perm])
        self.data = data

    # ------------------------------------------------------------ iteration
    def next_batch(self):
        """'epoch': redraw at the start of every epoch; 'step': redraw every step;
        'never': draw once and reshuffle each epoch."""
        if self.data is None or self.resample == "step" or (self.resample == "epoch" and self.pos == 0):
            self._draw()
        sl = slice(self.pos * self.mb, (self.pos + 1) * self.mb)
        batch = {}
        for s in self.specs:
            x, t, g = self.data[s.name]
            batch[s.name] = {"x": x[sl].to(self.dtype), "t": t[sl].to(self.dtype),
                             "target": g[sl].to(self.dtype), "spec": s}
            self.points_processed[s.name] += self.mb
        self.pos += 1
        if self.pos == self.steps_per_epoch:
            self.pos, self.epoch = 0, self.epoch + 1
            if self.resample == "never":
                perm = {k: torch.randperm(self.n, generator=self.gen) for k in self.data}
                self.data = {k: tuple(a[perm[k]] for a in v) for k, v in self.data.items()}
        return batch

    def set_pde_pool(self, x, t):
        """Install an adaptive PDE point set (float64, shape (n,1)); used from the next epoch."""
        assert x.shape == (self.n, 1) and t.shape == (self.n, 1)
        self.pde_pool = (x.clone(), t.clone())

    # ---------------------------------------------------------------- state
    def state_dict(self):
        return {"gen": self.gen.get_state(), "epoch": self.epoch, "pos": self.pos,
                "data": self.data, "points_processed": dict(self.points_processed),
                "pde_pool": self.pde_pool}

    def load_state_dict(self, st):
        self.gen.set_state(st["gen"])
        self.epoch, self.pos, self.data = st["epoch"], st["pos"], st["data"]
        self.points_processed = dict(st["points_processed"])
        self.pde_pool = st.get("pde_pool")
