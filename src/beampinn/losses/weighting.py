"""Loss weighting: fixed weights and the paper's NTK trace rule (Eq. 37).

    K_ii = J_i J_i^T,   tr(K_ii) = sum_k || d r_i(z_k) / d theta ||^2
    lambda_i = sum_j tr(K_jj) / tr(K_ii)

The trace is EXACT on the rows used (no subsampling estimator), computed with one
vectorised backward per term (`is_grads_batched`), with a per-row loop as fallback.
With trace_norm='mean' each trace is divided by its row count, which equals the paper's
rule whenever all terms have the same N (as in Eq. 48) and stays balanced otherwise.
"""
import torch


def per_row_sq_grad_norms(r, params, chunk=64, vectorised=True):
    """|| d r_k / d theta ||^2 for every row k of the residual vector r (graph retained)."""
    r = r.reshape(-1)
    n = r.numel()
    out = torch.empty(n, dtype=torch.float64)
    if vectorised:
        try:
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                eye = torch.zeros(b - a, n, dtype=r.dtype)
                eye[torch.arange(b - a), torch.arange(a, b)] = 1.0
                gs = torch.autograd.grad(r, params, grad_outputs=eye, retain_graph=True,
                                         is_grads_batched=True, allow_unused=True)
                acc = torch.zeros(b - a, dtype=torch.float64)
                for g in gs:
                    if g is not None:
                        acc += (g.reshape(b - a, -1).double() ** 2).sum(1)
                out[a:b] = acc
            return out
        except RuntimeError:
            pass                                   # fall back to the exact loop below
    for k in range(n):
        gs = torch.autograd.grad(r[k], params, retain_graph=True, allow_unused=True)
        out[k] = sum(float((g.double() ** 2).sum()) for g in gs if g is not None)
    return out


def ntk_traces(residuals, params, max_rows=None, trace_norm="mean", vectorised=None):
    """vectorised=None: batched backward on >1 thread, per-row loop on 1 thread (measured faster and
    ~0.5 GB lighter there). Both are exact and agree to 1e-10 (tests)."""
    if vectorised is None:
        vectorised = torch.get_num_threads() > 1
    traces, rows = {}, 0
    for name, r in residuals.items():
        rv = r.reshape(-1)
        if max_rows is not None and rv.numel() > max_rows:
            rv = rv[:max_rows]
        g2 = per_row_sq_grad_norms(rv, params, vectorised=vectorised)
        traces[name] = float(g2.sum() / (g2.numel() if trace_norm == "mean" else 1.0))
        rows += rv.numel()
    return traces, rows


def ntk_weights(traces, floor=1e-30):
    total = sum(traces.values())
    return {k: total / max(v, floor) for k, v in traces.items()}


class NTKWeighting:
    def __init__(self, names, ncfg):
        self.cfg = ncfg
        self.lam = {k: 1.0 for k in names}
        self.updates, self.rows, self.seconds = 0, 0, 0.0
        self.last_traces = {}

    def due(self, step):          # step is 0-based; update at step 0, k, 2k, ...
        return step % self.cfg.every == 0

    def update(self, residuals, params):
        import time
        t0 = time.perf_counter()
        traces, rows = ntk_traces(residuals, params, self.cfg.max_rows, self.cfg.trace_norm)
        new = ntk_weights(traces)
        b = self.cfg.ema
        self.lam = {k: (1 - b) * self.lam[k] + b * new[k] for k in self.lam}
        self.last_traces = traces
        self.updates += 1
        self.rows += rows
        self.seconds += time.perf_counter() - t0
        return rows

    def state_dict(self):
        return {"lam": self.lam, "updates": self.updates, "rows": self.rows,
                "seconds": self.seconds, "last_traces": self.last_traces}

    def load_state_dict(self, st):
        for k, v in st.items():
            setattr(self, k, v)


class FixedWeighting:
    def __init__(self, names, weights):
        self.lam = {k: float(weights.get(k, 1.0)) for k in names}
        self.updates, self.rows, self.seconds, self.last_traces = 0, 0, 0.0, {}

    def due(self, step):
        return False

    def update(self, *a):
        return 0

    def state_dict(self):
        return {"lam": self.lam}

    def load_state_dict(self, st):
        self.lam = st["lam"]
