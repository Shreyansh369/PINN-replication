"""Timing profile of the existing loss (fwd+bwd, NO optimizer step) and of an exact
separable forward-mode derivative alternative. Nothing is trained; no results/ writes."""
import sys, time, math, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebook_src"))
import sec_a, sec_b
g = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()):
    for c in [c for c in sec_a.cells() + sec_b.cells() if c.cell_type == "code"]:
        exec(c.source, g)
torch = g["torch"]; MS = g["MultiScaleFourierMLP"]; term_residuals = g["term_residuals"]
sample_batch = g["sample_batch"]; ntk_trace_estimates = g["ntk_trace_estimates"]
nd = g["NonDim"].from_beam(g["beam"], T_ref=1.0)          # paper 1 s window, SS
from torch.func import jacfwd, vmap, functional_call
torch.set_num_threads(4)

def timeit(fn, reps):
    fn(); t0 = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t0) / reps * 1e3

print(f"threads={torch.get_num_threads()}  torch={torch.__version__}")
print("A. existing per-point nested-autograd loss, fwd+bwd [ms/step]; N_ic=N_bc=N_pde=N")
gen = torch.Generator().manual_seed(0)
for (D, W, m) in [(3, 64, 64), (4, 200, 64), (4, 200, 100), (6, 200, 100)]:
    model = MS(D, W, m, 1.0, (1.0, 10.0), seed=0)
    P = sum(p.numel() for p in model.parameters())
    row = []
    for N in (32, 128, 640):
        def step():
            b = sample_batch(N, N, N, gen)
            r = term_residuals(model, b, nd, 1, nd.omega_star(1)**2)
            loss = sum((v**2).mean() for v in r.values())
            model.zero_grad(); loss.backward()
        row.append(timeit(step, 5 if N == 640 else 10))
    b = sample_batch(128, 128, 128, gen)
    r = term_residuals(model, b, nd, 1, nd.omega_star(1)**2)
    t0 = time.perf_counter(); ntk_trace_estimates(model, r, 16, gen); ntk_ms = (time.perf_counter()-t0)*1e3
    print(f"   {D}x{W} m={m:3d} P={P:7,d} | N=32: {row[0]:7.1f}  N=128: {row[1]:7.1f}  N=640: {row[2]:7.1f} | one NTK update (16 rows x5 terms): {ntk_ms:7.0f} ms")

print("\nB. separable exact derivatives: u = head([h_x(x)*h_t1(t), h_x(x)*h_t2(t)])")
print("   => u_xxxx needs only d^4 h_x/dx^4 (1-D input, forward mode); u_tt only d^2 h_t/dt^2")
def sep_pde(model, x, t, alpha):
    params = dict(model.named_parameters()); bufs = dict(model.named_buffers())
    def hx(xs):   # scalar -> (W,)
        return model.trunk(model._encode(xs.reshape(1, 1), model.Bx)).reshape(-1)
    def ht(ts, i):
        return model.trunk(model._encode(ts.reshape(1, 1), getattr(model, f"Bt{i}"))).reshape(-1)
    Hx  = vmap(hx)(x.reshape(-1)); H4 = vmap(jacfwd(jacfwd(jacfwd(jacfwd(hx)))))(x.reshape(-1))
    outs_u4, outs_tt = [], []
    for i in range(model.n_scales):
        f = lambda s, i=i: ht(s, i)
        Ht = vmap(f)(t.reshape(-1)); Htt = vmap(jacfwd(jacfwd(f)))(t.reshape(-1))
        outs_u4.append(H4 * Ht); outs_tt.append(Hx * Htt)
    Wh = model.head.weight            # bias drops out of derivatives
    u4 = torch.cat(outs_u4, 1) @ Wh.T; utt = torch.cat(outs_tt, 1) @ Wh.T
    return alpha * u4 + utt

for (D, W, m) in [(4, 200, 64), (6, 200, 100)]:
    model = MS(D, W, m, 1.0, (1.0, 10.0), seed=0)
    for N in (32, 640):
        x = torch.rand(N, 1, requires_grad=True); t = torch.rand(N, 1, requires_grad=True)
        r_ref = g["pde_residual"](model, x, t, nd.alpha, 0.0, 1.0)
        r_sep = sep_pde(model, x.detach(), t.detach(), nd.alpha)
        err = float((r_ref - r_sep).abs().max() / r_ref.abs().max())
        def a():
            xx = torch.rand(N, 1, requires_grad=True); tt = torch.rand(N, 1, requires_grad=True)
            l = (g["pde_residual"](model, xx, tt, nd.alpha, 0.0, 1.0)**2).mean(); model.zero_grad(); l.backward()
        def s():
            l = (sep_pde(model, torch.rand(N, 1), torch.rand(N, 1), nd.alpha)**2).mean(); model.zero_grad(); l.backward()
        ta, ts = timeit(a, 5), timeit(s, 5)
        print(f"   {D}x{W} m={m} N={N:4d}: nested-autograd PDE fwd+bwd {ta:7.1f} ms | separable fwd-mode {ts:7.1f} ms | speedup {ta/ts:4.2f}x | max rel diff {err:.1e}")
