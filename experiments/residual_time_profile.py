"""PDE-residual time profile of trained models (evaluation only): RMS|r| per time bin, normalised by
RMS(u_tt exact) over the whole field; plus the share of total squared residual in each bin."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from phaseE_figures import load
from beampinn.evaluation.metrics import grid, _autograd_field
from beampinn.losses.residuals import pde_residual
from beampinn.training.trainer import resolve_problem
torch.set_num_threads(4)
edges = [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
print("bins [s]:", edges)
for rid in sys.argv[1:]:
    cfg, bm, refs, model, _ = load(rid); _, _, c2, g, _ = resolve_problem(cfg)
    _, t, X, T = grid(bm.L, bm.t_end, 41, 1001)
    r = _autograd_field(model, X, T, lambda x, tt: pde_residual(model, x, tt, c2, g))
    scale = np.sqrt(np.mean(refs["exact"].u(X, T, 0, 2) ** 2))
    tot = (r ** 2).sum()
    cells = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (T >= a) & (T < b if b < 1 else T <= b)
        cells.append(f"{np.sqrt(np.mean(r[m]**2))/scale:6.3f} ({100*(r[m]**2).sum()/tot:4.1f}%)")
    print(f"{cfg.name:16s} " + " | ".join(cells))
