"""Does a trained model represent a GENUINE OSCILLATORY solution?  (evaluation only, no training)

    python experiments/oscillation_report.py <run_id> [<run_id> ...]  [--csv out.csv]

Per run: L2_paper / L2_exact, fitted frequency and ratio to the exact damped frequency, temporal
amplitude at x_norm (std of the predicted trace / std of the exact trace), max|u_t| over the field
(pred / exact), PDE residual, IC/BC errors, final NTK weights and gradient norms, wall time.
Verdict 'OSCILLATORY' requires |w_fit/w_d - 1| < 0.10 AND amplitude ratio in [0.5, 2] AND
max|u_t| ratio in [0.5, 2]; otherwise 'STATIC/LOW-FREQ' or 'WRONG-AMPLITUDE'.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beampinn.config import ExperimentConfig  # noqa: E402
from beampinn.evaluation.metrics import _autograd_field, grid, predict  # noqa: E402
from beampinn.losses.residuals import d  # noqa: E402
from beampinn.training.trainer import Trainer  # noqa: E402

RO = ROOT / "results_optimization"


def report(run_id):
    m = json.load(open(RO / "logs" / run_id / "metrics.json"))
    blob = torch.load(RO / "checkpoints" / run_id / "final.pt", weights_only=False)
    cfg = ExperimentConfig.from_dict(blob["config"])
    tr = Trainer.__new__(Trainer)                       # rebuild model exactly as trained, no run dirs
    from beampinn.training.trainer import resolve_problem, DTYPES
    from beampinn.models.networks import build_model
    from beampinn.models.constraints import HardConstrainedFF
    bm, refs, c2, g, _ = resolve_problem(cfg)
    model = build_model(cfg, bm).to(DTYPES[cfg.precision])
    if cfg.loss.hard_constraints in ("ff_tsq", "ff_tanh2"):
        from beampinn.training.trainer import build_hard
        model = build_hard(model, cfg, bm, refs).to(DTYPES[cfg.precision])
    model.load_state_dict(blob["model"])
    ref = refs["exact"]
    t = np.linspace(0, bm.t_end, 2001)
    xn = np.full_like(t, ref.x_norm)
    yp = predict(model, xn[:, None], t[:, None]).ravel()
    ye = ref.u(ref.x_norm, t)
    amp_ratio = float(yp.std() / ye.std())
    _, _, X, T = grid(bm.L, bm.t_end, 51, 501)
    ut = _autograd_field(model, X, T, lambda xx, tt: d(model(xx, tt), tt))
    ut_ratio = float(np.abs(ut).max() / np.abs(ref.u(X, T, 0, 1)).max())
    w_ratio = m["fit_w"] / ref.omega_d
    if abs(w_ratio - 1) < 0.10 and 0.5 <= amp_ratio <= 2 and 0.5 <= ut_ratio <= 2:
        verdict = "OSCILLATORY"
    elif abs(w_ratio - 1) < 0.10:
        verdict = "WRONG-AMPLITUDE"
    else:
        verdict = "STATIC/LOW-FREQ"
    h = list(csv.DictReader(open(RO / "logs" / run_id / "history.csv")))
    last = [r for r in h if r.get("loss")][-1]
    out = {"run_id": run_id, "name": m["name"], "steps": m["optimizer_steps"],
           "L2_paper": m["L2_paper"], "L2_exact": m["L2_exact"], "best_L2_exact": min(float(r["L2_exact"]) for r in h if r.get("L2_exact")),
           "fit_w": m["fit_w"], "w_exact": ref.omega_d, "freq_ratio": w_ratio, "amp_ratio": amp_ratio,
           "max_ut_ratio": ut_ratio, "PDE_residual_rel": m["PDE_residual_rel"], "IC_u_max": m["IC_u_max"],
           "IC_ut_max": m["IC_ut_max"], "BC_error_max": m["BC_error_max"],
           "train_seconds": m["train_seconds"], "ntk_seconds": m["ntk_seconds"], "rad_seconds": m.get("rad_seconds", 0.0),
           "verdict": verdict}
    out.update({k: float(v) for k, v in last.items() if k.startswith(("lam_", "gradnorm_", "L_")) and v})
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    if csv_out:
        args = [a for a in args if a != csv_out]
    torch.set_num_threads(4)
    rows = [report(r) for r in args]
    for r in rows:
        print(f"{r['name']:16s} L2p {r['L2_paper']:.3e} L2e {r['L2_exact']:.3e} (best {r['best_L2_exact']:.3e}) "
              f"w_fit/w {r['freq_ratio']:+.3f} amp {r['amp_ratio']:.2f} |u_t| {r['max_ut_ratio']:.2f} "
              f"PDE {r['PDE_residual_rel']:.2e} IC {r['IC_u_max']:.2e} BC {r['BC_error_max']:.2e} "
              f"t {r['train_seconds']:.0f}s -> {r['verdict']}")
        print("     " + " ".join(f"{k}={v:.2e}" for k, v in r.items() if k.startswith(("lam_", "gradnorm_"))))
    if csv_out:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
