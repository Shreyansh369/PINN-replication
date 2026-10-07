"""Phase E figures (evaluation only): mid-span u(t) traces and RAD point distributions.

    python experiments/phaseE_figures.py <run_id> [<run_id> ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beampinn.config import ExperimentConfig  # noqa: E402
from beampinn.evaluation.metrics import predict  # noqa: E402
from beampinn.models.constraints import HardConstrainedFF  # noqa: E402
from beampinn.models.networks import build_model  # noqa: E402
from beampinn.training.trainer import DTYPES, resolve_problem  # noqa: E402

RO = ROOT / "results_optimization"
EXACT, PRED, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"


def load(run_id):
    blob = torch.load(RO / "checkpoints" / run_id / "final.pt", weights_only=False)
    cfg = ExperimentConfig.from_dict(blob["config"])
    bm, refs, *_ = resolve_problem(cfg)
    model = build_model(cfg, bm).to(DTYPES[cfg.precision])
    if cfg.loss.hard_constraints in ("ff_tsq", "ff_tanh2"):
        from beampinn.training.trainer import build_hard
        model = build_hard(model, cfg, bm, refs).to(DTYPES[cfg.precision])
    model.load_state_dict(blob["model"])
    return cfg, bm, refs, model, blob


def style(ax):
    ax.set_facecolor(BG); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def velocity_trace(model, xn, t):
    from beampinn.evaluation.metrics import _autograd_field
    from beampinn.losses.residuals import d
    X = np.full((len(t), 1), xn); T = t[:, None]
    return _autograd_field(model, X, T, lambda xx, tt: d(model(xx, tt), tt)).ravel()


def main(run_ids, tag="phaseE", velocity=False):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    n = len(run_ids)
    ncol = 2 if velocity else 1
    fig, axes = plt.subplots(n, ncol, figsize=(11 if ncol == 1 else 15, 2.3 * n), facecolor=BG, squeeze=False)
    rad_runs = []
    for row_i, (ax, rid) in enumerate(zip(axes[:, 0], run_ids)):
        cfg, bm, refs, model, blob = load(rid)
        t = np.linspace(0, bm.t_end, 2001)
        xn = refs["exact"].x_norm
        yp = predict(model, np.full_like(t, xn)[:, None], t[:, None]).ravel()
        ye = refs["exact"].u(xn, t)
        style(ax)
        ax.plot(t, ye * 1e3, color=EXACT, lw=1.6, label="exact")
        ax.plot(t, yp * 1e3, color=PRED, lw=1.2, ls="--", label="PINN")
        ax.set_ylabel(f"u(x={xn:.2f} m) [mm]", fontsize=8)
        ax.set_title(f"{cfg.name}  ({blob['step']:,} steps)", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        if velocity:
            av = axes[row_i, 1]; style(av)
            av.plot(t, refs["exact"].u(xn, t, 0, 1), color=EXACT, lw=1.6, label="exact")
            av.plot(t, velocity_trace(model, xn, t), color=PRED, lw=1.2, ls="--", label="PINN")
            av.set_ylabel("u_t [m/s]", fontsize=8); av.set_title("mid-span velocity", fontsize=9, loc="left")
            av.legend(fontsize=7, frameon=False, loc="upper right")
        if blob.get("rad_snapshots"):
            rad_runs.append((cfg.name, bm, blob["rad_snapshots"]))
    for a in axes[-1, :]:
        a.set_xlabel("t [s]", fontsize=8)
    fig.tight_layout(); fig.savefig(RO / "figures" / f"{tag}_midspan_traces.png", dpi=150, facecolor=BG)
    plt.close(fig)
    if rad_runs:
        fig, axes = plt.subplots(len(rad_runs), 2, figsize=(11, 3.2 * len(rad_runs)), facecolor=BG, squeeze=False)
        for (name, bm, snaps), row in zip(rad_runs, axes):
            last = max(snaps)
            xs, ts = snaps[last]
            style(row[0]); style(row[1])
            row[0].scatter(ts, xs, s=4, color=EXACT, alpha=0.6, linewidths=0)
            row[0].set_xlabel("t [s]", fontsize=8); row[0].set_ylabel("x [m]", fontsize=8)
            row[0].set_title(f"{name}: RAD PDE points at step {last:,} (640 points)", fontsize=9, loc="left")
            row[1].hist(ts, bins=40, range=(0, bm.t_end), color=EXACT, edgecolor=BG, linewidth=0.5)
            row[1].axhline(len(ts) / 40, color=INK, lw=1, ls="--")
            row[1].text(bm.t_end * 0.98, len(ts) / 40 * 1.05, "uniform", fontsize=7, color=MUTED, ha="right")
            row[1].set_xlabel("t [s]", fontsize=8); row[1].set_ylabel("points per bin", fontsize=8)
            row[1].set_title("temporal density of RAD points", fontsize=9, loc="left")
        fig.tight_layout(); fig.savefig(RO / "figures" / f"{tag}_rad_distribution.png", dpi=150, facecolor=BG)
        plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main(sys.argv[1:])
