"""(1) Fourier-feature support vs the FE-D-M1 target frequencies for each convention, using the
ACTUAL seeded draws the models will use; (2) cost of the new loss pipeline and NTK update
(forward + backward only, no optimizer step, nothing trained)."""
import csv
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from beampinn.config import ExperimentConfig  # noqa: E402
from beampinn.losses.residuals import reduce_loss, term_residuals  # noqa: E402
from beampinn.losses.weighting import ntk_traces  # noqa: E402
from beampinn.models.networks import build_model, count_parameters  # noqa: E402
from beampinn.physics.benchmarks import get_benchmark  # noqa: E402
from beampinn.sampling.samplers import PaperEpochSampler  # noqa: E402
from beampinn.training.trainer import resolve_problem, setup_torch  # noqa: E402

bm = get_benchmark("FE-D-M1")
ref = bm.reference("exact")
CONV = {"paper (Eq.38-39: 2pi, physical)": dict(two_pi=True, input_norm="physical"),
        "reference code (no 2pi, standardised)": dict(two_pi=False, input_norm="standardize"),
        "legacy repo (no 2pi, unit)": dict(two_pi=False, input_norm="unit")}

rows = []
print(f"Targets: temporal wd = {ref.omega_d:.2f} rad/s; spatial beta1 = {ref.beta:.4f} rad/m")
for name, kw in CONV.items():
    cfg = ExperimentConfig(); [setattr(cfg.model, k, v) for k, v in kw.items()]
    m = build_model(cfg, bm)
    # angular frequency per PHYSICAL unit = feature frequency / transform scale
    for lab, encs, tr, target in [("t", m.enc_t, m.tt, ref.omega_d), ("x", m.enc_x, m.tx, ref.beta)]:
        for e in encs:
            wv = e.angular_frequencies() / float(tr.scale)
            r = dict(convention=name, coord=lab, sigma=e.sigma, m=wv.numel(),
                     target_rad_per_unit=target, max_feature=float(wv.max()),
                     median_feature=float(wv.median()), n_above_target=int((wv >= target).sum()),
                     n_within_10pct=int(((wv - target).abs() <= 0.1 * target).sum()))
            rows.append(r)
            print(f"  {name:38s} {lab} sigma={e.sigma:5.1f}: max {r['max_feature']:8.2f}  median {r['median_feature']:7.2f} "
                  f" >=target {r['n_above_target']:3d}/{r['m']}  within10% {r['n_within_10pct']:3d}")
out = ROOT / "results_optimization" / "tables" / "stage01_fourier_support_FE-D-M1.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote {out}\n")


def timeit(fn, reps):
    fn(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


prow = []
print("Cost per step of the new pipeline (fwd+bwd of all 4 terms, NO optimizer step):")
for depth, width in [(6, 200), (4, 200), (3, 64)]:
    cfg = ExperimentConfig(); cfg.model.depth, cfg.model.width = depth, width
    setup_torch(cfg)
    bm_, refs, c2, g, u0 = resolve_problem(cfg)
    model = build_model(cfg, bm_)
    params = list(model.parameters())
    for mb in (32, 128, 640):
        cfg.sampler.mini_batch = mb
        s = PaperEpochSampler(bm_, u0, cfg.sampler, "paper", 0)

        def step():
            res = term_residuals(model, s.next_batch(), c2, g)
            loss = sum(reduce_loss(r) for r in res.values())
            model.zero_grad(); loss.backward()
        ms = timeit(step, 5 if mb == 640 else 15)
        res = term_residuals(model, s.next_batch(), c2, g)
        t0 = time.perf_counter(); ntk_traces(res, params, vectorised=True); ntk_v = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter(); ntk_traces(res, params, vectorised=False); ntk_l = (time.perf_counter() - t0) * 1e3
        r = dict(net=f"{depth}x{width}", params=count_parameters(model), mini_batch=mb, step_ms=ms,
                 ntk_update_ms_vectorised=ntk_v, ntk_update_ms_loop=ntk_l, ntk_rows=4 * mb,
                 ntk_overhead_pct_every100=100 * ntk_v / 100 / ms)
        prow.append(r)
        print(f"  {r['net']:6s} P={r['params']:7,d} mb={mb:4d}: step {ms:7.1f} ms | NTK update vectorised {ntk_v:8.0f} ms, "
              f"loop {ntk_l:8.0f} ms ({4*mb} rows) | NTK overhead @every100 {r['ntk_overhead_pct_every100']:5.1f}%")
out = ROOT / "results_optimization" / "profiles" / "stage01_step_profile.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)
print(f"wrote {out}")
