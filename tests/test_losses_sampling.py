"""Sampler semantics, residual/loss definitions (via the analytic probe) and NTK weighting."""
import numpy as np
import pytest
import torch

from beampinn.config import ExperimentConfig, NTKCfg, SamplerCfg
from beampinn.losses.residuals import reduce_loss, term_residuals
from beampinn.losses.weighting import NTKWeighting, ntk_traces, ntk_weights, per_row_sq_grad_norms
from beampinn.models.networks import build_model
from beampinn.physics.benchmarks import get_benchmark
from beampinn.sampling.samplers import PaperEpochSampler, term_specs


def _sampler(bid="FE-D-M1", grouping="paper", n=640, mb=32, resample="epoch", seed=0, ic="exact"):
    bm = get_benchmark(bid)
    ref = bm.reference(ic)
    return bm, PaperEpochSampler(bm, ref.u0, SamplerCfg(n_per_term=n, mini_batch=mb, resample=resample),
                                 grouping, seed, torch.float64)


@pytest.mark.parametrize("bc,grouping,names", [
    ("fixed-fixed", "paper", ["u", "ut", "ux", "f"]),                        # Eq. 47-48
    ("simply-supported", "paper", ["u", "ut", "uxx", "f"]),                  # Eq. A.5-A.6
    ("cantilever", "paper", ["u", "ut", "ux", "uxx", "uxxx", "f"]),          # Eq. B.6-B.7
    ("fixed-fixed", "split", ["ic_u", "ic_ut", "bc_u", "bc_ux", "pde"]),
])
def test_term_specs_match_paper_losses(bc, grouping, names):
    assert [s.name for s in term_specs(bc, grouping, 640)] == names


def test_epoch_semantics_and_point_sets():
    bm, s = _sampler()
    assert s.steps_per_epoch == 20
    seen = {k: [] for k in ("u", "ut", "ux", "f")}
    for _ in range(20):
        b = s.next_batch()
        for k in seen:
            assert b[k]["x"].shape == (32, 1)
            seen[k].append(torch.cat([b[k]["x"], b[k]["t"], b[k]["target"]], 1))
    assert s.epoch == 1 and s.pos == 0
    for k in seen:                                  # every point of the epoch used exactly once
        assert torch.equal(torch.cat(seen[k]).sort(0).values, torch.cat(s.data[k], 1).sort(0).values)
    x, t, g = (a for a in s.data["u"])
    ic = t.ravel() == 0
    assert int(ic.sum()) == 320                    # ic_fraction_in_u = 0.5
    assert torch.allclose(g[ic], torch.from_numpy(bm.reference("exact").u0(x[ic].numpy())))
    assert set(x[~ic].ravel().tolist()) <= {0.0, 2.75} and float(g[~ic].abs().max()) == 0
    xs = s.data["ux"][0].ravel()
    assert int((xs == 0).sum()) == 320 and int((xs == 2.75).sum()) == 320
    xf, tf, _ = s.data["f"]
    assert 0 <= float(xf.min()) and float(xf.max()) <= 2.75 and 0 <= float(tf.min()) and float(tf.max()) <= 1
    assert all(v == 20 * 32 for v in s.points_processed.values())


def test_resample_each_epoch_and_reproducibility():
    _, a = _sampler(seed=3)
    _, b = _sampler(seed=3)
    for _ in range(25):
        ba, bb = a.next_batch(), b.next_batch()
    assert torch.equal(ba["f"]["x"], bb["f"]["x"])
    _, c = _sampler(seed=3)
    c.next_batch(); first = c.data["f"][0].clone()
    for _ in range(20):
        c.next_batch()
    assert not torch.equal(first, c.data["f"][0])   # epoch 2 drew new points


def test_sampler_state_resume():
    _, a = _sampler(seed=9)
    for _ in range(7):
        a.next_batch()
    st = a.state_dict()
    nxt = [a.next_batch()["f"]["x"].clone() for _ in range(30)]
    _, b = _sampler(seed=123)
    b.load_state_dict(st)
    assert all(torch.equal(n, b.next_batch()["f"]["x"]) for n in nxt)


@pytest.mark.parametrize("grouping", ["paper", "split"])
def test_exact_probe_has_zero_residuals(probe_factory, grouping):
    bm, s = _sampler(grouping=grouping)
    probe = probe_factory(bm.reference("exact"))
    c2, g = bm.pde_coeffs()
    res = term_residuals(probe, s.next_batch(), c2, g)
    scale = {"f": 1340.0, "pde": 1340.0}           # |u_tt| ~ omega^2 A0
    for k, r in res.items():
        err = float(r.detach().abs().max())
        assert err / scale.get(k, 1.0) < 1e-9, (k, err)


def test_paper_probe_violates_slope_bc(probe_factory):
    bm, s = _sampler()
    probe = probe_factory(bm.reference("paper"))
    c2, g = bm.pde_coeffs()
    res = term_residuals(probe, s.next_batch(), c2, g)
    assert float(res["ux"].detach().abs().max()) > 1e-6       # u_x(L) != 0 for the rounded root
    assert float(res["f"].detach().abs().max()) < 1e-6        # but it does satisfy the PDE


def test_reduce_loss_half_mean():
    r = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert abs(float(reduce_loss(r)) - 0.5 * 14 / 3) < 1e-12
    assert abs(float(reduce_loss(r, "mean")) - 14 / 3) < 1e-12


def _small_model():
    cfg = ExperimentConfig(); cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 12, 6
    return build_model(cfg, get_benchmark("FE-D-M1")).double()


def test_ntk_vectorised_trace_equals_loop():
    m = _small_model()
    bm, s = _sampler(n=64, mb=32)
    res = term_residuals(m, s.next_batch(), *bm.pde_coeffs())
    params = list(m.parameters())
    for k, r in res.items():
        a = per_row_sq_grad_norms(r, params, vectorised=True)
        b = per_row_sq_grad_norms(r, params, vectorised=False)
        assert torch.allclose(a, b, rtol=1e-10), k


def test_ntk_weight_rule_eq37():
    tr = {"u": 2.0, "ut": 4.0, "f": 10.0}
    w = ntk_weights(tr)
    assert w == {"u": 8.0, "ut": 4.0, "f": 1.6}
    m = _small_model()
    bm, s = _sampler(n=64, mb=32)
    res = term_residuals(m, s.next_batch(), *bm.pde_coeffs())
    tm, _ = ntk_traces(res, list(m.parameters()), trace_norm="mean")
    ts, _ = ntk_traces(res, list(m.parameters()), trace_norm="sum")
    wm, ws = ntk_weights(tm), ntk_weights(ts)      # equal N per term => identical weights
    assert all(abs(wm[k] - ws[k]) / ws[k] < 1e-12 for k in wm)


def test_ntk_schedule_and_ema():
    w = NTKWeighting(["a", "b"], NTKCfg(every=100, ema=1.0))
    assert [s for s in range(301) if w.due(s)] == [0, 100, 200, 300]
