"""Hard constraints (ff_tsq), RAD sampling and the mode-2 benchmark."""
import math

import numpy as np
import pytest
import torch

from beampinn.config import ExperimentConfig, RADCfg, SamplerCfg
from beampinn.losses.residuals import d, dxk
from beampinn.models.constraints import HardConstrainedFF
from beampinn.models.networks import build_model
from beampinn.physics.benchmarks import get_benchmark
from beampinn.sampling.rad import rad_probabilities, rad_update
from beampinn.sampling.samplers import PaperEpochSampler, term_specs
from beampinn.training.trainer import Trainer


def _hard_model(bid="FE-D-M1", seed=0):
    cfg = ExperimentConfig(seed=seed)
    cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 16, 8
    bm = get_benchmark(bid)
    net = build_model(cfg, bm).double()
    return HardConstrainedFF(net, bm.reference("exact"), bm.L, bm.t_end).double(), bm


@pytest.mark.parametrize("bid", ["FE-D-M1", "FE-D-M2"])
def test_hard_constraints_satisfy_all_ics_and_bcs_exactly(bid):
    m, bm = _hard_model(bid, seed=3)
    ref = bm.reference("exact")
    x = torch.linspace(0, bm.L, 101, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    t0 = torch.zeros_like(x).requires_grad_(True)
    u = m(x, t0)
    assert float((u.detach().numpy().ravel() - ref.u0(x.detach().numpy().ravel())).__abs__().max()) < 1e-12
    assert float(d(u, t0).abs().max()) < 1e-12                       # u_t(x, 0) = 0
    t = torch.rand(200, 1, dtype=torch.float64) * bm.t_end
    for xe in (0.0, bm.L):
        xb = torch.full_like(t, xe).requires_grad_(True)
        ub = m(xb, t)
        assert float(ub.abs().max()) / bm.A0 < 1e-12                  # u = 0 at both ends
        assert float(dxk(ub, xb, 1).abs().max()) * bm.L / bm.A0 < 1e-10   # u_x = 0 at both ends


def test_hard_constraint_does_not_freeze_the_interior():
    m, bm = _hard_model(seed=1)
    x = torch.tensor([[1.0]], dtype=torch.float64); t = torch.tensor([[0.5]], dtype=torch.float64)
    base = float(m.u0(x))
    assert abs(float(m(x, t)) - base) > 1e-6                          # the network term is active


def test_pde_only_grouping():
    assert [s.name for s in term_specs("fixed-fixed", "pde_only", 640)] == ["f"]


def test_rad_probability_formula():
    r = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    p = rad_probabilities(r, k=1.0, c=1.0)                   # r/mean(r) + c = [1, 1.75, 3.25]
    assert torch.allclose(p, torch.tensor([1.0, 1.75, 3.25], dtype=torch.float64) / 6.0)
    p_uniform = rad_probabilities(r, k=1.0, c=1e9)
    assert torch.allclose(p_uniform, torch.full((3,), 1 / 3, dtype=torch.float64))


def test_rad_update_installs_pool_without_duplicates_and_counts_candidates():
    m, bm = _hard_model(seed=2)
    s = PaperEpochSampler(bm, bm.reference("exact").u0, SamplerCfg(n_per_term=64, mini_batch=32),
                          "pde_only", 0, torch.float64)
    s.next_batch(); s.next_batch()                                    # finish epoch 0
    n_c, stats, (xs, ts) = rad_update(m, s, *bm.pde_coeffs(), RADCfg(k=1.0, c=1.0, n_candidates=500),
                                      torch.float64)
    assert n_c == 500 and len(xs) == 64 and len(np.unique(np.stack([xs, ts], 1), axis=0)) == 64
    b = s.next_batch()                                                # epoch 1 uses the pool
    pool = set(map(tuple, np.round(np.stack([xs, ts], 1), 12)))
    got = set(map(tuple, np.round(torch.cat([b["f"]["x"], b["f"]["t"]], 1).numpy(), 12)))
    assert got <= pool
    assert stats["rad_r_mean_selected"] >= stats["rad_r_mean_all"] * 0.99


def test_mode2_benchmark_identity():
    b1, b2 = get_benchmark("FE-D-M1"), get_benchmark("FE-D-M2")
    for f in ("bc_type", "L", "t_end", "b", "gamma_printed", "A0"):
        assert getattr(b1, f) == getattr(b2, f)
    assert b2.mode == 2 and math.isnan(b2.paper_L2)
    e = b2.reference("exact")
    assert abs(e.beta_l - 7.853204624) < 1e-8 and b2.reference("paper").beta_l == e.beta_l
    assert abs(e.omega / (2 * math.pi) - 56.76) < 0.01
    assert abs(e.x_norm - 2.75 / 2) > 0.3                             # mid-span is a node for mode 2


def _cfg(hard, rad, steps=6):
    c = ExperimentConfig(name="smoke_hr")
    c.model.depth, c.model.width, c.model.m_fourier = 2, 12, 6
    c.sampler.n_per_term, c.sampler.mini_batch = 64, 32
    if hard:
        c.loss.hard_constraints, c.loss.grouping, c.loss.weighting = "ff_tsq", "pde_only", "fixed"
    if rad:
        c.sampler.adaptive = "rad"; c.sampler.rad.every, c.sampler.rad.n_candidates = 2, 128
    c.train.max_steps, c.train.log_every, c.train.eval_every, c.train.ckpt_every = steps, 2, 3, 2
    return c


def test_trainer_hard_rad_smoke_and_accounting(tmp_path):
    c = _cfg(hard=True, rad=True)
    tr = Trainer(c, root=tmp_path)
    assert tr.run() == "completed"
    a = tr.acc
    assert a["optimizer_steps"] == 6 and a["training_points"] == 6 * 32      # one term only
    assert a["rad_updates"] == 2 and a["candidate_evaluations"] == 2 * 128     # steps 2 and 4
    assert set(tr.rad_snapshots) == {2, 4}


def test_hard_constraint_config_guards():
    c = _cfg(hard=True, rad=False); c.loss.weighting = "ntk"
    with pytest.raises(ValueError):
        c.validate()
    c = _cfg(hard=False, rad=True); c.sampler.rad.every = 3                   # not epoch-aligned
    with pytest.raises(ValueError):
        c.validate()


# ------------------------------------------------------------- Phase X additions
def _hard_tanh(bid="FE-D-M1", seed=0):
    cfg = ExperimentConfig(seed=seed)
    cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 16, 8
    bm = get_benchmark(bid)
    net = build_model(cfg, bm).double()
    w1 = get_benchmark("FE-D-M1").fundamental_omega()
    return HardConstrainedFF(net, bm.reference("exact"), bm.L, bm.t_end, "tanh2", w1).double(), bm


def test_fundamental_omega_is_mode1_frequency_from_pde_and_bcs():
    b1, b2 = get_benchmark("FE-D-M1"), get_benchmark("FE-D-M2")
    assert abs(b1.fundamental_omega() - b1.reference("exact").omega) < 1e-9
    assert b2.fundamental_omega() == b1.fundamental_omega()          # mode-independent
    assert abs(b1.fundamental_omega() - 129.37) < 0.01


@pytest.mark.parametrize("bid", ["FE-D-M1", "FE-D-M2"])
def test_tanh2_hard_constraints_satisfy_all_ics_and_bcs_exactly(bid):
    m, bm = _hard_tanh(bid, seed=4)
    ref = bm.reference("exact")
    x = torch.linspace(0, bm.L, 101, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    t0 = torch.zeros_like(x).requires_grad_(True)
    u = m(x, t0)
    assert np.abs(u.detach().numpy().ravel() - ref.u0(x.detach().numpy().ravel())).max() < 1e-12
    assert float(d(u, t0).abs().max()) < 1e-12
    t = torch.rand(200, 1, dtype=torch.float64) * bm.t_end
    for xe in (0.0, bm.L):
        xb = torch.full_like(t, xe).requires_grad_(True)
        ub = m(xb, t)
        assert float(ub.abs().max()) / bm.A0 < 1e-12
        assert float(dxk(ub, xb, 1).abs().max()) * bm.L / bm.A0 < 1e-10


def test_data_only_diagnostic_targets_exact_solution():
    bm = get_benchmark("FE-D-M1"); ref = bm.reference("exact")
    s = PaperEpochSampler(bm, ref.u0, SamplerCfg(n_per_term=64, mini_batch=32), "data_only", 0,
                          torch.float64, data_fn=ref.u)
    b = s.next_batch()
    assert list(b) == ["d"]
    x, t, g = (b["d"][k].numpy().ravel() for k in ("x", "t", "target"))
    assert np.abs(g - ref.u(x, t)).max() < 1e-15 and t.max() > 0


def test_data_only_and_tanh2_config_guards():
    c = ExperimentConfig(); c.loss.grouping, c.loss.weighting = "data_only", "fixed"
    c.validate()
    c.loss.weighting = "ntk"
    with pytest.raises(ValueError):
        c.validate()
    c = ExperimentConfig(); c.loss.hard_constraints, c.loss.grouping, c.loss.weighting = "ff_tanh2", "pde_only", "fixed"
    c.validate()


def test_existing_run_keys_unchanged():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name, rid in [("E4_hard_fourier", "E4_hard_fourier__s1234__705dd868f0"),
                      ("E4b_hard_fourier_rc", "E4b_hard_fourier_rc__s1234__1490710f87")]:
        c = ExperimentConfig.from_json(root / "results_optimization" / "configs" / f"{rid}.json")
        assert c.run_id() == rid
