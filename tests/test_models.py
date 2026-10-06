"""Fourier conventions, architecture, autograd derivatives and float32 adequacy."""
import copy
import math

import numpy as np
import pytest
import torch

from beampinn.config import ExperimentConfig, ModelCfg
from beampinn.losses.residuals import dxk, d, pde_residual
from beampinn.models.networks import (FourierEncoding, InputTransform, SpatioTemporalFourierPINN,
                                      build_model, count_parameters)
from beampinn.physics.benchmarks import get_benchmark


def test_fourier_two_pi_convention():
    g = torch.Generator().manual_seed(0)
    enc = FourierEncoding(8, 10.0, True, g)
    enc = enc.double()
    v = torch.tensor([[0.3]], dtype=torch.float64)
    p = 2 * math.pi * 0.3 * enc.B
    assert torch.allclose(enc(v), torch.cat([torch.cos(p), torch.sin(p)], 1), rtol=0, atol=1e-12)
    enc0 = FourierEncoding(8, 10.0, False, torch.Generator().manual_seed(0)).double()
    assert torch.equal(enc0.B, enc.B)                          # same draw, only the factor differs
    q = 0.3 * enc.B
    assert torch.allclose(enc0(v), torch.cat([torch.cos(q), torch.sin(q)], 1), rtol=0, atol=1e-12)


def test_fourier_sigma_is_standard_deviation():
    enc = FourierEncoding(200000, 10.0, False, torch.Generator().manual_seed(1))
    assert abs(float(enc.B.std()) - 10.0) < 0.1


@pytest.mark.parametrize("mode,lo,hi,exp", [("physical", 0, 2.75, (0.0, 1.0)), ("unit", 0, 2.75, (0.0, 2.75)),
                                             ("standardize", 0, 2.75, (1.375, 2.75 / math.sqrt(12)))])
def test_input_transforms(mode, lo, hi, exp):
    tr = InputTransform(mode, lo, hi).double()
    assert abs(float(tr.shift) - exp[0]) < 1e-6 and abs(float(tr.scale) - exp[1]) < 1e-6   # float32 buffers
    if mode == "standardize":
        v = torch.linspace(lo, hi, 200001, dtype=torch.float64).reshape(-1, 1)
        z = tr(v)
        assert abs(float(z.mean())) < 1e-6 and abs(float(z.std()) - 1) < 1e-4


def test_paper_architecture_parameter_count():
    cfg = ExperimentConfig()                                  # 6 x 200, m = 100, Mx = 1, Mt = 2
    m = build_model(cfg, get_benchmark("FE-D-M1"))
    expect = (200 * 200 + 200) + 5 * (200 * 200 + 200) + (200 * 2 + 1)
    assert count_parameters(m) == expect == 241601


def test_multiplicative_merge_structure():
    mc = ModelCfg(depth=2, width=8, m_fourier=4, sigma_x=(1.0,), sigma_t=(10.0, 1.0))
    m = SpatioTemporalFourierPINN(mc, 2.75, 1.0, seed=0).double()
    x = torch.rand(5, 1, dtype=torch.float64) * 2.75
    t = torch.rand(5, 1, dtype=torch.float64)
    hx = m.trunk(m.enc_x[0](m.tx(x)))
    hts = [m.trunk(e(m.tt(t))) for e in m.enc_t]
    manual = m.head(torch.cat([hx * h for h in hts], 1))
    assert torch.allclose(m(x, t), manual)


def test_weights_are_seeded_and_biases_follow_config():
    c1 = ExperimentConfig(); c1.model.depth, c1.model.width = 2, 16
    bm = get_benchmark("FE-D-M1")
    a, b = build_model(c1, bm), build_model(c1, bm)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    assert float(a.trunk[0].bias.detach().abs().sum()) > 0           # 'normal' (reference code)
    c1.model.bias_init = "zeros"
    assert float(build_model(c1, bm).trunk[0].bias.detach().abs().sum()) == 0


def test_autograd_fourth_derivative_matches_finite_difference():
    mc = ModelCfg(depth=2, width=16, m_fourier=8, sigma_x=(1.0,), sigma_t=(10.0, 1.0))
    m = SpatioTemporalFourierPINN(mc, 2.75, 1.0, seed=3).double()
    x0, t0 = 1.1, 0.37
    x = torch.tensor([[x0]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[t0]], dtype=torch.float64)
    u4 = float(dxk(m(x, t), x, 4).detach())
    f = lambda xx: float(m(torch.tensor([[xx]], dtype=torch.float64), t).detach())
    fd = lambda h: (f(x0 - 2 * h) - 4 * f(x0 - h) + 6 * f(x0) - 4 * f(x0 + h) + f(x0 + 2 * h)) / h ** 4
    h = 1e-2
    e1, e2 = abs(fd(h) - u4), abs(fd(h / 2) - u4)
    assert 3.5 < e1 / e2 < 4.5                               # O(h^2) convergence TO the autograd value
    rich = (4 * fd(h / 2) - fd(h)) / 3                      # Richardson extrapolation
    assert abs(rich - u4) / abs(u4) < 1e-4


def test_float32_adequate_for_pde_residual_at_paper_scales():
    """Same random paper-size network in float32 and float64: PDE residual of Eq. 49 at
    the FE-D-M1 scales (2 pi sigma_t = 63 rad/s features, c2 = 1912, gamma = 7.08)."""
    torch.manual_seed(0)
    cfg = ExperimentConfig()
    bm = get_benchmark("FE-D-M1")
    m32 = build_model(cfg, bm).float()
    m64 = copy.deepcopy(m32).double()
    g = torch.Generator().manual_seed(5)
    x = torch.rand(256, 1, generator=g, dtype=torch.float64) * 2.75
    t = torch.rand(256, 1, generator=g, dtype=torch.float64)
    c2, gam = bm.pde_coeffs()
    r64 = pde_residual(m64, x.clone().requires_grad_(True), t.clone().requires_grad_(True), c2, gam)
    r32 = pde_residual(m32, x.float().requires_grad_(True), t.float().requires_grad_(True), c2, gam)
    rel = float((r32.double() - r64).norm() / r64.norm())
    assert rel < 1e-3, rel
