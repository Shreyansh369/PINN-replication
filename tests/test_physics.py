"""Analytical physics: roots, frequencies, PDE, BCs, ICs, damped modal function, dual references."""
import math

import numpy as np
import pytest

from beampinn.physics.beam import BC_ORDERS, eigen_root, modal_time
from beampinn.physics.benchmarks import BENCHMARKS, C2_MATERIAL, C2_PRINTED, frequency_hz, get_benchmark

KNOWN_ROOTS = {"fixed-fixed": [4.730040745, 7.853204624, 10.995607838],
               "cantilever": [1.875104069, 4.694091133, 7.854757438],
               "simply-supported": [math.pi, 2 * math.pi, 3 * math.pi]}


@pytest.mark.parametrize("bc", list(KNOWN_ROOTS))
@pytest.mark.parametrize("n", [1, 2, 3])
def test_eigen_roots(bc, n):
    assert abs(eigen_root(bc, n) - KNOWN_ROOTS[bc][n - 1]) < 1e-8


def test_material_constants():
    assert abs(C2_MATERIAL - 1912.0459) < 1e-3
    assert abs(C2_PRINTED - 43.73 ** 2) < 1e-12


def test_paper_frequencies_reproduced():
    # PAPER Table 3 / A.10 / B.11 natural frequencies, to the paper's printed precision
    for bid, f_paper in [("FE-D-M1", 20.594), ("SS-U-M1", 9.085), ("CF-U-M1", 1.529)]:
        f = frequency_hz(get_benchmark(bid).reference("material"))
        assert abs(f - f_paper) / f_paper < 5e-4, (bid, f)


ALL_REFS = [(bid, k) for bid in BENCHMARKS for k in ("paper", "exact", "material")]


@pytest.mark.parametrize("bid,kind", ALL_REFS)
def test_reference_satisfies_its_pde(bid, kind):
    c = get_benchmark(bid).reference(kind)
    x = np.linspace(0, c.L, 101)[:, None]
    t = np.linspace(0, c.t_end, 501)[None, :]
    r = c.pde_residual(x, t)
    assert np.abs(r).max() / np.abs(c.u(x, t, 0, 2)).max() < 1e-12


@pytest.mark.parametrize("bid", list(BENCHMARKS))
def test_exact_reference_satisfies_all_bcs_and_ics(bid):
    c = get_benchmark(bid).reference("exact")
    t = np.linspace(0, c.t_end, 401)
    for end, orders in BC_ORDERS[c.bc_type].items():
        xe = 0.0 if end == "left" else c.L
        for k in orders:
            v = np.abs(c.u(xe, t, k, 0)).max() * c.L ** k / c.A0
            assert v < 1e-9, (end, k, v)
    x = np.linspace(0, c.L, 201)
    assert np.abs(c.u(x, 0.0) - c.u0(x)).max() < 1e-15
    assert np.abs(c.u(x, 0.0, 0, 1)).max() < 1e-12
    assert abs(c.u0(c.x_norm) - c.A0) < 1e-12


def test_paper_reference_violates_fixed_end_slope_bc():
    """Rounded beta1*l = 4.7300 keeps U(L) = 0 but breaks U'(L) = 0 (source of the floor)."""
    c = get_benchmark("FE-D-M1").reference("paper")
    slope = abs(c.u(c.L, 0.0, 1, 0)) * c.L / c.A0
    assert 1e-4 < slope < 1e-3
    assert abs(c.u(c.L, 0.0)) / c.A0 < 1e-12


def test_damped_modal_time_derivatives_and_ics():
    w, g = 129.3, 7.08
    t = np.linspace(0, 1, 20001)
    q, qd, qdd = (modal_time(w, g, t, k) for k in range(3))
    assert abs(q[0] - 1) < 1e-15 and abs(qd[0]) < 1e-12
    assert np.abs(qdd + g * qd + w * w * q).max() < 1e-9
    h = t[1] - t[0]
    fd = (q[2:] - q[:-2]) / (2 * h)
    assert np.abs(fd - qd[1:-1]).max() / np.abs(qd).max() < 1e-5


@pytest.mark.parametrize("bid", ["FE-D-M1", "FE-U-M1", "SS-D-M1"])
def test_damped_cosine_parameterisation(bid):
    c = get_benchmark(bid).reference("exact")
    a, lam, w, phi = c.damped_cosine_params()
    t = np.linspace(0, 1, 3001)
    assert np.abs(a * np.exp(-lam * t) * np.cos(w * t + phi) - modal_time(c.omega, c.gamma, t)).max() < 1e-12


def test_dual_reference_identity():
    bm = get_benchmark("FE-D-M1")
    p, e = bm.reference("paper"), bm.reference("exact")
    assert p.c2 == e.c2 == 43.73 ** 2 and p.gamma == e.gamma == 7.08
    assert p.beta_l == 4.7300 and abs(e.beta_l - 4.730040745) < 1e-9
    assert bm.paper_L2 == 4.64e-4 and bm.bc_type == "fixed-fixed" and bm.mode == 1
    assert (bm.L, bm.t_end, bm.b) == (2.75, 1.0, 50.0)
    # material reference only differs through the coefficients
    m = bm.reference("material")
    assert m.beta_l == e.beta_l and abs(m.gamma - 50.0 / 7.0605) < 1e-12
