"""Euler-Bernoulli beam: eigen-roots, mode shapes and the single-mode damped solution.

Governing equation in the paper's form (Eq. 2, divided by rho*A):

    c2 * u_xxxx + u_tt + gamma * u_t = 0,     c2 = EI/(rho A),  gamma = b/(rho A)

Everything here is float64 NumPy and closed form; nothing touches a network.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

# Required boundary conditions as derivative orders at each end (paper Table 1).
BC_ORDERS = {
    "fixed-fixed":      {"left": (0, 1), "right": (0, 1)},   # u = u_x = 0
    "simply-supported": {"left": (0, 2), "right": (0, 2)},   # u = u_xx = 0
    "cantilever":       {"left": (0, 1), "right": (2, 3)},   # fixed at 0, free at L
}


def eigen_root(bc_type: str, n: int = 1) -> float:
    """Exact beta_n * L for the n-th mode (root of the frequency equation)."""
    if bc_type == "simply-supported":
        return n * math.pi
    if bc_type == "fixed-fixed":          # cos z cosh z = 1, z_n ~ (n + 1/2) pi
        f, z0 = (lambda z: math.cos(z) * math.cosh(z) - 1.0), (n + 0.5) * math.pi
    elif bc_type == "cantilever":         # cos z cosh z = -1, z_n ~ (n - 1/2) pi
        f, z0 = (lambda z: math.cos(z) * math.cosh(z) + 1.0), (n - 0.5) * math.pi
    else:
        raise ValueError(f"unknown bc_type {bc_type!r}")
    return brentq(f, z0 - 0.4, z0 + 0.4, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)


def _hyp(fn, k, z):
    """k-th derivative pattern of cosh/sinh (without the beta^k factor)."""
    if fn == "cosh":
        return np.cosh(z) if k % 2 == 0 else np.sinh(z)
    return np.sinh(z) if k % 2 == 0 else np.cosh(z)


def _trig(fn, k, z):
    shift = k * math.pi / 2
    return np.cos(z + shift) if fn == "cos" else np.sin(z + shift)


def mode_shape_raw(bc_type: str, beta: float, L: float, x, k: int = 0):
    """k-th x-derivative of the un-normalised mode shape U(x) for wavenumber beta.

    fixed-fixed: U = cosh bx - cos bx - s (sinh bx - sin bx), s = (cosh bL - cos bL)/(sinh bL - sin bL)
    cantilever:  U = cosh bx - cos bx - s (sinh bx - sin bx), s = (cosh bL + cos bL)/(sinh bL + sin bL)
    simply-supp: U = sin bx
    (paper Eq. 25 / B.1 / A.1; paper Eq. 25 differs by an overall sign only.)
    """
    x = np.asarray(x, dtype=np.float64)
    z = beta * x
    bk = beta ** k
    if bc_type == "simply-supported":
        return bk * _trig("sin", k, z)
    bl = beta * L
    if bc_type == "fixed-fixed":
        s = (math.cosh(bl) - math.cos(bl)) / (math.sinh(bl) - math.sin(bl))
    elif bc_type == "cantilever":
        s = (math.cosh(bl) + math.cos(bl)) / (math.sinh(bl) + math.sin(bl))
    else:
        raise ValueError(bc_type)
    return bk * (_hyp("cosh", k, z) - _trig("cos", k, z) - s * (_hyp("sinh", k, z) - _trig("sin", k, z)))


def modal_time(omega: float, gamma: float, t, k: int = 0):
    """k-th derivative (k <= 2) of q(t):  q'' + gamma q' + omega^2 q = 0,  q(0)=1, q'(0)=0.

    Underdamped (paper Eq. 18 with eta'(0)=0):
        q = exp(-gamma t/2) [cos(wd t) + gamma/(2 wd) sin(wd t)],   wd = sqrt(omega^2 - gamma^2/4)
    """
    t = np.asarray(t, dtype=np.float64)
    if gamma == 0.0:
        return [np.cos(omega * t), -omega * np.sin(omega * t), -omega ** 2 * np.cos(omega * t)][k]
    wd2 = omega ** 2 - 0.25 * gamma ** 2
    if wd2 <= 0:
        raise NotImplementedError("only the underdamped case is implemented")
    wd = math.sqrt(wd2)
    env = np.exp(-0.5 * gamma * t)
    q = env * (np.cos(wd * t) + (0.5 * gamma / wd) * np.sin(wd * t))
    qd = -env * (omega ** 2 / wd) * np.sin(wd * t)
    if k == 0:
        return q
    if k == 1:
        return qd
    if k == 2:
        return -gamma * qd - omega ** 2 * q
    raise ValueError("k <= 2 only")


@dataclass(frozen=True)
class BeamCase:
    """One single-mode reference solution  u = A0 * U(x)/U(x_norm) * q(t).

    `beta_l` is the eigen-root USED (exact or the paper's rounded value). The same
    beta enters both the mode shape and the frequency, exactly as in paper Eq. 28.
    """
    name: str
    bc_type: str
    mode: int
    L: float
    t_end: float
    c2: float            # coefficient of u_xxxx
    gamma: float         # coefficient of u_t
    beta_l: float
    A0: float            # displacement at x_norm at t = 0 [m]

    @property
    def beta(self):
        return self.beta_l / self.L

    @property
    def omega(self):     # undamped natural frequency [rad/s]
        return self.beta ** 2 * math.sqrt(self.c2)

    @property
    def omega_d(self):   # damped frequency [rad/s]
        return math.sqrt(self.omega ** 2 - 0.25 * self.gamma ** 2)

    @property
    def x_norm(self):
        """Normalisation point: argmax |U| (mid-span for FF/SS mode 1, tip for CF)."""
        xs = np.linspace(0.0, self.L, 20001)
        return float(xs[np.argmax(np.abs(mode_shape_raw(self.bc_type, self.beta, self.L, xs)))])

    def _unorm(self):
        return float(mode_shape_raw(self.bc_type, self.beta, self.L, self.x_norm))

    def mode_shape(self, x, k=0):
        return mode_shape_raw(self.bc_type, self.beta, self.L, x, k) / self._unorm()

    def u(self, x, t, kx=0, kt=0):
        """d^kx/dx^kx d^kt/dt^kt of the reference displacement (broadcasting x, t)."""
        return self.A0 * self.mode_shape(x, kx) * modal_time(self.omega, self.gamma, t, kt)

    def u0(self, x):
        return self.A0 * self.mode_shape(x, 0)

    def pde_residual(self, x, t):
        """c2 u_xxxx + u_tt + gamma u_t of THIS reference (zero only if beta_l is the exact root
        and (c2, gamma) are the coefficients of the PDE being checked)."""
        return self.c2 * self.u(x, t, 4, 0) + self.u(x, t, 0, 2) + self.gamma * self.u(x, t, 0, 1)

    def damped_cosine_params(self):
        """Exact (a, lam, w, phi) with  q(t) = a exp(-lam t) cos(w t + phi)  (normalised q(0)=1)."""
        if self.gamma == 0.0:
            return 1.0, 0.0, self.omega, 0.0
        r = 0.5 * self.gamma / self.omega_d
        return math.sqrt(1.0 + r * r), 0.5 * self.gamma, self.omega_d, -math.atan(r)
