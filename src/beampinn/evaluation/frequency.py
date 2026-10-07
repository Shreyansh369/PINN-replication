"""Parametric frequency / phase / amplitude / damping extraction.

The single-mode solution at a fixed x is EXACTLY  y(t) = a exp(-lam t) cos(w t + phi), so a
nonlinear least-squares fit of that model (plus an offset c) is unbiased for the reference,
unlike an FFT peak (1/T = 1 Hz bins on a 1 s record). Formal standard errors come from the
Gauss-Newton covariance; the estimator's sensitivity to model error is quantified separately
by Monte Carlo (experiments/stage01/frequency_extractor_uncertainty.py).
"""
import math

import numpy as np
from scipy.optimize import least_squares


def _initial_guess(t, y):
    dt = t[1] - t[0]
    nfft = 1 << 18
    spec = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y)), n=nfft))
    k = int(np.argmax(spec[1:]) + 1)
    if 1 <= k < len(spec) - 1:                           # parabolic peak interpolation
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-300)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    w0 = 2 * math.pi * k / (nfft * dt)
    # damping from log-amplitude of successive peaks (robust enough for a start value)
    env = np.abs(y)
    n_half = len(t) // 2
    e1, e2 = env[:n_half].max(), env[n_half:].max()
    lam0 = max(0.0, math.log(max(e1, 1e-300) / max(e2, 1e-300)) / (t[n_half] - t[0])) if e2 > 0 else 0.0
    a0 = float(np.abs(y).max())
    return np.array([a0, lam0, w0, 0.0, 0.0])


def fit_damped_cosine(t, y, p0=None, fit_damping=True):
    """Return dict(a, lam, w, phi, c, se_*) for y ~ a e^{-lam t} cos(w t + phi) + c."""
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    scale = float(np.abs(y).max()) or 1.0
    yn = y / scale
    p = _initial_guess(t, yn) if p0 is None else np.array(p0, dtype=np.float64) / [scale, 1, 1, 1, scale]
    if not fit_damping:
        p[1] = 0.0

    def model(q):
        lam = q[1] if fit_damping else 0.0
        return q[0] * np.exp(np.clip(-lam * t, -700.0, 700.0)) * np.cos(q[2] * t + q[3]) + q[4]

    best = None
    for phi0 in (p[3], p[3] + math.pi / 2, p[3] + math.pi, p[3] - math.pi / 2):
        q0 = p.copy(); q0[3] = phi0
        with np.errstate(over="ignore", invalid="ignore"):   # LM may probe extreme damping
            sol = least_squares(lambda q: model(q) - yn, q0, method="lm", xtol=1e-15, ftol=1e-15,
                                gtol=1e-15, max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
    q = best.x.copy()
    if q[0] < 0:                         # canonical sign: a > 0
        q[0], q[3] = -q[0], q[3] + math.pi
    q[3] = (q[3] + math.pi) % (2 * math.pi) - math.pi
    resid = best.fun
    dof = max(len(t) - len(q), 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(5, np.nan)
    return {"a": q[0] * scale, "lam": q[1] if fit_damping else 0.0, "w": q[2], "phi": q[3],
            "c": q[4] * scale, "se_a": se[0] * scale, "se_lam": se[1], "se_w": se[2],
            "se_phi": se[3], "rms_fit_resid": math.sqrt(2 * best.cost / len(t)) * scale}


def wrap(p):
    return (p + math.pi) % (2 * math.pi) - math.pi


def compare_fits(fit, a_true, lam_true, w_true, phi_true):
    """Errors of a fitted damped cosine against exact parameters."""
    return {
        "frequency_error": abs(fit["w"] - w_true) / w_true,
        "phase_error": abs(wrap(fit["phi"] - phi_true)),
        "amplitude_error": abs(fit["a"] - a_true) / abs(a_true),
        "damping_error": (abs(fit["lam"] - lam_true) / lam_true) if lam_true > 0 else abs(fit["lam"]),
        "frequency_se": fit["se_w"] / w_true,
    }
