"""Hard enforcement of the fixed-fixed BCs and both ICs (loss.hard_constraints = 'ff_tsq').

    u(x,t) = u0(x) + (t/T)^2 * Phi(x) * A0 * N_theta(x,t),     Phi(x) = 16 x^2 (L-x)^2 / L^4

Derivation (u0 = exact-root mode shape * A0, so u0 = u0' = 0 at x = 0 and x = L):
  IC   u(x,0)   = u0(x)                                   (the (t/T)^2 factor vanishes)
       u_t(x,0) = [2t/T^2 Phi A0 N + (t/T)^2 Phi A0 N_t]_{t=0} = 0
  BC   u(0,t)   = u0(0) + (t/T)^2 Phi(0) A0 N = 0          (Phi(0) = 0)
       u_x(0,t) = u0'(0) + (t/T)^2 A0 [Phi'(0) N + Phi(0) N_x] = 0   (double root: Phi(0) = Phi'(0) = 0)
       identically at x = L (Phi(L) = Phi'(L) = 0).
All six conditions hold for ANY network N, so only the PDE residual is trained.
Phi is normalised to max 1 (at x = L/2) and N is scaled by A0 so that N = O(1).
Using u0 as the base is a PROBLEM-SPECIFIC choice (it uses the known IC shape); it does not
encode the temporal solution.
"""
import math

import torch
import torch.nn as nn

from ..physics.beam import mode_shape_raw


class FixedFixedModeShape(nn.Module):
    """Differentiable A0 * U(x)/U(x_norm) for the fixed-fixed mode shape (torch)."""

    def __init__(self, case):
        super().__init__()
        if case.bc_type != "fixed-fixed":
            raise ValueError("ff_tsq hard constraints are derived for fixed-fixed beams only")
        b, L = case.beta, case.L
        self.b = b
        self.s = (math.cosh(b * L) - math.cos(b * L)) / (math.sinh(b * L) - math.sin(b * L))
        self.scale = case.A0 / float(mode_shape_raw(case.bc_type, b, L, case.x_norm))

    def forward(self, x):
        z = self.b * x
        return self.scale * (torch.cosh(z) - torch.cos(z) - self.s * (torch.sinh(z) - torch.sin(z)))


class HardConstrainedFF(nn.Module):
    """time_factor 't2'    : g(t) = (t/T)^2                      (loss.hard_constraints = 'ff_tsq')
       time_factor 'tanh2' : g(t) = tanh^2(omega_1 t)            (loss.hard_constraints = 'ff_tanh2')
    omega_1 = (beta_1 L / L)^2 sqrt(c2): the beam's FUNDAMENTAL undamped frequency from the PDE
    coefficient and the BC eigenproblem only (never from the solution); a declared problem-specific
    prior. g(0) = 0 and g'(0) = 0 for both, so all six IC/BC conditions hold exactly (tests).
    Motivation (measured): with (t/T)^2 the network must output N* -> -omega^2/2 ~ -8.4e3 near t = 0;
    with tanh^2(omega_1 t) the required N* stays in [-1.9, -0.16] (phaseE_ansatz_conditioning.txt)."""

    def __init__(self, net, case, L, T, time_factor="t2", omega_1=None):
        super().__init__()
        self.net, self.u0 = net, FixedFixedModeShape(case)
        self.L, self.T, self.A0 = float(L), float(T), float(case.A0)
        if time_factor not in ("t2", "tanh2"):
            raise ValueError(time_factor)
        if time_factor == "tanh2" and not omega_1:
            raise ValueError("tanh2 needs omega_1")
        self.time_factor, self.omega_1 = time_factor, float(omega_1 or 0.0)

    def g(self, t):
        if self.time_factor == "t2":
            return (t / self.T) ** 2
        return torch.tanh(self.omega_1 * t) ** 2

    def phi(self, x):
        return 16.0 * x ** 2 * (self.L - x) ** 2 / self.L ** 4

    def forward(self, x, t):
        return self.u0(x) + self.g(t) * self.phi(x) * self.A0 * self.net(x, t)
