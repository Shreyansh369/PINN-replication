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
    def __init__(self, net, case, L, T):
        super().__init__()
        self.net, self.u0 = net, FixedFixedModeShape(case)
        self.L, self.T, self.A0 = float(L), float(T), float(case.A0)

    def phi(self, x):
        return 16.0 * x ** 2 * (self.L - x) ** 2 / self.L ** 4

    def forward(self, x, t):
        return self.u0(x) + (t / self.T) ** 2 * self.phi(x) * self.A0 * self.net(x, t)
