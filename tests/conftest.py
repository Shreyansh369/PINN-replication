import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beampinn.physics.beam import BeamCase  # noqa: E402


class AnalyticProbe(nn.Module):
    """Torch implementation of a BeamCase (fixed-fixed or simply-supported), float64.

    Used as a stand-in 'network' so that residuals, metrics and the frequency extractor can
    be checked against a model whose exact answer is known.
    """

    def __init__(self, case: BeamCase):
        super().__init__()
        self.case = case
        self.dummy = nn.Parameter(torch.zeros(1, dtype=torch.float64))   # gives the model a dtype
        b, L = case.beta, case.L
        if case.bc_type == "fixed-fixed":
            self.s = (math.cosh(b * L) - math.cos(b * L)) / (math.sinh(b * L) - math.sin(b * L))
        from beampinn.physics.beam import mode_shape_raw
        self.unorm = float(mode_shape_raw(case.bc_type, b, L, case.x_norm))

    def shape(self, x):
        b = self.case.beta
        if self.case.bc_type == "simply-supported":
            return torch.sin(b * x) / self.unorm
        return (torch.cosh(b * x) - torch.cos(b * x) - self.s * (torch.sinh(b * x) - torch.sin(b * x))) / self.unorm

    def time(self, t):
        w, g = self.case.omega, self.case.gamma
        if g == 0:
            return torch.cos(w * t)
        wd = math.sqrt(w * w - g * g / 4)
        return torch.exp(-g * t / 2) * (torch.cos(wd * t) + g / (2 * wd) * torch.sin(wd * t))

    def forward(self, x, t):
        x, t = x.double(), t.double()
        return self.case.A0 * self.shape(x) * self.time(t) + 0.0 * self.dummy


@pytest.fixture
def probe_factory():
    return AnalyticProbe


@pytest.fixture(autouse=True)
def _float64_default():
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)
