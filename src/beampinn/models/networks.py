"""Networks. Inputs are PHYSICAL (x [m], t [s]); any normalisation happens inside the model,
so autograd derivatives are always taken with respect to physical coordinates and the PDE
residual is in the physical units of paper Eq. 49.

Fourier convention (configurable, documented in results_optimization/reports/CONVENTIONS.md):
    paper Eqs. 38-39 : gamma(v) = [cos(2 pi B v), sin(2 pi B v)],  v physical    (two_pi=True, 'physical')
    reference code   : gamma(v) = [sin(B v), cos(B v)],  v standardised          (two_pi=False, 'standardize')
B ~ N(0, sigma^2) (sigma = standard deviation, as in the reference code), frozen.
"""
import math

import torch
import torch.nn as nn

ACTIVATIONS = {"tanh": nn.Tanh}


class InputTransform(nn.Module):
    """v -> (v - shift)/scale for one coordinate. Fixed buffers, not trained."""

    def __init__(self, mode: str, lo: float, hi: float):
        super().__init__()
        if mode == "physical":
            shift, scale = 0.0, 1.0
        elif mode == "unit":
            shift, scale = lo, hi - lo
        elif mode == "standardize":           # exact mean/std of U[lo, hi]
            shift, scale = 0.5 * (lo + hi), (hi - lo) / math.sqrt(12.0)
        else:
            raise ValueError(mode)
        self.register_buffer("shift", torch.tensor(float(shift)))
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.mode = mode

    def forward(self, v):
        return (v - self.shift) / self.scale


class FourierEncoding(nn.Module):
    def __init__(self, m: int, sigma: float, two_pi: bool, generator: torch.Generator):
        super().__init__()
        self.register_buffer("B", torch.randn(1, m, generator=generator) * sigma)
        self.factor = 2 * math.pi if two_pi else 1.0
        self.sigma, self.two_pi = sigma, two_pi

    def forward(self, v):
        p = self.factor * (v @ self.B)
        return torch.cat([torch.cos(p), torch.sin(p)], dim=1)

    def angular_frequencies(self):
        """Angular frequencies (rad per unit of the TRANSFORMED input) of the features."""
        return (self.factor * self.B).abs().reshape(-1)


def _mlp(d_in, depth, width, act):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), act()]
        d_in = width
    return nn.Sequential(*layers)


def _init(module, weight_init, bias_init, generator):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if weight_init == "xavier_normal":
                std = math.sqrt(2.0 / (m.in_features + m.out_features))
                with torch.no_grad():
                    m.weight.copy_(torch.randn(m.weight.shape, generator=generator) * std)
            else:
                raise ValueError(weight_init)
            with torch.no_grad():
                if bias_init == "zeros":
                    m.bias.zero_()
                elif bias_init == "normal":
                    m.bias.copy_(torch.randn(m.bias.shape, generator=generator))
                else:
                    raise ValueError(bias_init)


class SpatioTemporalFourierPINN(nn.Module):
    """Paper Eqs. 38-43: M_x spatial and M_t temporal Fourier mappings share ONE trunk;
    H^(i,j) = H_x^(i) (.) H_t^(j); u = W [H^(1,1), ..., H^(Mx,Mt)] + b."""

    def __init__(self, cfg, L, t_end, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.tx = InputTransform(cfg.input_norm, 0.0, L)
        self.tt = InputTransform(cfg.input_norm, 0.0, t_end)
        self.enc_x = nn.ModuleList(FourierEncoding(cfg.m_fourier, s, cfg.two_pi, g) for s in cfg.sigma_x)
        self.enc_t = nn.ModuleList(FourierEncoding(cfg.m_fourier, s, cfg.two_pi, g) for s in cfg.sigma_t)
        self.trunk = _mlp(2 * cfg.m_fourier, cfg.depth, cfg.width, ACTIVATIONS[cfg.activation])
        self.head = nn.Linear(cfg.width * len(cfg.sigma_x) * len(cfg.sigma_t), 1)
        gw = torch.Generator().manual_seed(seed + 1)
        _init(self, cfg.weight_init, cfg.bias_init, gw)
        self.output_scale = float(cfg.output_scale)

    def forward(self, x, t):
        xs, ts = self.tx(x), self.tt(t)
        hx = [self.trunk(e(xs)) for e in self.enc_x]
        ht = [self.trunk(e(ts)) for e in self.enc_t]
        merged = [a * b for a in hx for b in ht]
        return self.output_scale * self.head(torch.cat(merged, dim=1))


class VanillaPINN(nn.Module):
    """Plain tanh MLP on (transformed) (x, t)."""

    def __init__(self, cfg, L, t_end, seed):
        super().__init__()
        self.tx = InputTransform(cfg.input_norm, 0.0, L)
        self.tt = InputTransform(cfg.input_norm, 0.0, t_end)
        self.net = nn.Sequential(_mlp(2, cfg.depth, cfg.width, ACTIVATIONS[cfg.activation]),
                                 nn.Linear(cfg.width, 1))
        _init(self, cfg.weight_init, cfg.bias_init, torch.Generator().manual_seed(seed + 1))
        self.output_scale = float(cfg.output_scale)

    def forward(self, x, t):
        return self.output_scale * self.net(torch.cat([self.tx(x), self.tt(t)], dim=1))


def build_model(cfg, benchmark):
    m = cfg.model
    cls = {"st_fourier": SpatioTemporalFourierPINN, "vanilla": VanillaPINN}[m.arch]
    return cls(m, benchmark.L, benchmark.t_end, cfg.seed)


def unwrap(model):
    return getattr(model, "net", model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())
