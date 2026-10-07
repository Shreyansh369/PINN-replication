"""Target frequency vs Fourier-feature frequency support, for the three input conventions
that appear in the paper (Eq. 38-39: 2*pi*B*t), its reference code (B*standardised t) and
this repository (B*t on [0,1]).  Analytical: for B ~ N(0, s^2) the expected number of the m
features whose angular frequency exceeds w is m * P(|z| > w/s_eff)."""
import math
from scipy.stats import norm

targets = {"FE mode 1 (20.59 Hz)": 129.364, "SS mode 1 (9.08 Hz)": 57.067,
           "FE mode 2 (56.7 Hz)": 356.6, "CF mode 1 (1.53 Hz)": 9.609}
t_std = 1 / math.sqrt(12)                      # std of U[0,1]
conv = {"paper Eq.38-39 (2*pi*B*t, t in [0,1] s)": lambda s: 2 * math.pi * s,
        "reference code (B*(t-mu)/sd)":            lambda s: s / t_std,
        "this repo (B*t*, t* in [0,1])":            lambda s: s}
m = 100
print(f"{'convention':42s} {'sigma':>5s} {'s_eff [rad/s]':>13s} " + " ".join(f"{k:>22s}" for k in targets))
for name, f in conv.items():
    for s in (1.0, 10.0):
        se = f(s)
        cells = [f"{w/se:5.2f}sd, {m*2*norm.sf(w/se):6.2f} feats" for w in targets.values()]
        print(f"{name:42s} {s:5.0f} {se:13.2f} " + " ".join(f"{c:>22s}" for c in cells))
print(f"(m = {m} features per mapping; 'feats' = expected count with |angular freq| > target)")
