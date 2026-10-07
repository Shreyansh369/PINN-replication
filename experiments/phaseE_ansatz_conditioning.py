"""Required network output for the hard-constraint ansatz (analysis only, no training).

For u = u0(x) + g(t) * Phi(x) * A0 * N(x,t), the network must represent
    N*(x,t) = (u_exact - u0) / (g(t) Phi(x) A0).
At mid-span (Phi = 1, u0 = A0) this is N*(t) = (q(t) - 1) / g(t).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from beampinn.physics.beam import modal_time
from beampinn.physics.benchmarks import get_benchmark

for bid in ("FE-D-M1", "FE-D-M2"):
    ref = get_benchmark(bid).reference("exact")
    t = np.linspace(1e-6, 1.0, 200001)
    q = modal_time(ref.omega, ref.gamma, t)
    for name, g in [("current  g=(t/T)^2", t ** 2),
                    ("tanh^2(w t), w=omega_1 from PDE coeffs", np.tanh(ref.omega * t) ** 2)]:
        N = (q - 1) / g
        print(f"{bid} {name:42s} N* range [{N.min():9.1f}, {N.max():7.2f}]  |N*| max/min over t>0.1 s: "
              f"{np.abs(N).max():.3g} / {np.abs(N[t > 0.1]).min():.3g}")
