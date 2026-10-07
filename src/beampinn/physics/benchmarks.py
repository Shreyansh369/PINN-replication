"""Benchmark identities and the DUAL-REFERENCE system.

For every benchmark two references are built and BOTH are always reported:

  paper  : PAPER-FAITHFUL reference. The paper's analytical solution evaluated with the
           constants it PRINTS: rounded eigen-root (Eq. 28: beta1*l = 4.7300 for FF), the PDE
           coefficients of the printed PDE (Eq. 49: 43.73^2, 7.08). This is the reference the
           primary first-mode gate (L2_paper < 4.64e-4) is evaluated against.

  exact  : EXACT-PHYSICS reference. Exact solution of the PDE actually trained on: exact
           eigen-root of the BC frequency equation, same (c2, gamma) as the training PDE.

A third, diagnostic-only reference ('material': coefficients recomputed from Table 3
material data) is used to quantify the paper's internal constant inconsistencies; it is
never used for a gate.
"""
import math
from dataclasses import dataclass, field

from .beam import BeamCase, eigen_root

# PAPER Table 3 / A.10 / B.11: AISI 1040 steel, 30 mm square section.
STEEL = {"E": 2.0e11, "rho": 7845.0, "a": 0.030}
_A = STEEL["a"] ** 2
_I = STEEL["a"] ** 4 / 12.0
EI = STEEL["E"] * _I                 # 13 500 N m^2
RHO_A = STEEL["rho"] * _A            # 7.0605 kg/m
C2_MATERIAL = EI / RHO_A             # 1912.0459
C2_PRINTED = 43.73 ** 2              # Eqs. 46, 49, A.4, A.7, B.5, B.8 print "43.73^2"


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    bc_type: str
    mode: int
    L: float
    t_end: float
    b: float                     # damping constant [N s/m]
    gamma_printed: float         # PDE coefficient of u_t as printed by the paper
    beta_l_printed: float        # eigen-root as printed by the paper
    A0: float                    # IC displacement at x_norm [m] (read from the paper's figure)
    paper_L2: float
    paper_source: str
    paper_hyper: dict = field(default_factory=dict)
    notes: str = ""

    # ---- PDE coefficients of the TRAINING problem ------------------------------
    def pde_coeffs(self, which: str = "paper_eq49"):
        """(c2, gamma). 'paper_eq49' = the printed PDE; 'material' = Table 3 data."""
        if which == "paper_eq49":
            return C2_PRINTED, self.gamma_printed
        if which == "material":
            return C2_MATERIAL, self.b / RHO_A
        raise ValueError(which)

    # ---- references --------------------------------------------------------------
    def reference(self, kind: str, pde_coeffs: str = "paper_eq49") -> BeamCase:
        if kind == "paper":
            c2, g = self.pde_coeffs("paper_eq49")
            # no printed root (e.g. mode 2): the paper-style reference is the exact root
            bl = self.beta_l_printed if self.beta_l_printed is not None else eigen_root(self.bc_type, self.mode)
        elif kind == "exact":
            c2, g = self.pde_coeffs(pde_coeffs)
            bl = eigen_root(self.bc_type, self.mode)
        elif kind == "material":
            c2, g = self.pde_coeffs("material")
            bl = eigen_root(self.bc_type, self.mode)
        else:
            raise ValueError(kind)
        return BeamCase(f"{self.benchmark_id}:{kind}", self.bc_type, self.mode, self.L,
                        self.t_end, c2, g, bl, self.A0)

    def fundamental_omega(self, pde_coeffs: str = "paper_eq49"):
        """omega_1 = (beta_1 L / L)^2 sqrt(c2): mode-1 undamped frequency of THIS beam and BCs,
        from the PDE coefficient and the eigenproblem only (independent of mode and IC)."""
        c2, _ = self.pde_coeffs(pde_coeffs)
        return (eigen_root(self.bc_type, 1) / self.L) ** 2 * math.sqrt(c2)

    def references(self, pde_coeffs: str = "paper_eq49"):
        return {"paper": self.reference("paper"), "exact": self.reference("exact", pde_coeffs)}


_FE_HYPER = dict(depth=6, width=200, sigma_x=(1.0,), sigma_t=(10.0, 1.0), lr=1e-4,
                 batch=640, mini_batch=32, epochs=45000, source="Table 4 #12, Sec. 5.1.1")

BENCHMARKS = {
    # CANONICAL first-mode benchmark (approved).
    "FE-D-M1": Benchmark("FE-D-M1", "fixed-fixed", 1, 2.75, 1.0, 50.0, 7.08, 4.7300, 0.08,
                         4.64e-4, "Sec. 5.1.1, Eq. 49, Eq. 28, Fig. 5, Table 4 #12, Table 5, Fig. 6",
                         _FE_HYPER, "A0 = 0.08 m read from Fig. 5(c); rel-L2 is amplitude-invariant"),
    # Mode-2 HIGHER-FREQUENCY STRESS TEST (approved): identical to FE-D-M1 except mode = 2.
    # The paper publishes NO mode-2 result: paper_L2 is NaN and must never be quoted as a target.
    "FE-D-M2": Benchmark("FE-D-M2", "fixed-fixed", 2, 2.75, 1.0, 50.0, 7.08, None, 0.08,
                         float("nan"), "derived from FE-D-M1 (mode 2); no published target", _FE_HYPER,
                         "A0 at the mode-2 antinode; 'paper' reference = exact root (none printed)"),
    "FE-U-M1": Benchmark("FE-U-M1", "fixed-fixed", 1, 2.75, 1.0, 0.0, 0.0, 4.7300, 0.08,
                         2.70e-3, "Sec. 5.1.1, Eq. 46, Fig. 4", _FE_HYPER),
    # Secondary replication cases (later stages only).
    "SS-U-M1": Benchmark("SS-U-M1", "simply-supported", 1, 2.75, 1.0, 0.0, 0.0, 3.1416, 0.065,
                         2.3e-3, "App. A, Eq. A.3/A.4, Fig. A.19",
                         dict(batch=960, epochs=30000, source="App. A"),
                         "Eq. A.4 prints free-free BCs (typo); prose + A.3 give u = u_xx = 0"),
    "SS-D-M1": Benchmark("SS-D-M1", "simply-supported", 1, 2.75, 1.0, 50.0, 7.08, 3.1416, 0.065,
                         4.07e-2, "App. A, Eq. A.7, Fig. A.20", dict(batch=960, epochs=30000)),
    "CF-U-M1": Benchmark("CF-U-M1", "cantilever", 1, 4.0, 1.0, 0.0, 0.0, 1.8751, 0.15,
                         3.07e-5, "App. B, Eq. B.5, Fig. B.22", dict(batch=640, epochs=70000)),
    "CF-D-M1": Benchmark("CF-D-M1", "cantilever", 1, 4.0, 5.0, 5.0, 0.708, 1.8751, 0.15,
                         7.20e-4, "App. B, Eq. B.8, Fig. B.23", dict(batch=640, epochs=70000)),
}

CANONICAL = "FE-D-M1"


def get_benchmark(benchmark_id: str) -> Benchmark:
    try:
        return BENCHMARKS[benchmark_id]
    except KeyError:
        raise KeyError(f"unknown benchmark {benchmark_id!r}; known: {sorted(BENCHMARKS)}") from None


def frequency_hz(case: BeamCase) -> float:
    return case.omega / (2 * math.pi)
