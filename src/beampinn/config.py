"""Central experiment configuration. Every experimental parameter lives here; nothing is
hard-coded in experiment functions. Serialises to/from JSON; `key()` hashes everything
that affects the result.

Mechanisms that are NOT yet approved (RAD, RAR, annealing, GradNorm, hard constraints,
mixed formulation, causal weighting) exist as fields so that configurations are complete,
but `validate()` refuses to run them until they are implemented in their own phase.
"""
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class BenchmarkCfg:
    benchmark_id: str = "FE-D-M1"
    pde_coeffs: str = "paper_eq49"      # 'paper_eq49' (43.73^2, 7.08) | 'material' (Table 3 data)
    ic_shape: str = "exact"             # 'exact' (exact-root mode shape) | 'paper' (rounded beta1*l)
    amplitude: Optional[float] = None   # None -> benchmark default A0


@dataclass
class ModelCfg:
    arch: str = "st_fourier"            # 'st_fourier' (paper Eqs. 38-43) | 'vanilla'
    depth: int = 6                      # Table 4 #12
    width: int = 200
    activation: str = "tanh"
    m_fourier: int = 100                # features per mapping (reference code layers[0]//2)
    sigma_x: Tuple[float, ...] = (1.0,)
    sigma_t: Tuple[float, ...] = (10.0, 1.0)
    two_pi: bool = True                 # Eqs. 38-39: cos(2 pi B v)
    input_norm: str = "physical"        # 'physical' | 'unit' | 'standardize'
    output_scale: float = 1.0
    weight_init: str = "xavier_normal"
    bias_init: str = "normal"           # 'normal' N(0,1) (reference code) | 'zeros' (legacy)


@dataclass
class RADCfg:            # placeholder, Phase B
    k: float = 1.0
    c: float = 1.0
    every: int = 1000
    n_candidates: int = 10000


@dataclass
class RARCfg:            # placeholder, Phase C
    every: int = 1000
    n_candidates: int = 10000
    n_add: int = 32
    k: float = 2.0
    c: float = 0.0


@dataclass
class SamplerCfg:
    kind: str = "paper_epoch"           # 'paper_epoch'
    n_per_term: int = 640               # Eq. 48: N_u = N_ut = N_ux = N_f = 640
    mini_batch: int = 32                # Sec. 5.1.1
    resample: str = "epoch"             # 'epoch' (Sec. 5.1.2 "sampled ... for each epoch") | 'step' | 'never'
    ic_fraction_in_u: float = 0.5       # share of the grouped L_u points that are IC points
    adaptive: str = "none"              # 'none' | 'rad' | 'rar'  (not yet approved)
    rad: RADCfg = field(default_factory=RADCfg)
    rar: RARCfg = field(default_factory=RARCfg)


@dataclass
class NTKCfg:
    every: int = 100                    # reference code: it % 100 == 0
    ema: float = 1.0                    # 1.0 = replace (reference code); <1 = running average
    trace_norm: str = "mean"            # 'mean' | 'sum' (identical when N_i are equal)
    max_rows: int = 32                  # rows per term for the EXACT trace; = the paper mini-batch.
                                        # Held fixed when the mini-batch changes so NTK cost is not
                                        # confounded with batch size (cost grows superlinearly in rows).


@dataclass
class LossCfg:
    grouping: str = "paper"             # 'paper' (Eq. 48: L_u = IC u + BC u) | 'split'
    reduction: str = "half_mean"        # Eq. 48: 1/(2N) sum r^2 | 'mean'
    pde_scale: float = 1.0              # residual multiplier (1 = physical units of Eq. 49)
    weighting: str = "ntk"              # 'ntk' | 'fixed' | 'annealing' | 'gradnorm'
    fixed_weights: dict = field(default_factory=dict)
    ntk: NTKCfg = field(default_factory=NTKCfg)
    hard_constraints: str = "none"      # 'none' | 'ff_tsq' (not yet approved)
    mixed_formulation: bool = False     # not yet approved
    causal: bool = False                # REJECTED (negative result); kept for completeness


@dataclass
class OptimCfg:
    name: str = "adam"
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    schedule: str = "constant"          # 'constant' | 'exp_decay'
    decay_rate: float = 0.9
    decay_steps: int = 1000
    grad_clip: Optional[float] = None
    lbfgs_steps: int = 0                # not yet approved


@dataclass
class TrainCfg:
    epochs: int = 45000                 # Sec. 5.1.1
    max_steps: Optional[int] = None     # screening budget cap (S0/S1/S2); None = full epochs
    budget_label: str = "PAPER"
    log_every: int = 500
    eval_every: int = 2000
    ckpt_every: int = 10000
    snapshot_every: Optional[int] = None   # keep step_<N>.pt every N steps (analysis only)
    log_grad_norms: bool = True


@dataclass
class ExperimentConfig:
    name: str = "run"
    seed: int = 1234
    mode: int = 1
    precision: str = "float32"
    device: str = "cpu"
    threads: int = 4
    benchmark: BenchmarkCfg = field(default_factory=BenchmarkCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    sampler: SamplerCfg = field(default_factory=SamplerCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    notes: str = ""

    # ------------------------------------------------------------------ I/O
    def to_dict(self):
        return asdict(self)

    def to_json(self, path=None):
        s = json.dumps(self.to_dict(), indent=2, sort_keys=False)
        if path is not None:
            with open(path, "w") as f:
                f.write(s + "\n")
        return s

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def key(self):
        """Hash of every field that affects the result (labels and logging cadence excluded)."""
        d = self.to_dict()
        d.pop("name"); d.pop("notes")
        for k in ("log_every", "eval_every", "ckpt_every", "budget_label", "snapshot_every"):
            d["train"].pop(k, None)
        return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def run_id(self):
        return f"{self.name}__s{self.seed}__{self.key()}"

    # ------------------------------------------------------------- guards
    def validate(self):
        from .physics.benchmarks import get_benchmark
        bm = get_benchmark(self.benchmark.benchmark_id)
        if self.mode != bm.mode:
            raise ValueError(f"mode {self.mode} does not match benchmark mode {bm.mode}")
        pending = {
            "sampler.adaptive=rar": self.sampler.adaptive == "rar",
            "loss.weighting=annealing/gradnorm": self.loss.weighting in ("annealing", "gradnorm"),
            "loss.hard_constraints (unknown)": self.loss.hard_constraints not in ("none", "ff_tsq", "ff_tanh2"),
            "loss.mixed_formulation": self.loss.mixed_formulation,
            "optim.lbfgs_steps": self.optim.lbfgs_steps > 0,
        }
        for k, on in pending.items():
            if on:
                raise NotImplementedError(f"{k} belongs to a later, not yet approved phase")
        if self.loss.causal:
            raise NotImplementedError("temporal causal weighting is REJECTED (REPORT.md 5.3)")
        if self.loss.weighting not in ("ntk", "fixed"):
            raise ValueError(self.loss.weighting)
        if self.loss.grouping == "data_only":      # DIAGNOSTIC ONLY (uses the solution as data)
            if (self.loss.hard_constraints != "none" or self.loss.weighting != "fixed"
                    or self.sampler.adaptive != "none"):
                raise ValueError("data_only diagnostic: no hard constraints, NTK or RAD")
        if self.loss.hard_constraints in ("ff_tsq", "ff_tanh2"):
            if bm.bc_type != "fixed-fixed" or self.benchmark.ic_shape != "exact":
                raise ValueError("ff_tsq needs a fixed-fixed beam and the exact-root IC shape")
            if self.loss.grouping != "pde_only" or self.loss.weighting != "fixed":
                raise ValueError("ff_tsq: use grouping='pde_only' and weighting='fixed' (one loss term)")
        if self.sampler.adaptive == "rad" and self.sampler.rad.every % self.steps_per_epoch():
            raise ValueError("RAD 'every' must be a multiple of steps_per_epoch (epoch-aligned updates)")
        if self.sampler.n_per_term % self.sampler.mini_batch:
            raise ValueError("n_per_term must be a multiple of mini_batch")
        if self.model.input_norm not in ("physical", "unit", "standardize"):
            raise ValueError(self.model.input_norm)
        return self

    # ------------------------------------------------------------- budget
    def steps_per_epoch(self):
        return self.sampler.n_per_term // self.sampler.mini_batch

    def total_steps(self):
        full = self.train.epochs * self.steps_per_epoch()
        return full if self.train.max_steps is None else min(full, self.train.max_steps)


def _from_dict(cls, d):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if dataclasses.is_dataclass(f.type) or (isinstance(f.type, type) and dataclasses.is_dataclass(f.type)):
            v = _from_dict(f.type, v)
        elif isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)
