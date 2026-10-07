"""Per-step cost of the hard-constraint model and of a RAD candidate sweep (no optimizer step)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from beampinn.config import ExperimentConfig
from beampinn.training.trainer import resolve_problem, setup_torch
from beampinn.models.networks import build_model
from beampinn.models.constraints import HardConstrainedFF
from beampinn.sampling.samplers import PaperEpochSampler
from beampinn.sampling.rad import candidate_residuals
from beampinn.losses.residuals import term_residuals, reduce_loss
cfg = ExperimentConfig(threads=1); cfg.loss.hard_constraints, cfg.loss.grouping, cfg.loss.weighting = "ff_tsq", "pde_only", "fixed"
setup_torch(cfg); bm, refs, c2, g, u0 = resolve_problem(cfg)
m = HardConstrainedFF(build_model(cfg, bm), refs["exact"], bm.L, bm.t_end)
s = PaperEpochSampler(bm, u0, cfg.sampler, "pde_only", 0)
def step():
    r = term_residuals(m, s.next_batch(), c2, g); l = sum(reduce_loss(v) for v in r.values()); m.zero_grad(); l.backward()
for _ in range(5): step()
t0 = time.perf_counter()
for _ in range(40): step()
print(f"hard 6x200 mb32 (1 thread): {(time.perf_counter()-t0)/40*1e3:.1f} ms/step")
x = torch.rand(5000, 1, dtype=torch.float64) * bm.L; t = torch.rand(5000, 1, dtype=torch.float64)
t0 = time.perf_counter(); candidate_residuals(m, x, t, c2, g, torch.float32); print(f"RAD sweep 5000 candidates: {time.perf_counter()-t0:.2f} s")
