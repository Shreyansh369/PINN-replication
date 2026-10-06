"""Training loop with paper mini-batch semantics, NTK weighting, compute accounting and
checkpoint/resume.

Accounting (all cumulative, survive resume):
  optimizer_steps        Adam steps taken
  training_points        sum over terms of points that entered a training loss
  pde_evaluations        PDE-residual points evaluated for training (incl. those reused by NTK)
  grad_evaluations       full-loss backward passes (1 per step)
  ntk_row_gradients      per-row parameter gradients computed by NTK updates
  diag_gradients         per-term backward passes for gradient-norm logging (diagnostic only)
  candidate_evaluations  residual evaluations of sampler candidates (adaptive samplers; 0 here)
  train_seconds          wall-clock of training work only (steps + NTK); excludes validation,
                         diagnostics, checkpointing and final evaluation, which are timed apart
"""
import dataclasses
import math
import time

import numpy as np
import torch

from ..config import ExperimentConfig
from ..evaluation.metrics import evaluate_full, validation_metrics
from ..losses.residuals import reduce_loss, term_residuals
from ..losses.weighting import FixedWeighting, NTKWeighting
from ..models.networks import build_model, count_parameters, model_size_bytes
from ..optimization.optimizers import build_optimizer
from ..physics.benchmarks import get_benchmark
from ..profiling.resources import hardware_info, peak_rss_mb, peak_vram_mb, reset_peak_rss
from ..models.constraints import HardConstrainedFF
from ..sampling.rad import rad_update
from ..sampling.samplers import PaperEpochSampler
from ..utils.io import RESULTS, run_paths, save_checkpoint, write_history, write_json

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def resolve_problem(cfg: ExperimentConfig):
    """Benchmark, both references, training-PDE coefficients and the IC function."""
    bm = get_benchmark(cfg.benchmark.benchmark_id)
    refs = bm.references(cfg.benchmark.pde_coeffs)
    if cfg.benchmark.amplitude is not None:
        refs = {k: dataclasses.replace(r, A0=float(cfg.benchmark.amplitude)) for k, r in refs.items()}
    c2, gamma = bm.pde_coeffs(cfg.benchmark.pde_coeffs)
    ic_ref = refs[cfg.benchmark.ic_shape]
    return bm, refs, c2, gamma, ic_ref.u0


def build_hard(net, cfg, bm, refs):
    tf = {"ff_tsq": "t2", "ff_tanh2": "tanh2"}[cfg.loss.hard_constraints]
    return HardConstrainedFF(net, refs["exact"], bm.L, bm.t_end, time_factor=tf,
                             omega_1=bm.fundamental_omega(cfg.benchmark.pde_coeffs))


def setup_torch(cfg):
    torch.set_num_threads(cfg.threads)
    torch.set_default_dtype(DTYPES[cfg.precision])
    torch.set_flush_denormal(True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)


class Trainer:
    def __init__(self, cfg: ExperimentConfig, root=RESULTS):
        self.cfg = cfg.validate()
        setup_torch(cfg)
        self.bm, self.refs, self.c2, self.gamma, self.ic_fn = resolve_problem(cfg)
        self.model = build_model(cfg, self.bm).to(DTYPES[cfg.precision])
        if cfg.loss.hard_constraints in ("ff_tsq", "ff_tanh2"):
            self.model = build_hard(self.model, cfg, self.bm, self.refs).to(DTYPES[cfg.precision])
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt, self.sched = build_optimizer(self.params, cfg.optim)
        self.sampler = PaperEpochSampler(self.bm, lambda x: self.ic_fn(x), cfg.sampler,
                                         cfg.loss.grouping, cfg.seed, DTYPES[cfg.precision],
                                         data_fn=self.refs["exact"].u)
        names = [s.name for s in self.sampler.specs]
        self.weighting = (NTKWeighting(names, cfg.loss.ntk) if cfg.loss.weighting == "ntk"
                          else FixedWeighting(names, cfg.loss.fixed_weights))
        self.run_id = cfg.run_id()
        self.paths = run_paths(self.run_id, root)
        self.step = 0
        self.history = []
        self.rad_snapshots = {}
        self.status = "initialised"
        self.acc = dict(optimizer_steps=0, training_points=0, pde_evaluations=0, grad_evaluations=0,
                        ntk_row_gradients=0, ntk_updates=0, ntk_seconds=0.0, diag_gradients=0,
                        diag_seconds=0.0, candidate_evaluations=0, rad_updates=0, rad_seconds=0.0,
                        train_seconds=0.0,
                        validation_seconds=0.0, peak_rss_mb=0.0, segments=0)

    # ----------------------------------------------------------- checkpoints
    def _blob(self):
        return {"run_id": self.run_id, "config": self.cfg.to_dict(), "step": self.step,
                "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "sampler": self.sampler.state_dict(),
                "weighting": self.weighting.state_dict(), "acc": dict(self.acc),
                "history": list(self.history), "status": self.status, "rad_snapshots": self.rad_snapshots,
                "torch_rng": torch.get_rng_state()}

    def save(self, name="latest.pt"):
        save_checkpoint(self.paths["ckpt"] / name, self._blob())

    def try_resume(self):
        p = self.paths["ckpt"] / "latest.pt"
        if not p.exists():
            return False
        blob = torch.load(p, weights_only=False)
        if blob["run_id"] != self.run_id:
            raise RuntimeError("checkpoint belongs to a different configuration")
        self.model.load_state_dict(blob["model"]); self.opt.load_state_dict(blob["opt"])
        self.sched.load_state_dict(blob["sched"]); self.sampler.load_state_dict(blob["sampler"])
        self.weighting.load_state_dict(blob["weighting"]); self.acc = blob["acc"]
        self.history = blob["history"]; self.step = blob["step"]; self.status = blob["status"]
        self.rad_snapshots = blob.get("rad_snapshots", {})
        torch.set_rng_state(blob["torch_rng"])
        return True

    # ------------------------------------------------------------- one step
    def _grad_norms(self, parts):
        t0 = time.perf_counter()
        out = {}
        for k, v in parts.items():
            gs = torch.autograd.grad(v, self.params, retain_graph=True, allow_unused=True)
            out[f"gradnorm_{k}"] = math.sqrt(sum(float((g.double() ** 2).sum()) for g in gs if g is not None))
        self.acc["diag_gradients"] += len(parts)
        self.acc["diag_seconds"] += time.perf_counter() - t0
        return out

    def train_step(self, log_row=None):
        cfg = self.cfg
        batch = self.sampler.next_batch()
        res = term_residuals(self.model, batch, self.c2, self.gamma, cfg.loss.pde_scale)
        if self.weighting.due(self.step):
            t0 = time.perf_counter()
            rows = self.weighting.update(res, self.params)
            self.acc["ntk_row_gradients"] += rows
            self.acc["ntk_updates"] += 1
            self.acc["ntk_seconds"] += time.perf_counter() - t0
        parts = {k: reduce_loss(r, cfg.loss.reduction) for k, r in res.items()}
        lam = self.weighting.lam
        total = sum(lam[k] * parts[k] for k in parts)
        if log_row is not None and cfg.train.log_grad_norms:
            log_row.update(self._grad_norms(parts))
        if not torch.isfinite(total):
            return None, parts
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        if cfg.optim.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.params, cfg.optim.grad_clip)
        self.opt.step()
        self.sched.step()
        mb = cfg.sampler.mini_batch
        self.acc["optimizer_steps"] += 1
        self.acc["grad_evaluations"] += 1
        self.acc["training_points"] += mb * len(res)
        self.acc["pde_evaluations"] += sum(mb for b in batch.values() if b["spec"].kind == "pde")
        return float(total.detach()), parts

    # ------------------------------------------------------------- main loop
    def run(self, resume=True, max_wall_seconds=None, final_eval=True, stop_at_step=None):
        """stop_at_step / max_wall_seconds end a SEGMENT (status 'interrupted'); rerun to resume."""
        cfg = self.cfg
        cfg.to_json(self.paths["config"])
        write_json(self.paths["logs"] / "hardware.json", hardware_info())
        if resume:
            self.try_resume()
        if self.status in ("completed", "diverged"):
            return self.status
        reset_peak_rss()
        self.acc["segments"] += 1
        total_steps = cfg.total_steps()
        seg_start = time.perf_counter()
        self.status = "running"
        while self.step < total_steps:
            log_now = self.step == 0 or (self.step + 1) % cfg.train.log_every == 0
            row = {} if log_now else None
            diag0, t0 = self.acc["diag_seconds"], time.perf_counter()
            if (cfg.sampler.adaptive == "rad" and self.step > 0 and self.sampler.pos == 0
                    and self.step % cfg.sampler.rad.every == 0):
                n_c, stats, pts = rad_update(self.model, self.sampler, self.c2, self.gamma,
                                             cfg.sampler.rad, DTYPES[cfg.precision])
                self.acc["candidate_evaluations"] += n_c
                self.acc["rad_updates"] += 1
                self.acc["rad_seconds"] += stats["rad_seconds_update"]
                self.history.append({"step": self.step, **stats})
                self.rad_snapshots[self.step] = pts
            loss, parts = self.train_step(row)
            # gradient-norm logging is diagnostic, not training cost: exclude its time
            self.acc["train_seconds"] += time.perf_counter() - t0 - (self.acc["diag_seconds"] - diag0)
            if loss is None:
                self.status = "diverged"
                self.history.append({"step": self.step + 1, "status": "non-finite loss"})
                break
            self.step += 1
            last = self.step == total_steps
            if row is not None or last:
                row = row or {}
                row.update(step=self.step, epoch=self.sampler.epoch, loss=loss,
                           lr=self.opt.param_groups[0]["lr"], train_seconds=self.acc["train_seconds"],
                           pde_evaluations=self.acc["pde_evaluations"])
                row.update({f"L_{k}": float(v.detach()) for k, v in parts.items()})
                row.update({f"lam_{k}": v for k, v in self.weighting.lam.items()})
            if self.step % cfg.train.eval_every == 0 or last or self.step == 1:
                t1 = time.perf_counter()
                row = row if row is not None else {"step": self.step, "train_seconds": self.acc["train_seconds"]}
                row.update(validation_metrics(self.model, self.refs, self.bm.L, self.bm.t_end))
                self.acc["validation_seconds"] += time.perf_counter() - t1
            if row is not None:
                self.history.append(row)
            if self.step % cfg.train.ckpt_every == 0:
                self._snapshot()
            if cfg.train.snapshot_every and self.step % cfg.train.snapshot_every == 0:
                self.save(f"step_{self.step}.pt")             # analysis snapshot, no effect on training
            if (stop_at_step is not None and self.step >= stop_at_step and not last) or \
               (max_wall_seconds is not None and time.perf_counter() - seg_start > max_wall_seconds):
                self.status = "interrupted"
                break
        if self.status == "running":
            self.status = "completed"
        self._snapshot()
        if self.status in ("completed", "diverged") and final_eval:
            self.finalise()
        return self.status

    def _snapshot(self):
        self.acc["peak_rss_mb"] = max(self.acc["peak_rss_mb"], peak_rss_mb())
        self.save("latest.pt")
        write_history(self.paths["logs"] / "history.csv", self.history)

    def finalise(self):
        train_peak = max(self.acc["peak_rss_mb"], peak_rss_mb())   # training memory (this segment)
        reset_peak_rss()
        t0 = time.perf_counter()
        metrics, _ = evaluate_full(self.model, self.bm, self.refs, self.c2, self.gamma)
        eval_s = time.perf_counter() - t0
        eval_peak = peak_rss_mb()
        self.save("final.pt")
        bm = self.bm
        out = {"run_id": self.run_id, "name": self.cfg.name, "benchmark_id": bm.benchmark_id,
               "status": self.status, "seed": self.cfg.seed, "mode": bm.mode,
               "paper_L2_target": bm.paper_L2,
               "L2_paper_over_target": metrics["L2_paper"] / bm.paper_L2,
               "parameters": count_parameters(self.model), "model_size_bytes": model_size_bytes(self.model),
               "precision": self.cfg.precision, "device": self.cfg.device,
               "budget_label": self.cfg.train.budget_label, "final_eval_seconds": eval_s,
               "peak_vram_mb": peak_vram_mb(), **self.acc, **metrics}
        out["peak_rss_mb"] = train_peak          # TRAINING peak (reported as 'memory')
        out["peak_rss_eval_mb"] = eval_peak       # final-evaluation peak, reported separately
        write_json(self.paths["logs"] / "metrics.json", out)
        write_json(self.paths["logs"] / "runtime.json",
                   {"train_seconds": self.acc["train_seconds"], "ntk_seconds": self.acc["ntk_seconds"],
                    "validation_seconds": self.acc["validation_seconds"],
                    "diag_seconds": self.acc["diag_seconds"], "final_eval_seconds": eval_s,
                    "segments": self.acc["segments"]})
        return out
