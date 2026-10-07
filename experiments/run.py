"""Run one experiment from a JSON config.  (Stage 0.1: provided, NOT yet used for any run.)

    python experiments/run.py --config configs/phaseA/C0_paper.json --budget S1 [--seed 1234]
                              [--max-wall-hours 2] [--status CANDIDATE] [--dry-run]

Budgets (optimizer steps at the configured mini-batch):
    S0 = 200, S1 = 20 000, S2 = 100 000, PAPER = full epochs (45 000 x 20 = 9.0e5 for FE-D-M1)
Re-running the same command resumes from results_optimization/checkpoints/<run_id>/latest.pt.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from beampinn.config import ExperimentConfig  # noqa: E402

BUDGETS = {"S0": 200, "D5K": 5000, "E5K": 5000, "X5K": 5000, "Y5K": 5000, "Z5K": 5000, "S1": 20000, "S2": 100000, "PAPER": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", choices=list(BUDGETS), default="S0")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-wall-hours", type=float)
    ap.add_argument("--status", default="CANDIDATE")
    ap.add_argument("--threads", type=int, help="torch threads (part of the run key)")
    ap.add_argument("--log-every", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--ckpt-every", type=int)
    ap.add_argument("--snapshot-every", type=int)
    ap.add_argument("--no-leaderboard", action="store_true", help="sanity runs: do not rank")
    ap.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    a = ap.parse_args()
    cfg = ExperimentConfig.from_json(a.config)
    if a.seed is not None:
        cfg.seed = a.seed
    cfg.train.max_steps, cfg.train.budget_label = BUDGETS[a.budget], a.budget
    if a.threads is not None:
        cfg.threads = a.threads
    for k in ("log_every", "eval_every", "ckpt_every", "snapshot_every"):       # cadence only; not in the run key
        if getattr(a, k) is not None:
            setattr(cfg.train, k, getattr(a, k))
    cfg.validate()
    steps = cfg.total_steps()
    print(f"run_id {cfg.run_id()} | {a.budget}: {steps:,} steps x mini-batch {cfg.sampler.mini_batch} "
          f"= {steps * cfg.sampler.mini_batch:,} PDE evaluations")
    if a.dry_run:
        return
    from beampinn.training.trainer import Trainer
    from update_leaderboard import append_run
    tr = Trainer(cfg)
    status = tr.run(max_wall_seconds=None if a.max_wall_hours is None else a.max_wall_hours * 3600)
    print("status:", status)
    if status in ("completed", "diverged") and not a.no_leaderboard:
        append_run(cfg.run_id(), a.status if status == "completed" else "REJECTED")


if __name__ == "__main__":
    main()
