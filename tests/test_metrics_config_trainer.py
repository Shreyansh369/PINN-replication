"""Evaluation metrics, frequency extractor, config guards, and trainer SOFTWARE smoke tests.

The trainer tests take a handful of optimizer steps on a 2 x 12 network purely to check the
code path (files, accounting, resume). They are not experiments and produce no results.
"""
import json
import math

import numpy as np
import pytest
import torch

from beampinn.config import ExperimentConfig
from beampinn.evaluation.frequency import fit_damped_cosine, compare_fits
from beampinn.evaluation.metrics import evaluate_full, grid, l2_pair
from beampinn.physics.benchmarks import get_benchmark
from beampinn.training.trainer import Trainer
from beampinn.utils.io import LEGACY, run_paths


# ---------------------------------------------------------------- frequency fit
@pytest.mark.parametrize("kind", ["exact", "paper"])
def test_fit_recovers_reference_parameters(kind):
    c = get_benchmark("FE-D-M1").reference(kind)
    t = np.linspace(0, 1, 2001)
    fit = fit_damped_cosine(t, c.u(c.x_norm, t))
    a, lam, w, phi = c.damped_cosine_params()
    e = compare_fits(fit, a * c.A0, lam, w, phi)
    assert e["frequency_error"] < 1e-10 and e["phase_error"] < 1e-8
    assert e["amplitude_error"] < 1e-8 and e["damping_error"] < 1e-8


def test_fit_resolves_paper_vs_exact_frequency_offset():
    bm = get_benchmark("FE-D-M1")
    p, e = bm.reference("paper"), bm.reference("exact")
    t = np.linspace(0, 1, 2001)
    fit = fit_damped_cosine(t, p.u(p.x_norm, t))
    assert abs((fit["w"] - e.omega_d) / e.omega_d - (p.omega_d - e.omega_d) / e.omega_d) < 1e-9


def test_fit_undamped():
    c = get_benchmark("FE-U-M1").reference("exact")
    t = np.linspace(0, 1, 2001)
    fit = fit_damped_cosine(t, c.u(c.x_norm, t))
    assert abs(fit["w"] - c.omega) / c.omega < 1e-10 and abs(fit["lam"]) < 1e-9


# ------------------------------------------------------------------ L2 metrics
def test_l2_pair_and_floor():
    bm = get_benchmark("FE-D-M1")
    refs = bm.references()
    _, _, X, T = grid(bm.L, bm.t_end, 201, 2001)
    m = l2_pair(refs["exact"].u(X, T), refs, X, T)
    assert m["L2_exact"] == 0.0
    assert 3e-4 < m["L2_paper"] < 6e-4                 # floor: see reference_comparison table


def test_evaluate_full_on_exact_probe(probe_factory):
    bm = get_benchmark("FE-D-M1")
    refs = bm.references()
    probe = probe_factory(refs["exact"])
    met, _ = evaluate_full(probe, bm, refs, *bm.pde_coeffs())
    assert met["L2_exact"] < 1e-13 and met["finite"]
    assert met["PDE_residual_rel"] < 1e-10
    assert met["BC_error_max"] < 1e-9 and met["IC_error_max"] < 1e-12
    assert met["frequency_error_exact"] < 1e-10
    for k in ("L2_paper", "L2_late_paper", "L2_late_exact", "RMSE_exact", "phase_error_exact",
              "amplitude_error_exact", "inference_single_point_us"):
        assert k in met


# ---------------------------------------------------------------------- config
def test_config_roundtrip_and_key(tmp_path):
    c = ExperimentConfig()
    p = tmp_path / "c.json"
    c.to_json(p)
    d = ExperimentConfig.from_json(p)
    assert d == c and d.key() == c.key()
    e = ExperimentConfig(name="other", notes="x")
    e.train.log_every = 7
    assert e.key() == c.key()                          # labels / cadence do not change the key
    e.optim.lr = 1e-3
    assert e.key() != c.key()


@pytest.mark.parametrize("path,val", [("sampler.adaptive", "rar"), ("loss.weighting", "annealing"),
                                      ("loss.weighting", "gradnorm"), ("loss.hard_constraints", "other"),
                                      ("loss.mixed_formulation", True), ("optim.lbfgs_steps", 10)])
def test_unapproved_mechanisms_refuse_to_run(path, val):
    c = ExperimentConfig()
    sec, f = path.split(".")
    setattr(getattr(c, sec), f, val)
    with pytest.raises(NotImplementedError):
        c.validate()


def test_causal_is_rejected_and_mode_checked():
    c = ExperimentConfig(); c.loss.causal = True
    with pytest.raises(NotImplementedError):
        c.validate()
    c = ExperimentConfig(mode=2)
    with pytest.raises(ValueError):
        c.validate()


def test_legacy_results_tree_is_protected():
    with pytest.raises(PermissionError):
        run_paths("x", LEGACY)
    with pytest.raises(PermissionError):
        run_paths("x", LEGACY / "tables")


# ------------------------------------------------------------- trainer smoke
def _tiny(steps=6, ntk_every=2):
    c = ExperimentConfig(name="smoke")
    c.model.depth, c.model.width, c.model.m_fourier = 2, 12, 6
    c.sampler.n_per_term, c.sampler.mini_batch = 64, 32
    c.loss.ntk.every = ntk_every
    c.train.max_steps, c.train.log_every, c.train.eval_every, c.train.ckpt_every = steps, 2, 3, 2
    c.train.budget_label = "SMOKE"
    return c


def test_trainer_accounting_and_outputs(tmp_path):
    c = _tiny()
    tr = Trainer(c, root=tmp_path)
    assert tr.run() == "completed"
    met = json.loads((tmp_path / "logs" / c.run_id() / "metrics.json").read_text())
    assert met["optimizer_steps"] == 6 and met["grad_evaluations"] == 6
    assert met["pde_evaluations"] == 6 * 32 and met["training_points"] == 6 * 32 * 4
    assert met["ntk_updates"] == 3 and met["ntk_row_gradients"] == 3 * 4 * 32
    for k in ("L2_paper", "L2_exact", "L2_late_paper", "RMSE_paper", "frequency_error_exact",
              "phase_error_exact", "amplitude_error_exact", "PDE_residual_rel", "BC_error_max",
              "IC_error_max", "train_seconds", "peak_rss_mb", "parameters", "model_size_bytes",
              "inference_single_point_us", "paper_L2_target"):
        assert k in met, k
    assert met["paper_L2_target"] == 4.64e-4
    for f in ("history.csv", "hardware.json", "runtime.json"):
        assert (tmp_path / "logs" / c.run_id() / f).exists()
    assert (tmp_path / "checkpoints" / c.run_id() / "final.pt").exists()
    assert (tmp_path / "configs" / f"{c.run_id()}.json").exists()


def test_trainer_resume_is_exact(tmp_path):
    c = _tiny()
    a = Trainer(c, root=tmp_path / "a"); a.run(final_eval=False)
    b = Trainer(c, root=tmp_path / "b")
    assert b.run(final_eval=False, stop_at_step=3) == "interrupted" and b.step == 3
    b2 = Trainer(c, root=tmp_path / "b")
    assert b2.run(final_eval=False) == "completed" and b2.step == 6
    for p, q in zip(a.model.state_dict().values(), b2.model.state_dict().values()):
        assert torch.allclose(p, q, rtol=0, atol=1e-6)
    assert b2.acc["optimizer_steps"] == 6 and b2.acc["segments"] == 2


def test_leaderboard_row_reports_both_references(tmp_path):
    import csv, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from update_leaderboard import COLS, append_run
    c = _tiny(steps=2)
    Trainer(c, root=tmp_path).run()
    board = tmp_path / "board.csv"
    append_run(c.run_id(), "BASELINE", logs=tmp_path / "logs", board=board)
    append_run(c.run_id(), "BASELINE", logs=tmp_path / "logs", board=board)   # idempotent
    with open(board) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and list(rows[0]) == COLS
    met = json.loads((tmp_path / "logs" / c.run_id() / "metrics.json").read_text())
    assert float(rows[0]["L2"]) == met["L2_paper"] and float(rows[0]["L2_exact"]) == met["L2_exact"]
