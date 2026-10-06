"""Append/refresh a run's row in optimization_leaderboard.csv from its metrics.json.

L2 is the PAPER-FAITHFUL relative L2 (primary gate); L2_exact is the exact-physics value.
Both are always written.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "optimization_leaderboard.csv"
LOGS = ROOT / "results_optimization" / "logs"
COLS = ["run_id", "benchmark_id", "method", "seed", "mode", "L2", "L2_exact", "L2_late", "RMSE",
        "frequency_error", "phase_error", "amplitude_error", "PDE_residual", "IC_error", "BC_error",
        "optimizer_steps", "training_points", "PDE_evaluations", "candidate_evaluations", "wall_time",
        "peak_RAM", "peak_VRAM", "parameters", "model_size", "inference_latency", "device",
        "precision", "status", "notes"]


def row_from_metrics(m, status, notes=""):
    return {"run_id": m["run_id"], "benchmark_id": m["benchmark_id"], "method": m["name"],
            "seed": m["seed"], "mode": m["mode"], "L2": m["L2_paper"], "L2_exact": m["L2_exact"],
            "L2_late": m["L2_late_paper"], "RMSE": m["RMSE_paper"],
            "frequency_error": m["frequency_error_exact"], "phase_error": m["phase_error_exact"],
            "amplitude_error": m["amplitude_error_exact"], "PDE_residual": m["PDE_residual_rel"],
            "IC_error": m["IC_error_max"], "BC_error": m["BC_error_max"],
            "optimizer_steps": m["optimizer_steps"], "training_points": m["training_points"],
            "PDE_evaluations": m["pde_evaluations"], "candidate_evaluations": m["candidate_evaluations"],
            "wall_time": m["train_seconds"], "peak_RAM": m["peak_rss_mb"], "peak_VRAM": m["peak_vram_mb"],
            "parameters": m["parameters"], "model_size": m["model_size_bytes"],
            "inference_latency": m["inference_single_point_us"], "device": m["device"],
            "precision": m["precision"], "status": status,
            "notes": notes or f"budget={m['budget_label']}; ntk_s={m['ntk_seconds']:.1f}"}


def append_run(run_id, status, notes="", logs=LOGS, board=BOARD):
    m = json.loads((Path(logs) / run_id / "metrics.json").read_text())
    rows = []
    if Path(board).exists():
        with open(board) as f:
            rows = [r for r in csv.DictReader(f) if r["run_id"] != run_id]
    rows.append(row_from_metrics(m, status, notes))
    with open(board, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) == 1:                       # (re)write the header only
        with open(BOARD, "w", newline="") as f:
            csv.writer(f).writerow(COLS)
    else:
        append_run(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]))
