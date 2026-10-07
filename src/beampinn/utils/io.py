"""Run-directory layout under results_optimization/ and safe writers.

    configs/<run_id>.json
    logs/<run_id>/{history.csv, metrics.json, hardware.json, runtime.json}
    checkpoints/<run_id>/{latest.pt, final.pt}

Nothing here ever writes into the legacy results/ tree.
"""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results_optimization"
LEGACY = REPO / "results"


def run_paths(run_id, root=RESULTS):
    root = Path(root)
    if root.resolve() == LEGACY.resolve() or LEGACY.resolve() in root.resolve().parents:
        raise PermissionError("refusing to write into the legacy results/ tree")
    p = {"config": root / "configs" / f"{run_id}.json",
         "logs": root / "logs" / run_id,
         "ckpt": root / "checkpoints" / run_id}
    for k in ("logs", "ckpt"):
        p[k].mkdir(parents=True, exist_ok=True)
    p["config"].parent.mkdir(parents=True, exist_ok=True)
    return p


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def write_history(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def save_checkpoint(path, blob):
    tmp = str(path) + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
