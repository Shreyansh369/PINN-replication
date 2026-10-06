#!/usr/bin/env bash
# Phase D: D1-D4 baseline-repair diagnostics, 5k steps, 4 concurrent 1-thread runs.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_optimization/logs/_console
for c in D1_unit D2_bias0 D3_unit_bias0 D4_rc_bias0; do
  python3 -I experiments/run.py --config configs/phaseD/$c.json --budget D5K --threads 1 \
    --log-every 250 --eval-every 500 --ckpt-every 2500 --status DIAGNOSTIC \
    > results_optimization/logs/_console/D_$c.log 2>&1 &
done
wait; echo "D done"; tail -n 1 results_optimization/logs/_console/D_*.log
